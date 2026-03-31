import math
import os
import torch
import torch.nn as nn
from typing import Dict, List
from commons.datasets.hstu_batch import HSTUBatch
from configs import (
    InferenceHSTUConfig,
    KVCacheConfig,
    RankingConfig,
    copy_kvcache_metadata,
    get_kvcache_metadata_buffer,
)
from modules.hstu_block_inference import HSTUBlockInference
from modules.jagged_data import JaggedData
from modules.mlp import MLP
from torchrec.sparse.jagged_tensor import JaggedTensor

def get_jagged_metadata_buffer(max_batch_size, max_seq_len, contextual_max_seqlen):
    int_dtype = torch.int32
    device = torch.cuda.current_device()
    return JaggedData(
        values=None,
        max_seqlen=max_seq_len,
        seqlen=torch.full((max_batch_size,), max_seq_len, dtype=int_dtype, device=device),
        seqlen_offsets=torch.arange(end=max_batch_size + 1, dtype=int_dtype, device=device) * max_seq_len,
        max_num_candidates=max_seq_len // 2,
        num_candidates=torch.full((max_batch_size,), max_seq_len // 2, dtype=int_dtype, device=device),
        num_candidates_offsets=torch.arange(end=max_batch_size + 1, dtype=int_dtype, device=device) * (max_seq_len // 2),
        contextual_max_seqlen=contextual_max_seqlen,
        contextual_seqlen=None,
        contextual_seqlen_offsets=None,
        has_interleaved_action=True,
        scaling_seqlen=-1,
    )

class InferenceDenseModule(nn.Module):
    def __init__(self, hstu_config, kvcache_config, task_config, use_cudagraph=False, cudagraph_configs=None):
        super().__init__()
        self._device = torch.cuda.current_device()
        self._hstu_config = hstu_config
        self._embedding_dim = hstu_config.hidden_size
        self._hstu_block = HSTUBlockInference(hstu_config, kvcache_config).cuda()
        self._mlp = MLP(hstu_config.hidden_size, task_config.prediction_head_arch, task_config.prediction_head_act_type, task_config.prediction_head_bias, device=self._device).cuda()
        
        # Blackwell BF16 Alignment
        self.to(torch.bfloat16)
        self._hidden_states = torch.zeros((hstu_config.max_batch_size * hstu_config.max_seq_len, hstu_config.hidden_size), dtype=torch.bfloat16, device=self._device)
        self._jagged_metadata = get_jagged_metadata_buffer(hstu_config.max_batch_size, hstu_config.max_seq_len, hstu_config.contextual_max_seqlen)
        self._kvcache_metadata = get_kvcache_metadata_buffer(hstu_config=hstu_config, kvcache_config=kvcache_config)

        if use_cudagraph and cudagraph_configs:
            print("Setting up cuda graphs ...")
            self._hstu_block.set_cudagraph(
                hstu_config.max_batch_size, 
                hstu_config.max_seq_len, 
                self._hidden_states, 
                self._jagged_metadata, 
                None, 
                cudagraph_configs=cudagraph_configs
            )

    def forward_nokvcache(self, batch, embeddings):
        feat = embeddings.get("item_feat", list(embeddings.values())[0])
        val = feat.values() if hasattr(feat, "values") and callable(feat.values) else feat.values
        
        # Original multi-bucket logic: The block chooses the graph based on input size
        self._hidden_states[:val.shape[0]].copy_(val, non_blocking=True)
        # This call inside hstu_block handles the bucket selection
        _out = self._hstu_block.predict(batch.batch_size, val.shape[0], self._hidden_states, self._jagged_metadata, None)
        return self._mlp(_out)

    def forward(self, batch, embeddings, *args, **kwargs):
        return self.forward_nokvcache(batch, embeddings)

    def load_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "torch_module", "model.0.pth")
        if os.path.exists(path):
            sd = torch.load(path, map_location="cpu")["model_state_dict"]
            new_sd = {k.replace("dense_module.", ""): v for k, v in sd.items() if "embedding" not in k}
            self.load_state_dict(new_sd, strict=False)
            self.to(torch.bfloat16)

def copy_jagged_metadata(dst_metadata, src_metata):
    def copy_tensor(dst, src):
        if src is None: return
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = 0
    def copy_offsets(dst, src):
        if src is None: return
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = src[-1, ...]
    sl = getattr(src_metata, "seqlen", None)
    if sl is None and hasattr(src_metata, "lengths"):
        sl = src_metata.lengths() if callable(src_metata.lengths) else src_metata.lengths
    so = getattr(src_metata, "seqlen_offsets", None)
    if so is None and hasattr(src_metata, "offsets"):
        so = src_metata.offsets() if callable(src_metata.offsets) else src_metata.offsets
    bs = sl.shape[0] if sl is not None else 1
    dst_metadata.max_seqlen = src_metata.max_seqlen
    if sl is not None: copy_tensor(dst_metadata.seqlen, sl[:bs])
    if so is not None: copy_offsets(dst_metadata.seqlen_offsets, so[: bs + 1])
    dst_metadata.max_num_candidates = getattr(src_metata, "max_num_candidates", 0)
    nc = getattr(src_metata, "num_candidates", None)
    if nc is not None: copy_tensor(dst_metadata.num_candidates, nc[:bs])
    dst_metadata.scaling_seqlen = getattr(src_metata, "scaling_seqlen", -1)
