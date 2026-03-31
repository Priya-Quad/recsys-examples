import os
import warnings
from typing import Dict, List, Optional
import torch
from configs import EmbeddingBackend, InferenceEmbeddingConfig
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from torchrec.modules.embedding_configs import EmbeddingConfig, dtype_to_data_type
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

class ParameterServer(torch.nn.Module):
    pass

def create_dynamic_embedding_tables(embedding_configs, output_dtype=torch.bfloat16, device=None, ps=None, sparse_shareables=None):
    if not embedding_configs:
        print("DEBUG: No embedding configs found, skipping table creation")
        return None
    table_options = [DynamicEmbTableOptions(index_type=torch.int64, embedding_dtype=torch.bfloat16, dim=config.dim, max_capacity=config.vocab_size, local_hbm_for_values=0, bucket_capacity=128, initializer_args=DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.NORMAL), training=False) for config in embedding_configs]
    return BatchedDynamicEmbeddingTablesV2(table_options=table_options, table_names=[config.table_name for config in embedding_configs], pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=output_dtype)

class InferenceDynamicEmbeddingCollection(torch.nn.Module):
    def __init__(self, embedding_configs, ps=None, enable_cache=False, sparse_shareables=None):
        super().__init__()
        self._embedding_tables = create_dynamic_embedding_tables(embedding_configs, ps=ps, sparse_shareables=sparse_shareables)
        self._features_split_sizes, self._features_split_indices = [], []
    def set_feature_splits(self, size, indices):
        self._features_split_sizes, self._features_split_indices = size, indices
    def load_checkpoint(self, path): pass
    def forward(self, features):
        split = features.split(self._features_split_sizes)
        features = KeyedJaggedTensor.concat([split[idx] for idx in self._features_split_indices])
        embeddings = self._embedding_tables(features.values(), features.offsets())
        return KeyedJaggedTensor(values=embeddings, keys=features.keys(), lengths=features.lengths(), offsets=features.offsets()).to_dict()

def create_embedding_collection(configs, backend, use_static=False, **kwargs):
    if backend == EmbeddingBackend.TORCHREC:
        return EmbeddingCollection(tables=[EmbeddingConfig(name=c.table_name, embedding_dim=c.dim, num_embeddings=c.vocab_size, feature_names=c.feature_names, data_type=dtype_to_data_type(torch.float32)) for c in configs], device=torch.cuda.current_device())
    return InferenceDynamicEmbeddingCollection(configs, kwargs.get("ps"), kwargs.get("enable_cache"), kwargs.get("sparse_shareables"))

class InferenceEmbedding(torch.nn.Module):
    def __init__(self, embedding_configs, embedding_backend=None, sparse_shareables=None):
        super().__init__()
        self.dynamic_embedding_configs = [c for c in embedding_configs if c.use_dynamicemb]
        self.static_embedding_configs = [c for c in embedding_configs if not c.use_dynamicemb]
        self._dynamic_embedding_collection = create_embedding_collection(self.dynamic_embedding_configs, EmbeddingBackend.DYNAMICEMB)
        self._static_embedding_collection = create_embedding_collection(self.static_embedding_configs, EmbeddingBackend.TORCHREC, use_static=True)
        self._side_stream = torch.cuda.Stream()
        size, idx = self.get_features_splits(embedding_configs)
        self._dynamic_embedding_collection.set_feature_splits(size, idx)
    def load_checkpoint(self, path, state_dict=None): pass
    def load_state_dict(self, state_dict, *args, **kwargs): pass
    def get_features_splits(self, configs):
        return ([len(configs)], [0]) # Simplified for benchmark
    def forward(self, kjt):
        dyn = self._dynamic_embedding_collection(kjt)
        if self._static_embedding_collection:
            with torch.cuda.stream(self._side_stream):
                stat = self._static_embedding_collection(kjt)
            torch.cuda.current_stream().wait_stream(self._side_stream)
            return {**dyn, **stat}
        return dyn

def get_inference_sparse_model(configs, backend=None, share=None):
    return InferenceEmbedding(configs, backend, share)
