"""
Non-LLM learning-based planner baselines for LLM-DG benchmark.

This module provides behavioral cloning (BC) baselines that map kinematic state
representations to continuous acceleration outputs, evaluated under the same
metrics as LLM-based planners.
"""

from .dataset import BCDataset, get_train_val_test_loaders, StandardScaler
from .bc_mlp import BCMLP, create_mlp_model
from .bc_transformer import BCTransformer, create_transformer_model
from .utils import set_seed, save_checkpoint, load_checkpoint, count_parameters

__all__ = [
    "BCDataset",
    "get_train_val_test_loaders",
    "StandardScaler",
    "BCMLP",
    "BCTransformer",
    "create_mlp_model",
    "create_transformer_model",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
]
