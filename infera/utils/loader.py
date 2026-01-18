import os
from glob import glob
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import nn
from safetensors import safe_open


# ================================
# Infera-style Config
# ================================

@dataclass(frozen=True)
class WeightLoadConfig:
    """
    Controls how model weights are loaded.

    Design goals:
    - Explicit behavior
    - Safe defaults
    - Extensible for sharding / quant / offload
    """
    strict: bool = True
    device: str = "cpu"
    dtype: Optional[torch.dtype] = None


# ================================
# Weight Loader Interfaces
# ================================

def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
):
    param.data.copy_(loaded_weight)


def shard_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int,
):
    """
    Loader for packed / sharded parameters.
    """
    param.weight_loader(param, loaded_weight, shard_id)


# ================================
# Core Load Logic
# ================================

def load_model(
    model: nn.Module,
    path: str,
    config: WeightLoadConfig = WeightLoadConfig(),
):
    """
    Infera-style model loader.

    Features:
    - Packed module support
    - Config-driven behavior
    - Explicit error handling
    """
    packed_modules_mapping: Dict[str, tuple] = getattr(
        model, "packed_modules_mapping", {}
    )

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model path not found: {path}")

    weight_files = glob(os.path.join(path, "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")

    for file in weight_files:
        _load_single_file(model, file, packed_modules_mapping, config)


def _load_single_file(
    model: nn.Module,
    file: str,
    packed_modules_mapping: Dict[str, tuple],
    config: WeightLoadConfig,
):
    with safe_open(file, framework="pt", device=config.device) as f:
        for weight_name in f.keys():
            weight = f.get_tensor(weight_name)

            if config.dtype is not None:
                weight = weight.to(config.dtype)

            # Try packed mapping first
            handled = _try_load_packed_weight(
                model, weight_name, weight, packed_modules_mapping
            )

            if handled:
                continue

            # Fallback: normal parameter
            _load_regular_weight(
                model, weight_name, weight, config.strict
            )


def _try_load_packed_weight(
    model: nn.Module,
    weight_name: str,
    weight: torch.Tensor,
    packed_modules_mapping: Dict[str, tuple],
) -> bool:
    """
    Returns True if weight is handled as a packed parameter.
    """
    for packed_key, (target_key, shard_id) in packed_modules_mapping.items():
        if packed_key in weight_name:
            param_name = weight_name.replace(packed_key, target_key)

            try:
                param = model.get_parameter(param_name)
            except KeyError:
                raise KeyError(f"Packed parameter not found: {param_name}")

            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, weight, shard_id)
            return True

    return False


def _load_regular_weight(
    model: nn.Module,
    weight_name: str,
    weight: torch.Tensor,
    strict: bool,
):
    try:
        param = model.get_parameter(weight_name)
    except KeyError:
        if strict:
            raise KeyError(f"Unexpected weight: {weight_name}")
        return

    weight_loader: Callable = getattr(
        param, "weight_loader", default_weight_loader
    )
    weight_loader(param, weight)
