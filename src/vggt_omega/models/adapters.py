# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import torch
import torch.nn as nn


class DenseMixtureOfConvolutionAdapter(nn.Module):
    """Patch-token-only Dense Mixture of Convolution adapter for dense geometry."""

    min_adapter_dim = 3

    def __init__(
        self,
        token_dim: int = 2048,
        adapter_dim: int = 64,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        token_dim = int(token_dim)
        adapter_dim = int(adapter_dim)
        residual_scale = float(residual_scale)
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if adapter_dim < self.min_adapter_dim:
            raise ValueError(
                "dense adapter_dim must be at least "
                f"{self.min_adapter_dim}, got {adapter_dim}"
            )
        if residual_scale < 0:
            raise ValueError(
                f"dense adapter residual_scale must be non-negative, got {residual_scale}"
            )

        branch_widths = _split_widths_exact(adapter_dim, 3)
        branch_1_width, branch_2_width, branch_3_width = branch_widths

        self.token_dim = token_dim
        self.adapter_dim = adapter_dim
        self.branch_widths = branch_widths
        self.residual_scale = residual_scale

        self.norm = nn.LayerNorm(token_dim, eps=1e-5)
        self.down = nn.Linear(token_dim, adapter_dim)
        self.branch_1x1 = nn.Conv2d(adapter_dim, branch_1_width, kernel_size=1)
        self.branch_3x3_reduce = nn.Conv2d(
            adapter_dim + branch_1_width,
            branch_2_width,
            kernel_size=1,
        )
        self.branch_3x3 = nn.Conv2d(
            branch_2_width,
            branch_2_width,
            kernel_size=3,
            padding=1,
        )
        self.branch_5x5_reduce = nn.Conv2d(
            adapter_dim + branch_1_width + branch_2_width,
            branch_3_width,
            kernel_size=1,
        )
        self.branch_5x5 = nn.Conv2d(
            branch_3_width,
            branch_3_width,
            kernel_size=5,
            padding=2,
        )
        self.activation = nn.GELU()
        self.up = nn.Linear(adapter_dim, token_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        patch_token_start: int,
        patch_grid: Sequence[int],
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(
                "DenseMixtureOfConvolutionAdapter expects cached tokens with shape "
                f"[B, S, N, C], got {tuple(tokens.shape)}"
            )
        batch_size, num_frames, num_tokens, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(
                f"Dense adapter token_dim mismatch: expected {self.token_dim}, got {token_dim}"
            )
        if patch_token_start < 0 or patch_token_start > num_tokens:
            raise ValueError(
                f"patch_token_start ({patch_token_start}) must be between 0 and token length ({num_tokens})"
            )
        if len(patch_grid) != 2:
            raise ValueError(f"patch_grid must be a length-2 sequence, got {patch_grid!r}")
        patch_h, patch_w = int(patch_grid[0]), int(patch_grid[1])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_grid entries must be positive, got {patch_h}x{patch_w}")

        patch_token_count = num_tokens - patch_token_start
        expected_patch_tokens = patch_h * patch_w
        if patch_token_count != expected_patch_tokens:
            raise ValueError(
                f"patch token count ({patch_token_count}) does not match patch grid "
                f"{patch_h}x{patch_w} ({expected_patch_tokens})"
            )

        patch_tokens = tokens[:, :, patch_token_start:]
        patch_tokens_float = patch_tokens.float() if patch_tokens.dtype != torch.float32 else patch_tokens

        bottleneck = self.down(self.norm(patch_tokens_float))
        bottleneck_2d = bottleneck.reshape(
            batch_size * num_frames,
            patch_h,
            patch_w,
            self.adapter_dim,
        ).permute(0, 3, 1, 2)

        branch_1 = self.activation(self.branch_1x1(bottleneck_2d))
        branch_2_input = torch.cat([bottleneck_2d, branch_1], dim=1)
        branch_2 = self.activation(self.branch_3x3_reduce(branch_2_input))
        branch_2 = self.activation(self.branch_3x3(branch_2))
        branch_3_input = torch.cat([bottleneck_2d, branch_1, branch_2], dim=1)
        branch_3 = self.activation(self.branch_5x5_reduce(branch_3_input))
        branch_3 = self.activation(self.branch_5x5(branch_3))

        mixed = torch.cat([branch_1, branch_2, branch_3], dim=1)
        if mixed.shape[1] != self.adapter_dim:
            raise AssertionError(
                f"dense adapter branch widths sum to {mixed.shape[1]}, expected {self.adapter_dim}"
            )
        mixed = self.activation(mixed) + bottleneck_2d
        mixed = mixed.permute(0, 2, 3, 1).reshape(
            batch_size,
            num_frames,
            patch_token_count,
            self.adapter_dim,
        )

        patch_update = self.up(mixed) * self.residual_scale
        adapted_patch_tokens = patch_tokens_float + patch_update
        adapted_tokens = tokens.clone()
        adapted_tokens[:, :, patch_token_start:] = adapted_patch_tokens.to(dtype=tokens.dtype)
        return adapted_tokens


def _split_widths_exact(total_width: int, parts: int) -> tuple[int, ...]:
    base_width, remainder = divmod(total_width, parts)
    widths = tuple(base_width + (1 if index < remainder else 0) for index in range(parts))
    if sum(widths) != total_width or any(width <= 0 for width in widths):
        raise AssertionError(f"invalid branch width split for total_width={total_width}: {widths}")
    return widths


_DENSE_ADAPTER_MARKER = "dense_adapters."
_DENSE_ADAPTER_EXPECTED_KEYS = {
    "norm.weight",
    "norm.bias",
    "down.weight",
    "down.bias",
    "branch_1x1.weight",
    "branch_1x1.bias",
    "branch_3x3_reduce.weight",
    "branch_3x3_reduce.bias",
    "branch_3x3.weight",
    "branch_3x3.bias",
    "branch_5x5_reduce.weight",
    "branch_5x5_reduce.bias",
    "branch_5x5.weight",
    "branch_5x5.bias",
    "up.weight",
    "up.bias",
}


def infer_dense_adapter_config_from_state_dict(
    state_dict: Mapping[str, object],
    *,
    dense_adapter_scale: float = 1.0,
) -> dict[str, object]:
    """Infer VGGT-Omega dense adapter constructor kwargs from raw weights.

    The residual scale is not serialized in the module state, so callers must
    provide an explicit override or accept the export/runtime default of 1.0.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"state_dict must be a mapping, got {type(state_dict).__name__}")
    dense_adapter_scale = float(dense_adapter_scale)
    if dense_adapter_scale < 0:
        raise ValueError(f"dense_adapter_scale must be non-negative, got {dense_adapter_scale}")

    layer_states: dict[int, dict[str, object]] = {}
    malformed_keys = []
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        marker_index = key.find(_DENSE_ADAPTER_MARKER)
        if marker_index < 0:
            continue
        prefix = key[:marker_index]
        if prefix and not prefix.endswith("."):
            continue
        suffix = key[marker_index + len(_DENSE_ADAPTER_MARKER) :]
        layer_text, separator, parameter_name = suffix.partition(".")
        if not separator or not layer_text.isdecimal() or not parameter_name:
            malformed_keys.append(key)
            continue
        layer_states.setdefault(int(layer_text), {})[parameter_name] = value

    if malformed_keys:
        sample = ", ".join(sorted(malformed_keys)[:5])
        raise ValueError(f"malformed dense adapter state key(s): {sample}")
    if not layer_states:
        return {"dense_adapter_enabled": False}

    adapter_dim = None
    token_dim = None
    for layer_idx in sorted(layer_states):
        layer_adapter_dim, layer_token_dim = _validate_dense_adapter_layer_state(
            layer_idx,
            layer_states[layer_idx],
        )
        if adapter_dim is None:
            adapter_dim = layer_adapter_dim
            token_dim = layer_token_dim
        elif adapter_dim != layer_adapter_dim or token_dim != layer_token_dim:
            raise ValueError(
                "inconsistent dense adapter dimensions: "
                f"layer {layer_idx} has adapter_dim={layer_adapter_dim}, token_dim={layer_token_dim}; "
                f"expected adapter_dim={adapter_dim}, token_dim={token_dim}"
            )

    return {
        "dense_adapter_enabled": True,
        "dense_adapter_layers": tuple(sorted(layer_states)),
        "dense_adapter_dim": adapter_dim,
        "dense_adapter_scale": dense_adapter_scale,
    }


def _validate_dense_adapter_layer_state(
    layer_idx: int,
    layer_state: Mapping[str, object],
) -> tuple[int, int]:
    keys = set(layer_state)
    missing = _DENSE_ADAPTER_EXPECTED_KEYS - keys
    extra = keys - _DENSE_ADAPTER_EXPECTED_KEYS
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"unexpected={sorted(extra)}")
        raise ValueError(
            f"partial or malformed dense adapter state for layer {layer_idx}: "
            + ", ".join(details)
        )

    down_weight_shape = _tensor_shape(layer_idx, "down.weight", layer_state["down.weight"])
    if len(down_weight_shape) != 2:
        raise ValueError(
            f"dense adapter layer {layer_idx} down.weight must be rank 2, "
            f"got shape {down_weight_shape}"
        )
    adapter_dim, token_dim = down_weight_shape
    if adapter_dim < DenseMixtureOfConvolutionAdapter.min_adapter_dim:
        raise ValueError(
            f"dense adapter layer {layer_idx} adapter_dim must be at least "
            f"{DenseMixtureOfConvolutionAdapter.min_adapter_dim}, got {adapter_dim}"
        )
    branch_1_width, branch_2_width, branch_3_width = _split_widths_exact(adapter_dim, 3)

    expected_shapes = {
        "norm.weight": (token_dim,),
        "norm.bias": (token_dim,),
        "down.bias": (adapter_dim,),
        "branch_1x1.weight": (branch_1_width, adapter_dim, 1, 1),
        "branch_1x1.bias": (branch_1_width,),
        "branch_3x3_reduce.weight": (branch_2_width, adapter_dim + branch_1_width, 1, 1),
        "branch_3x3_reduce.bias": (branch_2_width,),
        "branch_3x3.weight": (branch_2_width, branch_2_width, 3, 3),
        "branch_3x3.bias": (branch_2_width,),
        "branch_5x5_reduce.weight": (
            branch_3_width,
            adapter_dim + branch_1_width + branch_2_width,
            1,
            1,
        ),
        "branch_5x5_reduce.bias": (branch_3_width,),
        "branch_5x5.weight": (branch_3_width, branch_3_width, 5, 5),
        "branch_5x5.bias": (branch_3_width,),
        "up.weight": (token_dim, adapter_dim),
        "up.bias": (token_dim,),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = _tensor_shape(layer_idx, name, layer_state[name])
        if actual_shape != expected_shape:
            raise ValueError(
                f"dense adapter layer {layer_idx} {name} has shape {actual_shape}; "
                f"expected {expected_shape}"
            )
    return int(adapter_dim), int(token_dim)


def _tensor_shape(layer_idx: int, name: str, value: object) -> tuple[int, ...]:
    if not torch.is_tensor(value):
        raise TypeError(
            f"dense adapter layer {layer_idx} {name} must be a tensor, "
            f"got {type(value).__name__}"
        )
    return tuple(int(dim) for dim in value.shape)
