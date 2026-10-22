# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.adapters import DenseMixtureOfConvolutionAdapter
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


DEFAULT_DENSE_ADAPTER_LAYERS = (4, 11, 17, 23)


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        dense_adapter_enabled: bool = False,
        dense_adapter_dim: int = 64,
        dense_adapter_layers: tuple[int, ...] | list[int] | None = None,
        dense_adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        self.patch_size = patch_size
        self.dense_adapter_enabled = bool(dense_adapter_enabled)
        self.dense_adapter_layers = _normalize_dense_adapter_layers(dense_adapter_layers)
        if self.dense_adapter_enabled and self.dense_head is None:
            raise ValueError("dense_adapter_enabled=True requires enable_depth=True")
        if self.dense_adapter_enabled and not self.dense_adapter_layers:
            raise ValueError("dense_adapter_layers must contain at least one layer when adapters are enabled")
        self.dense_adapters = nn.ModuleDict(
            {
                str(layer_idx): DenseMixtureOfConvolutionAdapter(
                    token_dim=2 * embed_dim,
                    adapter_dim=dense_adapter_dim,
                    residual_scale=dense_adapter_scale,
                )
                for layer_idx in self.dense_adapter_layers
            }
            if self.dense_adapter_enabled
            else {}
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                dense_tokens_list = aggregated_tokens_list
                if self.dense_adapter_enabled:
                    dense_tokens_list = self._adapt_dense_tokens(
                        aggregated_tokens_list,
                        images=images,
                        patch_token_start=patch_token_start,
                    )
                depth, depth_conf = self.dense_head(
                    dense_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions

    def _adapt_dense_tokens(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        *,
        images: torch.Tensor,
        patch_token_start: int,
    ) -> list[torch.Tensor | None]:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, S, 3, H, W], got {tuple(images.shape)}")
        patch_grid = (images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size)
        adapted_tokens_list = list(aggregated_tokens_list)
        for layer_idx in self.dense_adapter_layers:
            if layer_idx >= len(aggregated_tokens_list):
                raise ValueError(
                    f"dense adapter layer {layer_idx} exceeds cached token list length "
                    f"{len(aggregated_tokens_list)}"
                )
            tokens = aggregated_tokens_list[layer_idx]
            if tokens is None:
                raise ValueError(f"dense adapter layer {layer_idx} was not cached by Aggregator")
            adapted_tokens_list[layer_idx] = self.dense_adapters[str(layer_idx)](
                tokens,
                patch_token_start=patch_token_start,
                patch_grid=patch_grid,
            )
        return adapted_tokens_list


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )


def _normalize_dense_adapter_layers(layers: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if layers is None:
        return DEFAULT_DENSE_ADAPTER_LAYERS
    normalized = tuple(int(layer_idx) for layer_idx in layers)
    if any(layer_idx < 0 for layer_idx in normalized):
        raise ValueError(f"dense_adapter_layers must be non-negative, got {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"dense_adapter_layers must be unique, got {normalized}")
    return normalized
