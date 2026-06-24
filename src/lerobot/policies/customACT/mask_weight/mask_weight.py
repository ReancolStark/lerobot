import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

from lerobot.policies.customACT.mask_weight.configuration_mask_weight import MaskWeightConfig


def _odd_kernel_size(kernel_size: int) -> int:
    kernel_size = max(1, int(kernel_size))
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def yolo_result_to_soft_mask(
    results: Union[List, object],
    kernel_size: int = 7,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Convert Ultralytics YOLO results to a confidence-weighted soft mask.

    Returns:
        Tensor with shape [B, 1, H, W]. Instance masks are preferred; boxes are
        used as a fallback when segmentation masks are unavailable.
    """
    if not isinstance(results, list):
        results = [results]

    if len(results) == 0:
        raise ValueError("YOLO results must contain at least one result.")

    height, width = results[0].orig_shape
    soft_masks = []
    kernel_size = _odd_kernel_size(kernel_size)

    for result in results:
        if result.masks is not None:
            device = result.masks.data.device
        elif result.boxes is not None:
            device = result.boxes.conf.device
        else:
            device = torch.device("cpu")

        hard_mask = torch.zeros((1, height, width), dtype=torch.float32, device=device)

        if result.masks is not None and result.boxes is not None and len(result.masks.data) > 0:
            masks = result.masks.data.float()
            confs = result.boxes.conf.to(device=device, dtype=torch.float32)

            for i in range(len(masks)):
                mask_i = F.interpolate(
                    masks[i].unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                hard_mask[0] = torch.maximum(hard_mask[0], mask_i * confs[i])

        elif result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                conf = boxes.conf[i].to(device=device, dtype=torch.float32)
                x1, y1, x2, y2 = boxes.xyxy[i].detach().round().to(torch.int64).tolist()

                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))
                if x2 > x1 and y2 > y1:
                    hard_mask[0, y1:y2, x1:x2] = torch.maximum(hard_mask[0, y1:y2, x1:x2], conf)

        if hard_mask.max() > 0 and kernel_size > 1:
            blur = GaussianBlur(kernel_size=kernel_size, sigma=float(sigma))
            soft_mask = blur(hard_mask.unsqueeze(0)).squeeze(0)
            soft_mask = torch.clamp(soft_mask, 0.0, 1.0)
        else:
            soft_mask = hard_mask

        soft_masks.append(soft_mask)

    return torch.stack(soft_masks, dim=0)


def yolo_results_to_object_weight_inputs(
    results: Union[List, object],
    max_instances: int,
    kernel_size: int = 7,
    sigma: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Convert YOLO results into fixed-size per-instance inputs for ObjectWeightNet."""
    if not isinstance(results, list):
        results = [results]

    if len(results) == 0:
        raise ValueError("YOLO results must contain at least one result.")

    max_instances = max(0, int(max_instances))
    height, width = results[0].orig_shape
    kernel_size = _odd_kernel_size(kernel_size)

    first_result = results[0]
    if first_result.masks is not None:
        default_device = first_result.masks.data.device
    elif first_result.boxes is not None:
        default_device = first_result.boxes.conf.device
    else:
        default_device = torch.device("cpu")

    masks_out = torch.zeros(
        len(results),
        max_instances,
        1,
        height,
        width,
        dtype=torch.float32,
        device=default_device,
    )
    class_ids_out = torch.zeros(
        len(results),
        max_instances,
        dtype=torch.long,
        device=default_device,
    )
    confidences_out = torch.zeros(
        len(results),
        max_instances,
        dtype=torch.float32,
        device=default_device,
    )
    valid_out = torch.zeros(
        len(results),
        max_instances,
        dtype=torch.float32,
        device=default_device,
    )
    if max_instances == 0:
        return {
            "instance_masks": masks_out,
            "class_ids": class_ids_out,
            "confidences": confidences_out,
            "valid": valid_out,
        }

    blur = GaussianBlur(kernel_size=kernel_size, sigma=float(sigma)) if kernel_size > 1 else None

    for batch_idx, result in enumerate(results):
        if result.masks is not None:
            device = result.masks.data.device
        elif result.boxes is not None:
            device = result.boxes.conf.device
        else:
            device = default_device

        num_masks = 0 if result.masks is None else len(result.masks.data)
        num_boxes = 0 if result.boxes is None else len(result.boxes)
        num_candidates = max(num_masks, num_boxes)
        candidates: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for i in range(num_candidates):
            if result.boxes is not None and i < num_boxes:
                conf = result.boxes.conf[i].to(device=device, dtype=torch.float32)
                cls_id = result.boxes.cls[i].to(device=device, dtype=torch.long)
            else:
                conf = torch.ones((), dtype=torch.float32, device=device)
                cls_id = torch.zeros((), dtype=torch.long, device=device)

            mask_i = torch.zeros((1, height, width), dtype=torch.float32, device=device)
            if result.masks is not None and i < num_masks:
                raw_mask = result.masks.data[i].to(device=device, dtype=torch.float32)
                mask_i = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                mask_i = torch.clamp(mask_i * conf, 0.0, 1.0)
            elif result.boxes is not None and i < num_boxes:
                x1, y1, x2, y2 = result.boxes.xyxy[i].detach().round().to(torch.int64).tolist()
                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))
                if x2 > x1 and y2 > y1:
                    mask_i[:, y1:y2, x1:x2] = conf

            if mask_i.max() > 0:
                candidates.append((float(conf.detach().item()), conf, cls_id, mask_i))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for out_idx, (_, conf, cls_id, mask_i) in enumerate(candidates[:max_instances]):
            if blur is not None:
                mask_i = blur(mask_i.unsqueeze(0)).squeeze(0)
            masks_out[batch_idx, out_idx] = torch.clamp(mask_i.to(default_device), 0.0, 1.0)
            class_ids_out[batch_idx, out_idx] = cls_id.to(default_device)
            confidences_out[batch_idx, out_idx] = conf.to(default_device)
            valid_out[batch_idx, out_idx] = 1.0

    return {
        "instance_masks": masks_out,
        "class_ids": class_ids_out,
        "confidences": confidences_out,
        "valid": valid_out,
    }


def make_mask_guided_background_augmentation(
    images: torch.Tensor,
    masks: torch.Tensor,
    config: MaskWeightConfig,
    apply_mask: torch.Tensor | None = None,
    min_strength_override: float | None = None,
    max_strength_override: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Replace background pixels while preserving the YOLO target and context.

    Args:
        images: Unnormalized image batch in [0, 1], shaped [B, C, H, W].
        masks: YOLO soft masks shaped [B, 1, H, W] or resizable to image size.
        config: MaskWeightConfig with background augmentation parameters.
        apply_mask: Optional per-sample boolean mask shaped [B, 1, 1, 1].
        min_strength_override: Optional lower bound for background replacement strength.
        max_strength_override: Optional upper bound for background replacement strength.

    Returns:
        Augmented images and tensor debug metrics. Samples without a detected mask
        are left unchanged.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B, C, H, W], got {tuple(images.shape)}.")
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"masks must have shape [B, 1, H, W], got {tuple(masks.shape)}.")
    if images.shape[0] != masks.shape[0]:
        raise ValueError("images and masks must have the same batch size.")

    masks = masks.to(dtype=images.dtype, device=images.device)
    if masks.shape[-2:] != images.shape[-2:]:
        masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
    masks = torch.clamp(masks, 0.0, 1.0)

    dilation = int(config.background_aug_context_dilation)
    if dilation > 0:
        kernel_size = 2 * dilation + 1
        keep_mask = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=dilation)
    else:
        keep_mask = masks
    keep_mask = torch.clamp(keep_mask, 0.0, 1.0)

    threshold = float(config.background_aug_keep_threshold)
    has_target = keep_mask.amax(dim=(2, 3), keepdim=True) > threshold
    if threshold > 0.0:
        keep_mask = (keep_mask > threshold).to(dtype=images.dtype)

    batch_size, channels, _, _ = images.shape
    if apply_mask is None:
        apply_mask = torch.rand(
            batch_size,
            1,
            1,
            1,
            dtype=images.dtype,
            device=images.device,
        ) < float(config.background_aug_p)
    else:
        apply_mask = apply_mask.to(device=images.device).bool()
    apply_mask = apply_mask & has_target

    random_color = torch.rand(
        batch_size,
        channels,
        1,
        1,
        dtype=images.dtype,
        device=images.device,
    ).expand_as(images)
    noise = torch.rand_like(images)

    if config.background_aug_mode == "random_color":
        background = random_color
    elif config.background_aug_mode == "noise":
        background = noise
    elif config.background_aug_mode == "shuffle":
        if batch_size > 1:
            background = images[torch.randperm(batch_size, device=images.device)]
        else:
            background = random_color
    else:
        use_noise = torch.rand(
            batch_size,
            1,
            1,
            1,
            dtype=images.dtype,
            device=images.device,
        ) < float(config.background_aug_noise_p)
        background = torch.where(use_noise, noise, random_color)

    min_strength = (
        float(config.background_aug_min_strength)
        if min_strength_override is None
        else float(min_strength_override)
    )
    max_strength = (
        float(config.background_aug_max_strength)
        if max_strength_override is None
        else float(max_strength_override)
    )
    if not 0.0 <= min_strength <= 1.0 or not 0.0 <= max_strength <= 1.0:
        raise ValueError("background augmentation strength bounds must be in [0, 1].")
    if min_strength > max_strength:
        raise ValueError("background augmentation min strength must be <= max strength.")
    if min_strength == max_strength:
        strength = torch.full(
            (batch_size, 1, 1, 1),
            min_strength,
            dtype=images.dtype,
            device=images.device,
        )
    else:
        strength = min_strength + torch.rand(
            batch_size,
            1,
            1,
            1,
            dtype=images.dtype,
            device=images.device,
        ) * (max_strength - min_strength)
    mixed_background = images * (1.0 - strength) + background * strength

    candidate = keep_mask * images + (1.0 - keep_mask) * mixed_background
    augmented = torch.where(apply_mask, candidate, images)
    augmented = torch.clamp(augmented, 0.0, 1.0)

    applied = apply_mask.to(dtype=images.dtype)
    replaced = applied * (1.0 - keep_mask)
    effective_change = replaced * strength
    applied_strength = (strength * applied).sum() / applied.sum().clamp(min=1.0)
    debug = {
        "applied_ratio": apply_mask.detach().float().mean(),
        "keep_mask_mean": keep_mask.detach().float().mean(),
        "background_replaced_ratio": replaced.detach().float().mean(),
        "background_effective_change_ratio": effective_change.detach().float().mean(),
        "strength_mean": applied_strength.detach().float(),
        "strength_min": torch.as_tensor(min_strength, dtype=images.dtype, device=images.device),
        "strength_max": torch.as_tensor(max_strength, dtype=images.dtype, device=images.device),
        "image_delta_l1": (augmented.detach() - images.detach()).abs().float().mean(),
    }
    return augmented, debug


class MaskGuidedObjectPerceiverLayer(nn.Module):
    """Refine compact object tokens with masked cross-attention over visual tokens."""

    def __init__(self, dim_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim_model)
        self.self_attn = nn.MultiheadAttention(
            dim_model,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.cross_norm = nn.LayerNorm(dim_model)
        self.cross_attn = nn.MultiheadAttention(
            dim_model,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.ffn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim_model),
        )
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(
        self,
        object_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        attn_bias: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self_tokens = self.self_norm(object_tokens)
        self_out, _ = self.self_attn(
            self_tokens,
            self_tokens,
            self_tokens,
            need_weights=False,
        )
        object_tokens = object_tokens + self.dropout(self_out)

        cross_tokens = self.cross_norm(object_tokens)
        cross_out, cross_weights = self.cross_attn(
            cross_tokens,
            visual_tokens,
            visual_tokens,
            attn_mask=attn_bias,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        object_tokens = object_tokens + self.dropout(cross_out)
        object_tokens = object_tokens + self.dropout(self.ffn(self.ffn_norm(object_tokens)))
        return object_tokens, cross_weights


class ObjectWeightRelationLayer(nn.Module):
    """Lightweight relation layer for YOLO instance weighting."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_gate = valid.unsqueeze(-1).to(dtype=tokens.dtype)
        tokens = tokens * valid_gate
        attn_in = self.self_norm(tokens)
        key_padding_mask = valid <= 0.0
        if key_padding_mask.any():
            key_padding_mask = key_padding_mask.bool()
            all_padded = key_padding_mask.all(dim=1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded, 0] = False
        else:
            key_padding_mask = None
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = (tokens + attn_out) * valid_gate
        tokens = (tokens + self.ffn(self.ffn_norm(tokens))) * valid_gate
        return tokens


class ObjectWeightNet(nn.Module):
    """Predict per-instance weights before building the mask used by v4.2."""

    def __init__(
        self,
        dim_model: int,
        num_classes: int,
        embed_dim: int,
        hidden_dim: int,
        relation_layers: int,
        attention_heads: int,
        weight_init: float,
    ):
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                f"object_weight_attention_heads={attention_heads} must divide "
                f"object_weight_hidden_dim={hidden_dim}."
            )
        self.num_classes = max(1, int(num_classes))
        self.class_embed = nn.Embedding(self.num_classes + 1, embed_dim, padding_idx=0)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(dim_model + embed_dim + 6),
            nn.Linear(dim_model + embed_dim + 6, hidden_dim),
            nn.GELU(),
        )
        self.relation_layers = nn.ModuleList(
            [
                ObjectWeightRelationLayer(hidden_dim, attention_heads)
                for _ in range(int(relation_layers))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.output[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.output[-1].bias, math.log(weight_init / (1.0 - weight_init)))

    def forward(
        self,
        pooled_features: torch.Tensor,
        geometry: torch.Tensor,
        class_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        class_ids = torch.clamp(class_ids.to(dtype=torch.long), min=0, max=self.num_classes - 1)
        class_tokens = self.class_embed(class_ids + 1)
        inputs = torch.cat([pooled_features, geometry, class_tokens], dim=-1)
        tokens = self.input_proj(inputs) * valid.unsqueeze(-1).to(dtype=inputs.dtype)
        for layer in self.relation_layers:
            tokens = layer(tokens, valid)
        weights = torch.sigmoid(self.output(tokens)).squeeze(-1)
        return weights * valid.to(dtype=weights.dtype)


class MaskGuidedVisualAdapter(nn.Module):
    """Inject YOLO-derived spatial priors into projected ACT visual features.

    The adapter keeps the RGB backbone path intact. YOLO masks are used only as
    intermediate token guidance: spatial mask embeddings, an optional residual
    target gate, optional target summary tokens, and a v4 region-attention path.
    """

    def __init__(self, dim_model: int, config: MaskWeightConfig):
        super().__init__()
        self.config = config
        self.dim_model = dim_model
        self.latest_debug: dict[str, float] = {}
        self.latest_consistency: dict[str, torch.Tensor] = {}
        self.latest_target_token_debug: dict[str, torch.Tensor | None] = {}
        self.latest_mask_geometry_debug: dict[str, torch.Tensor | None] = {}
        self.latest_object_weight_debug: dict[str, torch.Tensor | None] = {}
        self.latest_object_weighted_mask: torch.Tensor | None = None
        self.latest_object_token_mask: torch.Tensor | None = None

        hidden_dim = int(config.adapter_hidden_dim)
        if config.use_spatial_embedding:
            self.mask_encoder = nn.Sequential(
                nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, dim_model, kernel_size=1),
            )
            nn.init.zeros_(self.mask_encoder[-1].weight)
            nn.init.zeros_(self.mask_encoder[-1].bias)
        else:
            self.mask_encoder = None

        if config.use_residual_gate:
            self.feature_adapter = nn.Sequential(
                nn.Conv2d(dim_model, dim_model, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(dim_model, dim_model, kernel_size=1),
            )
            nn.init.normal_(self.feature_adapter[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.feature_adapter[-1].bias)
            self.gate_scale = nn.Parameter(torch.tensor(float(config.gate_init)))
        else:
            self.feature_adapter = None
            self.register_parameter("gate_scale", None)

        self.use_region_attention = bool(config.use_region_attention)
        self.use_reliability_gate = bool(config.use_reliability_gate)
        if self.use_region_attention:
            num_heads = int(config.region_attention_heads)
            if dim_model % num_heads != 0:
                raise ValueError(
                    f"region_attention_heads={num_heads} must divide dim_model={dim_model}."
                )
            self.region_token_proj = nn.Sequential(
                nn.LayerNorm(dim_model),
                nn.Linear(dim_model, dim_model),
                nn.GELU(),
                nn.Linear(dim_model, dim_model),
            )
            nn.init.zeros_(self.region_token_proj[-1].weight)
            nn.init.zeros_(self.region_token_proj[-1].bias)
            self.region_type_embed = nn.Parameter(torch.zeros(3, 1, dim_model))
            self.region_cross_attn = nn.MultiheadAttention(
                dim_model,
                num_heads,
                dropout=float(config.region_attention_dropout),
                batch_first=False,
            )
            self.region_attn_out_proj = nn.Sequential(
                nn.LayerNorm(dim_model),
                nn.Linear(dim_model, dim_model),
            )
            nn.init.normal_(self.region_attn_out_proj[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.region_attn_out_proj[-1].bias)
            self.region_attention_scale = nn.Parameter(torch.tensor(float(config.region_attention_gate_init)))
        else:
            self.region_token_proj = None
            self.register_parameter("region_type_embed", None)
            self.region_cross_attn = None
            self.region_attn_out_proj = None
            self.register_parameter("region_attention_scale", None)

        self.num_target_tokens = max(0, int(config.num_target_tokens))
        if config.use_target_tokens and self.num_target_tokens > 0:
            self.target_token_proj = nn.Sequential(
                nn.LayerNorm(dim_model),
                nn.Linear(dim_model, dim_model),
                nn.GELU(),
                nn.Linear(dim_model, dim_model),
            )
            nn.init.zeros_(self.target_token_proj[-1].weight)
            nn.init.zeros_(self.target_token_proj[-1].bias)
            self.target_token_pos_embed = nn.Parameter(torch.zeros(self.num_target_tokens, 1, dim_model))

            if config.use_target_token_attention:
                num_heads = int(config.target_token_attention_heads)
                if dim_model % num_heads != 0:
                    raise ValueError(
                        f"target_token_attention_heads={num_heads} must divide dim_model={dim_model}."
                    )
                self.target_object_perceiver = nn.ModuleList(
                    [
                        MaskGuidedObjectPerceiverLayer(
                            dim_model=dim_model,
                            num_heads=num_heads,
                            ffn_dim=int(config.target_object_perceiver_ffn_dim),
                            dropout=float(config.target_object_perceiver_dropout),
                        )
                        for _ in range(int(config.target_object_perceiver_layers))
                    ]
                )
                self.target_token_attention_scale = nn.Parameter(
                    torch.tensor(float(config.target_token_attention_gate_init))
                )
            else:
                self.target_object_perceiver = None
                self.register_parameter("target_token_attention_scale", None)
        else:
            self.target_token_proj = None
            self.target_object_perceiver = None
            self.register_parameter("target_token_pos_embed", None)
            self.register_parameter("target_token_attention_scale", None)

        if config.use_mask_geometry_token:
            self.mask_geometry_token_proj = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, dim_model),
                nn.GELU(),
                nn.Linear(dim_model, dim_model),
            )
            nn.init.zeros_(self.mask_geometry_token_proj[-1].weight)
            nn.init.zeros_(self.mask_geometry_token_proj[-1].bias)
            self.mask_geometry_token_pos_embed = nn.Parameter(torch.zeros(1, 1, dim_model))
            self.mask_geometry_token_scale = nn.Parameter(torch.tensor(float(config.mask_geometry_token_gate_init)))
        else:
            self.mask_geometry_token_proj = None
            self.register_parameter("mask_geometry_token_pos_embed", None)
            self.register_parameter("mask_geometry_token_scale", None)

        if config.use_object_weighted_mask and int(config.max_object_weight_instances) > 0:
            self.object_weight_net = ObjectWeightNet(
                dim_model=dim_model,
                num_classes=int(config.num_yolo_classes or 80),
                embed_dim=int(config.object_weight_embed_dim),
                hidden_dim=int(config.object_weight_hidden_dim),
                relation_layers=int(config.object_weight_relation_layers),
                attention_heads=int(config.object_weight_attention_heads),
                weight_init=float(config.object_weight_init),
            )
        else:
            self.object_weight_net = None

    @staticmethod
    def _shift_mask(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        shifted = torch.roll(mask, shifts=(dy, dx), dims=(-2, -1))
        if dy > 0:
            shifted[..., :dy, :] = 0
        elif dy < 0:
            shifted[..., dy:, :] = 0
        if dx > 0:
            shifted[..., :, :dx] = 0
        elif dx < 0:
            shifted[..., :, dx:] = 0
        return shifted

    def _apply_mask_noise(self, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.config.mask_noise_p <= 0.0:
            return target_mask, torch.zeros(
                target_mask.shape[0],
                1,
                1,
                1,
                dtype=target_mask.dtype,
                device=target_mask.device,
            )

        batch_size = target_mask.shape[0]
        apply_noise = (
            torch.rand(batch_size, 1, 1, 1, dtype=target_mask.dtype, device=target_mask.device)
            < float(self.config.mask_noise_p)
        )
        noisy_mask = target_mask

        jitter_px = int(self.config.mask_noise_jitter_px)
        if jitter_px > 0:
            shifted_masks = []
            shifts = torch.randint(
                -jitter_px,
                jitter_px + 1,
                (batch_size, 2),
                device=target_mask.device,
            )
            for i in range(batch_size):
                dx = int(shifts[i, 0].item())
                dy = int(shifts[i, 1].item())
                shifted_masks.append(self._shift_mask(noisy_mask[i : i + 1], dx=dx, dy=dy))
            noisy_mask = torch.cat(shifted_masks, dim=0)

        if self.config.mask_noise_confidence_min < 1.0:
            min_scale = float(self.config.mask_noise_confidence_min)
            scale = min_scale + torch.rand(
                batch_size,
                1,
                1,
                1,
                dtype=target_mask.dtype,
                device=target_mask.device,
            ) * (1.0 - min_scale)
            noisy_mask = noisy_mask * scale

        if self.config.mask_noise_dropout_p > 0.0:
            keep_mask = (
                torch.rand(batch_size, 1, 1, 1, dtype=target_mask.dtype, device=target_mask.device)
                >= float(self.config.mask_noise_dropout_p)
            ).to(dtype=target_mask.dtype)
            noisy_mask = noisy_mask * keep_mask

        apply_noise_float = apply_noise.to(dtype=target_mask.dtype)
        target_mask = torch.where(apply_noise, noisy_mask, target_mask)
        return target_mask, apply_noise_float

    def _apply_shared_mask_stochastic(
        self,
        base_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_applied = torch.zeros(
            base_mask.shape[0],
            1,
            1,
            1,
            dtype=base_mask.dtype,
            device=base_mask.device,
        )
        if not self.training:
            return base_mask, object_mask, noise_applied

        batch_size = base_mask.shape[0]
        if self.config.mask_dropout_p > 0:
            keep = torch.rand(
                batch_size,
                1,
                1,
                1,
                dtype=base_mask.dtype,
                device=base_mask.device,
            )
            keep = (keep >= float(self.config.mask_dropout_p)).to(dtype=base_mask.dtype)
            base_mask = base_mask * keep
            object_mask = object_mask * keep

        if self.config.mask_noise_p <= 0.0:
            return base_mask, object_mask, noise_applied

        apply_noise = (
            torch.rand(batch_size, 1, 1, 1, dtype=base_mask.dtype, device=base_mask.device)
            < float(self.config.mask_noise_p)
        )
        noisy_base_mask = base_mask
        noisy_object_mask = object_mask

        jitter_px = int(self.config.mask_noise_jitter_px)
        if jitter_px > 0:
            shifted_base_masks = []
            shifted_object_masks = []
            shifts = torch.randint(
                -jitter_px,
                jitter_px + 1,
                (batch_size, 2),
                device=base_mask.device,
            )
            for i in range(batch_size):
                dx = int(shifts[i, 0].item())
                dy = int(shifts[i, 1].item())
                shifted_base_masks.append(self._shift_mask(noisy_base_mask[i : i + 1], dx=dx, dy=dy))
                shifted_object_masks.append(self._shift_mask(noisy_object_mask[i : i + 1], dx=dx, dy=dy))
            noisy_base_mask = torch.cat(shifted_base_masks, dim=0)
            noisy_object_mask = torch.cat(shifted_object_masks, dim=0)

        if self.config.mask_noise_confidence_min < 1.0:
            min_scale = float(self.config.mask_noise_confidence_min)
            scale = min_scale + torch.rand(
                batch_size,
                1,
                1,
                1,
                dtype=base_mask.dtype,
                device=base_mask.device,
            ) * (1.0 - min_scale)
            noisy_base_mask = noisy_base_mask * scale
            noisy_object_mask = noisy_object_mask * scale

        if self.config.mask_noise_dropout_p > 0.0:
            keep_mask = (
                torch.rand(batch_size, 1, 1, 1, dtype=base_mask.dtype, device=base_mask.device)
                >= float(self.config.mask_noise_dropout_p)
            ).to(dtype=base_mask.dtype)
            noisy_base_mask = noisy_base_mask * keep_mask
            noisy_object_mask = noisy_object_mask * keep_mask

        base_mask = torch.where(apply_noise, noisy_base_mask, base_mask)
        object_mask = torch.where(apply_noise, noisy_object_mask, object_mask)
        noise_applied = apply_noise.to(dtype=base_mask.dtype)
        return base_mask, object_mask, noise_applied

    def _prepare_object_weight_inputs(
        self,
        object_weight_inputs: dict[str, torch.Tensor] | None,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.object_weight_net is None or object_weight_inputs is None:
            return None

        instance_masks = object_weight_inputs.get("instance_masks")
        if not isinstance(instance_masks, torch.Tensor):
            return None
        if instance_masks.ndim == 4:
            instance_masks = instance_masks.unsqueeze(2)
        if instance_masks.ndim != 5 or instance_masks.shape[2] != 1:
            raise ValueError(
                "object_weight instance_masks must have shape [B, N, 1, H, W] or [B, N, H, W], "
                f"got {tuple(instance_masks.shape)}."
            )
        if instance_masks.shape[0] != features.shape[0]:
            raise ValueError("object_weight instance masks and visual features must have the same batch size.")

        batch_size, num_instances, _, mask_h, mask_w = instance_masks.shape
        max_instances = int(self.config.max_object_weight_instances)
        if num_instances > max_instances:
            instance_masks = instance_masks[:, :max_instances]
            num_instances = max_instances
        elif num_instances < max_instances:
            pad = torch.zeros(
                batch_size,
                max_instances - num_instances,
                1,
                mask_h,
                mask_w,
                dtype=instance_masks.dtype,
                device=instance_masks.device,
            )
            instance_masks = torch.cat([instance_masks, pad], dim=1)
            num_instances = max_instances

        instance_masks = instance_masks.to(dtype=features.dtype, device=features.device)
        if instance_masks.shape[-2:] != features.shape[-2:]:
            flat_masks = instance_masks.reshape(batch_size * num_instances, 1, mask_h, mask_w)
            flat_masks = F.interpolate(
                flat_masks,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            instance_masks = flat_masks.reshape(batch_size, num_instances, 1, *features.shape[-2:])
        instance_masks = torch.clamp(instance_masks, 0.0, 1.0)

        class_ids = object_weight_inputs.get("class_ids")
        if not isinstance(class_ids, torch.Tensor):
            class_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=features.device)
        class_ids = class_ids.to(device=features.device, dtype=torch.long)
        if class_ids.ndim != 2 or class_ids.shape[0] != batch_size:
            raise ValueError(f"object_weight class_ids must have shape [B, N], got {tuple(class_ids.shape)}.")
        if class_ids.shape[1] > num_instances:
            class_ids = class_ids[:, :num_instances]
        elif class_ids.shape[1] < num_instances:
            class_ids = F.pad(class_ids, (0, num_instances - class_ids.shape[1]), value=0)

        confidences = object_weight_inputs.get("confidences")
        if not isinstance(confidences, torch.Tensor):
            confidences = instance_masks.amax(dim=(2, 3, 4))
        confidences = confidences.to(device=features.device, dtype=features.dtype)
        if confidences.ndim != 2 or confidences.shape[0] != batch_size:
            raise ValueError(
                f"object_weight confidences must have shape [B, N], got {tuple(confidences.shape)}."
            )
        if confidences.shape[1] > num_instances:
            confidences = confidences[:, :num_instances]
        elif confidences.shape[1] < num_instances:
            confidences = F.pad(confidences, (0, num_instances - confidences.shape[1]), value=0.0)

        valid = object_weight_inputs.get("valid")
        if not isinstance(valid, torch.Tensor):
            valid = (instance_masks.amax(dim=(2, 3, 4)) > 0.0).to(dtype=features.dtype)
        valid = valid.to(device=features.device, dtype=features.dtype)
        if valid.ndim != 2 or valid.shape[0] != batch_size:
            raise ValueError(f"object_weight valid must have shape [B, N], got {tuple(valid.shape)}.")
        if valid.shape[1] > num_instances:
            valid = valid[:, :num_instances]
        elif valid.shape[1] < num_instances:
            valid = F.pad(valid, (0, num_instances - valid.shape[1]), value=0.0)
        valid = valid * (instance_masks.amax(dim=(2, 3, 4)) > 0.0).to(dtype=features.dtype)
        return instance_masks, class_ids, confidences, valid

    @staticmethod
    def _object_geometry(instance_masks: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, _, height, width = instance_masks.shape
        dtype = instance_masks.dtype
        device = instance_masks.device
        x_coords = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device).view(1, 1, 1, 1, width)
        y_coords = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device).view(1, 1, 1, height, 1)

        weights = torch.clamp(instance_masks, 0.0, 1.0)
        raw_denom = weights.sum(dim=(2, 3, 4), keepdim=True)
        has_mask = (raw_denom > 1e-6).to(dtype=dtype)
        denom = raw_denom.clamp(min=1e-6)
        center_x = (weights * x_coords).sum(dim=(2, 3, 4), keepdim=True) / denom
        center_y = (weights * y_coords).sum(dim=(2, 3, 4), keepdim=True) / denom
        center_x = torch.where(has_mask.bool(), center_x, torch.full_like(center_x, 0.5))
        center_y = torch.where(has_mask.bool(), center_y, torch.full_like(center_y, 0.5))
        var_x = (weights * (x_coords - center_x).pow(2)).sum(dim=(2, 3, 4), keepdim=True) / denom
        var_y = (weights * (y_coords - center_y).pow(2)).sum(dim=(2, 3, 4), keepdim=True) / denom
        spread_x = var_x.clamp(min=1e-12).sqrt() * has_mask
        spread_y = var_y.clamp(min=1e-12).sqrt() * has_mask
        area = weights.mean(dim=(2, 3, 4), keepdim=True)
        geometry = torch.cat(
            [
                confidences.view(batch_size, num_instances, 1, 1, 1),
                area,
                center_x,
                center_y,
                spread_x,
                spread_y,
            ],
            dim=2,
        )
        return geometry.view(batch_size, num_instances, 6)

    def _make_object_weighted_mask(
        self,
        features: torch.Tensor,
        union_mask: torch.Tensor,
        object_weight_inputs: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        self.latest_object_weighted_mask = union_mask
        self.latest_object_token_mask = union_mask
        if self.object_weight_net is None:
            self.latest_object_weight_debug = {}
            return union_mask

        prepared = self._prepare_object_weight_inputs(object_weight_inputs, features)
        if prepared is None:
            self.latest_object_weight_debug = {}
            return union_mask

        instance_masks, class_ids, confidences, valid = prepared
        weights = torch.clamp(instance_masks, 0.0, 1.0)
        denom = weights.sum(dim=(3, 4)).clamp(min=1e-6)
        pooled_features = (features.unsqueeze(1) * weights).sum(dim=(3, 4)) / denom
        geometry = self._object_geometry(instance_masks, confidences)
        object_weights = self.object_weight_net(pooled_features, geometry, class_ids, valid)

        weighted_mask = torch.clamp(
            (object_weights.view(object_weights.shape[0], object_weights.shape[1], 1, 1, 1) * instance_masks).amax(
                dim=1
            ),
            0.0,
            1.0,
        )
        floor_mask = torch.clamp(union_mask * float(self.config.object_weight_union_floor), 0.0, 1.0)
        base_mask = torch.maximum(floor_mask, weighted_mask)
        has_valid = (valid.sum(dim=1).view(-1, 1, 1, 1) > 0.0)
        base_mask = torch.where(has_valid, base_mask, union_mask)
        object_mix = float(self.config.object_token_mask_mix)
        object_token_mask = torch.clamp(
            base_mask * (1.0 - object_mix) + weighted_mask * object_mix,
            0.0,
            1.0,
        )
        object_token_mask = torch.where(has_valid, object_token_mask, union_mask)
        self.latest_object_weighted_mask = base_mask
        self.latest_object_token_mask = object_token_mask
        self.latest_object_weight_debug = {
            "weights": object_weights.detach(),
            "valid": valid.detach(),
            "class_ids": class_ids.detach(),
            "confidences": confidences.detach(),
            "union_mask": union_mask.detach(),
            "weighted_mask": weighted_mask.detach(),
            "final_mask": base_mask.detach(),
            "base_mask": base_mask.detach(),
            "object_token_mask": object_token_mask.detach(),
        }
        return base_mask

    def _compute_reliability(self, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        confidence_score = target_mask.amax(dim=(2, 3), keepdim=True)
        area = target_mask.mean(dim=(2, 3), keepdim=True)

        min_area = float(self.config.reliability_min_area)
        max_area = float(self.config.reliability_max_area)
        if min_area > 0.0:
            lower_area_score = torch.clamp(area / min_area, 0.0, 1.0)
        else:
            lower_area_score = torch.ones_like(area)
        if max_area < 1.0:
            upper_area_score = torch.clamp((1.0 - area) / (1.0 - max_area), 0.0, 1.0)
        else:
            upper_area_score = torch.ones_like(area)
        area_score = lower_area_score * upper_area_score

        reliability = torch.clamp(confidence_score * area_score, 0.0, 1.0)
        floor = float(self.config.reliability_floor)
        if floor > 0.0:
            reliability_with_floor = floor + (1.0 - floor) * reliability
            reliability = torch.where(area > 0.0, reliability_with_floor, torch.zeros_like(reliability))
        return reliability, confidence_score, area

    def _build_guidance(
        self,
        mask: torch.Tensor,
        apply_stochastic: bool = True,
        mask_noise_applied: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_mask = torch.clamp(mask, 0.0, 1.0)

        if apply_stochastic and self.training and self.config.mask_dropout_p > 0:
            keep = torch.rand(
                target_mask.shape[0],
                1,
                1,
                1,
                dtype=target_mask.dtype,
                device=target_mask.device,
            )
            keep = (keep >= float(self.config.mask_dropout_p)).to(dtype=target_mask.dtype)
            target_mask = target_mask * keep

        if apply_stochastic:
            target_mask, mask_noise_applied = self._apply_mask_noise(target_mask)
        elif mask_noise_applied is None:
            mask_noise_applied = torch.zeros(
                target_mask.shape[0],
                1,
                1,
                1,
                dtype=target_mask.dtype,
                device=target_mask.device,
            )
        reliability, confidence_score, mask_area = self._compute_reliability(target_mask)

        dilation = max(0, int(self.config.context_dilation))
        if dilation > 0:
            kernel_size = 2 * dilation + 1
            dilated_mask = F.max_pool2d(target_mask, kernel_size=kernel_size, stride=1, padding=dilation)
        else:
            dilated_mask = target_mask

        context_mask = torch.clamp(dilated_mask - target_mask, 0.0, 1.0)
        background_mask = torch.clamp(1.0 - dilated_mask, 0.0, 1.0)
        confidence_map = confidence_score.expand_as(target_mask)
        return target_mask, context_mask, background_mask, confidence_map, reliability, mask_area, mask_noise_applied

    @staticmethod
    def _masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = torch.clamp(mask, 0.0, 1.0)
        denom = weights.sum(dim=(2, 3)).clamp(min=1e-6)
        pooled = (features * weights).sum(dim=(2, 3)) / denom
        return pooled

    @staticmethod
    def _mask_geometry(
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        confidence_score: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = target_mask.shape
        dtype = target_mask.dtype
        device = target_mask.device
        x_coords = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device).view(1, 1, 1, width)
        y_coords = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device).view(1, 1, height, 1)

        weights = torch.clamp(target_mask, 0.0, 1.0)
        area = weights.mean(dim=(2, 3), keepdim=True)
        raw_denom = weights.sum(dim=(2, 3), keepdim=True)
        has_mask = (raw_denom > 1e-6).to(dtype=dtype)
        denom = raw_denom.clamp(min=1e-6)

        center_x = (weights * x_coords).sum(dim=(2, 3), keepdim=True) / denom
        center_y = (weights * y_coords).sum(dim=(2, 3), keepdim=True) / denom
        center_x = torch.where(has_mask.bool(), center_x, torch.full_like(center_x, 0.5))
        center_y = torch.where(has_mask.bool(), center_y, torch.full_like(center_y, 0.5))

        var_x = (weights * (x_coords - center_x).pow(2)).sum(dim=(2, 3), keepdim=True) / denom
        var_y = (weights * (y_coords - center_y).pow(2)).sum(dim=(2, 3), keepdim=True) / denom
        spread_x = var_x.clamp(min=1e-12).sqrt() * has_mask
        spread_y = var_y.clamp(min=1e-12).sqrt() * has_mask

        context_area = torch.clamp(context_mask, 0.0, 1.0).mean(dim=(2, 3), keepdim=True)
        geometry = torch.cat(
            [
                center_x,
                center_y,
                spread_x,
                spread_y,
                area,
                confidence_score,
                reliability,
                context_area,
            ],
            dim=1,
        )
        return geometry.view(batch_size, 8)

    def _make_mask_geometry_token(
        self,
        features: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        confidence_score: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.mask_geometry_token_proj is None:
            self.latest_mask_geometry_debug = {}
            return None, None

        geometry = self._mask_geometry(target_mask, context_mask, confidence_score, reliability)
        geometry = geometry.to(dtype=features.dtype, device=features.device)
        reliability_gate = reliability.to(dtype=features.dtype, device=features.device).view(1, geometry.shape[0], 1)
        geometry_token = self.mask_geometry_token_proj(geometry).unsqueeze(0)
        geometry_token = self.mask_geometry_token_scale * reliability_gate * geometry_token
        geometry_pos_embed = self.mask_geometry_token_pos_embed.to(dtype=features.dtype, device=features.device)

        self.latest_mask_geometry_debug = {
            "geometry": geometry.detach(),
            "geometry_token": geometry_token,
        }
        return geometry_token, geometry_pos_embed

    @staticmethod
    def _token_pool_masks(
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        background_mask: torch.Tensor,
        num_tokens: int,
    ) -> list[torch.Tensor]:
        global_mask = torch.ones_like(target_mask)
        target_context = torch.clamp(target_mask + context_mask, 0.0, 1.0)
        pool_masks = [
            target_mask,
            context_mask,
            target_context,
            global_mask,
            background_mask,
        ]
        return [torch.clamp(pool_masks[i] if i < len(pool_masks) else global_mask, 0.0, 1.0) for i in range(num_tokens)]

    def _target_token_attention_gate(self) -> torch.Tensor | None:
        if self.target_token_attention_scale is None:
            return None
        gate = torch.clamp(self.target_token_attention_scale, min=0.0)
        gate_max = float(getattr(self.config, "target_token_attention_gate_max", 0.0))
        if gate_max > 0.0:
            gate = torch.clamp(gate, max=gate_max)
        return gate

    def _apply_target_token_attention(
        self,
        features: torch.Tensor,
        token_seed: torch.Tensor,
        token_masks: list[torch.Tensor],
        reliability: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        record_debug: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.target_object_perceiver is None:
            return token_seed, None, None, None

        batch_size, channels, height, width = features.shape
        visual_tokens = features.flatten(2).permute(2, 0, 1)
        token_mask_tensor = torch.stack(token_masks, dim=0).flatten(3).squeeze(2).permute(1, 0, 2)
        attn_bias = token_mask_tensor * float(self.config.target_token_mask_bias_scale)
        num_heads = int(self.config.target_token_attention_heads)
        attn_bias = attn_bias.repeat_interleave(num_heads, dim=0).to(dtype=features.dtype, device=features.device)

        refined_tokens = token_seed
        attn_weights = None
        for layer in self.target_object_perceiver:
            refined_tokens, attn_weights = layer(
                refined_tokens,
                visual_tokens,
                attn_bias,
                need_weights=record_debug,
            )
        attn_delta = refined_tokens - token_seed
        reliability_gate = reliability if self.use_reliability_gate else torch.ones_like(reliability)
        token_gate = reliability_gate.view(1, batch_size, 1)
        attention_gate = self._target_token_attention_gate()
        if attention_gate is None:
            return token_seed, None, attn_weights, None
        attn_delta = attention_gate * token_gate * attn_delta

        limiter_scale = torch.ones(1, batch_size, 1, dtype=features.dtype, device=features.device)
        delta_ratio_limit = float(getattr(self.config, "target_token_delta_ratio_limit", 0.0))
        if delta_ratio_limit > 0.0:
            visual_rms = features.detach().float().pow(2).mean(dim=(1, 2, 3)).sqrt().clamp(min=1e-6)
            delta_rms = attn_delta.detach().float().pow(2).mean(dim=(0, 2)).sqrt().clamp(min=1e-6)
            limiter_scale = torch.clamp(
                (visual_rms * delta_ratio_limit / delta_rms).view(1, batch_size, 1),
                max=1.0,
            ).to(dtype=features.dtype, device=features.device)
            attn_delta = attn_delta * limiter_scale
        return token_seed + attn_delta, attn_delta, attn_weights, limiter_scale

    def _make_target_tokens(
        self,
        features: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        background_mask: torch.Tensor,
        reliability: torch.Tensor,
        record_debug: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.target_token_proj is None:
            self.latest_target_token_debug = {}
            return None, None

        token_masks = self._token_pool_masks(target_mask, context_mask, background_mask, self.num_target_tokens)
        pooled_tokens = []
        for pool_mask in token_masks:
            pooled_tokens.append(self._masked_pool(features, pool_mask))
        target_tokens = torch.stack(pooled_tokens, dim=0)
        target_tokens = target_tokens + self.target_token_proj(target_tokens)

        target_tokens, attention_delta, attention_weights, attention_limiter = self._apply_target_token_attention(
            features,
            target_tokens,
            token_masks,
            reliability,
            target_mask,
            context_mask,
            record_debug=record_debug,
        )
        target_pos_embed = self.target_token_pos_embed.to(dtype=features.dtype, device=features.device)

        self.latest_target_token_debug = {
            "attention_delta": attention_delta,
            "attention_weights": attention_weights,
            "attention_gate": self._target_token_attention_gate().detach()
            if self._target_token_attention_gate() is not None
            else None,
            "attention_gate_raw": self.target_token_attention_scale.detach()
            if self.target_token_attention_scale is not None
            else None,
            "attention_limiter": attention_limiter.detach() if attention_limiter is not None else None,
            "target_mask": target_mask.detach(),
            "context_mask": context_mask.detach(),
        }
        return target_tokens, target_pos_embed

    def _make_consistency_data(
        self,
        guided_features: torch.Tensor,
        target_mask: torch.Tensor,
        background_mask: torch.Tensor,
        target_tokens: torch.Tensor | None,
        object_target_mask: torch.Tensor | None = None,
        object_background_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        data = {
            "target_feature": self._masked_pool(guided_features, target_mask),
            "background_feature": self._masked_pool(guided_features, background_mask),
        }
        if object_target_mask is not None:
            data["object_target_feature"] = self._masked_pool(guided_features, object_target_mask)
        if object_background_mask is not None:
            data["object_background_feature"] = self._masked_pool(guided_features, object_background_mask)
        if target_tokens is not None:
            data["target_tokens"] = target_tokens
        return data

    def _make_region_tokens(
        self,
        features: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        background_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = [
            self._masked_pool(features, target_mask),
            self._masked_pool(features, context_mask),
            self._masked_pool(features, background_mask),
        ]
        region_tokens = torch.stack(pooled, dim=0)
        region_tokens = region_tokens + self.region_token_proj(region_tokens)
        region_type_embed = self.region_type_embed.to(dtype=features.dtype, device=features.device)
        return region_tokens + region_type_embed

    def _apply_region_attention(
        self,
        visual_features: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        background_mask: torch.Tensor,
        reliability: torch.Tensor,
        record_debug: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.use_region_attention:
            return visual_features, None, None

        batch_size, channels, height, width = visual_features.shape
        visual_tokens = visual_features.flatten(2).permute(2, 0, 1)
        region_tokens = self._make_region_tokens(visual_features, target_mask, context_mask, background_mask)
        attn_out, attn_weights = self.region_cross_attn(
            visual_tokens,
            region_tokens,
            region_tokens,
            need_weights=record_debug,
            average_attn_weights=True,
        )
        attn_delta = self.region_attn_out_proj(attn_out)
        attn_delta = attn_delta.permute(1, 2, 0).reshape(batch_size, channels, height, width)

        spatial_weight = torch.clamp(
            target_mask
            + float(self.config.region_attention_context_weight) * context_mask
            + float(self.config.region_attention_background_weight) * background_mask,
            0.0,
            1.0,
        )
        reliability_gate = reliability if self.use_reliability_gate else torch.ones_like(reliability)
        region_delta = self.region_attention_scale * reliability_gate * spatial_weight * attn_delta
        return visual_features + region_delta, region_delta, attn_weights

    def forward(
        self,
        visual_features: torch.Tensor,
        mask: torch.Tensor,
        object_weight_inputs: dict[str, torch.Tensor] | None = None,
        record_debug: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        mask = mask.to(dtype=visual_features.dtype, device=visual_features.device)
        mask = self._make_object_weighted_mask(visual_features, mask, object_weight_inputs)
        object_token_mask = self.latest_object_token_mask
        if object_token_mask is None:
            object_token_mask = mask
        object_token_mask = object_token_mask.to(dtype=visual_features.dtype, device=visual_features.device)
        mask, object_token_mask, mask_noise_applied = self._apply_shared_mask_stochastic(mask, object_token_mask)
        (
            target_mask,
            context_mask,
            background_mask,
            confidence_map,
            reliability,
            mask_area,
            mask_noise_applied,
        ) = self._build_guidance(mask, apply_stochastic=False, mask_noise_applied=mask_noise_applied)
        (
            object_target_mask,
            object_context_mask,
            object_background_mask,
            _object_confidence_map,
            object_reliability,
            _object_mask_area,
            _object_mask_noise_applied,
        ) = self._build_guidance(object_token_mask, apply_stochastic=False)
        guidance = torch.cat([target_mask, context_mask, background_mask, confidence_map], dim=1)
        reliability_gate = reliability if self.use_reliability_gate else torch.ones_like(reliability)

        guided_features = visual_features
        spatial_delta = None
        if self.mask_encoder is not None:
            spatial_delta = reliability_gate * self.mask_encoder(guidance)
            guided_features = guided_features + spatial_delta

        residual_delta = None
        if self.feature_adapter is not None:
            residual_delta = reliability_gate * self.gate_scale * target_mask * self.feature_adapter(visual_features)
            guided_features = guided_features + residual_delta

        region_delta = None
        attn_weights = None
        guided_features, region_delta, attn_weights = self._apply_region_attention(
            guided_features,
            target_mask,
            context_mask,
            background_mask,
            reliability,
            record_debug=record_debug,
        )

        confidence_score = confidence_map.amax(dim=(2, 3), keepdim=True)
        target_tokens, target_pos_embed = self._make_target_tokens(
            guided_features,
            object_target_mask,
            object_context_mask,
            object_background_mask,
            object_reliability,
            record_debug=record_debug,
        )
        mask_geometry_token, mask_geometry_pos_embed = self._make_mask_geometry_token(
            guided_features,
            target_mask,
            context_mask,
            confidence_score,
            reliability,
        )
        self.latest_consistency = self._make_consistency_data(
            guided_features,
            target_mask,
            background_mask,
            target_tokens,
            object_target_mask=object_target_mask,
            object_background_mask=object_background_mask,
        )
        if not record_debug:
            self.latest_debug = {}
            return guided_features, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed

        with torch.no_grad():
            visual_float = visual_features.detach().float()
            guided_float = guided_features.detach().float()

            def _masked_rms(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
                weights_float = weights.detach().float()
                weighted_energy = values.pow(2) * weights_float
                denom = (weights_float.sum() * values.shape[1]).clamp(min=1e-6)
                return (weighted_energy.sum() / denom).sqrt()

            visual_rms = visual_float.pow(2).mean().sqrt().clamp(min=1e-6)
            output_delta = guided_float - visual_float
            output_delta_rms = output_delta.float().pow(2).mean().sqrt()
            input_target_rms = _masked_rms(visual_float, target_mask)
            input_background_rms = _masked_rms(visual_float, background_mask)
            output_target_rms = _masked_rms(guided_float, target_mask)
            output_background_rms = _masked_rms(guided_float, background_mask)
            target_background_ratio_before = input_target_rms / input_background_rms.clamp(min=1e-6)
            target_background_ratio_after = output_target_rms / output_background_rms.clamp(min=1e-6)
            target_background_ratio_gain = target_background_ratio_after / target_background_ratio_before.clamp(
                min=1e-6
            )
            target_delta_ratio = _masked_rms(output_delta, target_mask) / input_target_rms.clamp(min=1e-6)
            background_delta_ratio = _masked_rms(output_delta, background_mask) / input_background_rms.clamp(
                min=1e-6
            )
            spatial_delta_rms = (
                torch.zeros((), device=visual_features.device)
                if spatial_delta is None
                else spatial_delta.detach().float().pow(2).mean().sqrt()
            )
            residual_delta_rms = (
                torch.zeros((), device=visual_features.device)
                if residual_delta is None
                else residual_delta.detach().float().pow(2).mean().sqrt()
            )
            region_delta_rms = (
                torch.zeros((), device=visual_features.device)
                if region_delta is None
                else region_delta.detach().float().pow(2).mean().sqrt()
            )
            target_token_debug = self.latest_target_token_debug
            target_token_attention_delta = target_token_debug.get("attention_delta")
            target_token_attention_delta_rms = (
                torch.zeros((), device=visual_features.device)
                if target_token_attention_delta is None
                else target_token_attention_delta.detach().float().pow(2).mean().sqrt()
            )
            object_weight_debug = self.latest_object_weight_debug
            object_weights = object_weight_debug.get("weights")
            object_valid = object_weight_debug.get("valid")
            object_class_ids = object_weight_debug.get("class_ids")
            object_confidences = object_weight_debug.get("confidences")
            object_union_mask = object_weight_debug.get("union_mask")
            object_weighted_mask = object_weight_debug.get("weighted_mask")
            object_final_mask = object_weight_debug.get("final_mask")
            object_base_mask = object_weight_debug.get("base_mask")
            object_token_mask_debug = object_weight_debug.get("object_token_mask")
            target_token_attention_gate = target_token_debug.get("attention_gate")
            target_token_attention_gate_raw = target_token_debug.get("attention_gate_raw")
            target_token_attention_limiter = target_token_debug.get("attention_limiter")
            if isinstance(object_weights, torch.Tensor) and isinstance(object_valid, torch.Tensor):
                valid_float = object_valid.detach().float()
                valid_sum = valid_float.sum()
                has_valid_objects = bool(valid_sum.item() > 0.0)
                valid_denom = valid_sum.clamp(min=1.0)
                object_weights_float = object_weights.detach().float()
                valid_weights = object_weights_float * valid_float
                object_weight_mean = valid_weights.sum() / valid_denom
                object_weight_max = valid_weights.max()
                object_weight_min = torch.where(
                    valid_float > 0.0,
                    object_weights_float,
                    torch.ones_like(object_weights_float),
                ).min()
                if not has_valid_objects:
                    object_weight_max = torch.zeros_like(object_weight_max)
                    object_weight_min = torch.zeros_like(object_weight_min)
                centered = (object_weights_float - object_weight_mean) * valid_float
                object_weight_std = (centered.pow(2).sum() / valid_denom).sqrt()
                object_weight_valid_count = valid_float.sum(dim=1).mean()
                top_scores, top_indices = valid_weights.max(dim=1)
                top_confidence = (
                    torch.zeros((), device=visual_features.device)
                    if not isinstance(object_confidences, torch.Tensor)
                    else object_confidences.detach().float().gather(1, top_indices.view(-1, 1)).mean()
                )
                top_class = (
                    torch.zeros((), device=visual_features.device)
                    if not isinstance(object_class_ids, torch.Tensor)
                    else object_class_ids.detach().float().gather(1, top_indices.view(-1, 1)).mean()
                )
                top_weight = top_scores.mean()
            else:
                object_weight_mean = torch.zeros((), device=visual_features.device)
                object_weight_max = torch.zeros((), device=visual_features.device)
                object_weight_min = torch.zeros((), device=visual_features.device)
                object_weight_std = torch.zeros((), device=visual_features.device)
                object_weight_valid_count = torch.zeros((), device=visual_features.device)
                top_confidence = torch.zeros((), device=visual_features.device)
                top_class = torch.zeros((), device=visual_features.device)
                top_weight = torch.zeros((), device=visual_features.device)
            union_mask_mean = (
                torch.zeros((), device=visual_features.device)
                if object_union_mask is None
                else object_union_mask.detach().float().mean()
            )
            weighted_mask_mean = (
                torch.zeros((), device=visual_features.device)
                if object_weighted_mask is None
                else object_weighted_mask.detach().float().mean()
            )
            final_mask_mean = (
                torch.zeros((), device=visual_features.device)
                if object_final_mask is None
                else object_final_mask.detach().float().mean()
            )
            base_mask_mean = (
                torch.zeros((), device=visual_features.device)
                if object_base_mask is None
                else object_base_mask.detach().float().mean()
            )
            object_token_mask_mean = (
                torch.zeros((), device=visual_features.device)
                if object_token_mask_debug is None
                else object_token_mask_debug.detach().float().mean()
            )
            mask_geometry_debug = self.latest_mask_geometry_debug
            mask_geometry_values = mask_geometry_debug.get("geometry")
            mask_geometry_token_debug = mask_geometry_debug.get("geometry_token")
            mask_geometry_token_norm = (
                torch.zeros((), device=visual_features.device)
                if mask_geometry_token_debug is None
                else mask_geometry_token_debug.detach().float().pow(2).mean().sqrt()
            )
            self.latest_debug = {
                "visual_rms": float(visual_rms.item()),
                "spatial_delta_rms": float(spatial_delta_rms.item()),
                "spatial_delta_ratio": float((spatial_delta_rms / visual_rms).item()),
                "residual_delta_rms": float(residual_delta_rms.item()),
                "residual_delta_ratio": float((residual_delta_rms / visual_rms).item()),
                "region_attention_delta_rms": float(region_delta_rms.item()),
                "region_attention_delta_ratio": float((region_delta_rms / visual_rms).item()),
                "target_token_attention_delta_rms": float(target_token_attention_delta_rms.item()),
                "target_token_attention_delta_ratio": float((target_token_attention_delta_rms / visual_rms).item()),
                "object_perceiver_delta_rms": float(target_token_attention_delta_rms.item()),
                "object_perceiver_delta_ratio": float((target_token_attention_delta_rms / visual_rms).item()),
                "object_perceiver_layers": float(
                    0 if self.target_object_perceiver is None else len(self.target_object_perceiver)
                ),
                "object_weight_mean": float(object_weight_mean.item()),
                "object_weight_max": float(object_weight_max.item()),
                "object_weight_min": float(object_weight_min.item()),
                "object_weight_std": float(object_weight_std.item()),
                "object_weight_valid_count": float(object_weight_valid_count.item()),
                "object_weight_final_mask_mean": float(final_mask_mean.item()),
                "object_weight_base_mask_mean": float(base_mask_mean.item()),
                "object_weight_object_token_mask_mean": float(object_token_mask_mean.item()),
                "object_weight_union_mask_mean": float(union_mask_mean.item()),
                "object_weight_weighted_mask_mean": float(weighted_mask_mean.item()),
                "object_weight_floor": float(self.config.object_weight_union_floor),
                "object_weight_object_token_mix": float(self.config.object_token_mask_mix),
                "object_weight_top_confidence": float(top_confidence.item()),
                "object_weight_top_class": float(top_class.item()),
                "object_weight_top_weight": float(top_weight.item()),
                "target_token_geometry_norm": 0.0,
                "mask_geometry_token_norm": float(mask_geometry_token_norm.item()),
                "output_delta_rms": float(output_delta_rms.item()),
                "output_delta_ratio": float((output_delta_rms / visual_rms).item()),
                "input_target_rms": float(input_target_rms.item()),
                "input_background_rms": float(input_background_rms.item()),
                "output_target_rms": float(output_target_rms.item()),
                "output_background_rms": float(output_background_rms.item()),
                "target_background_rms_ratio_before": float(target_background_ratio_before.item()),
                "target_background_rms_ratio_after": float(target_background_ratio_after.item()),
                "target_background_rms_ratio_gain": float(target_background_ratio_gain.item()),
                "target_delta_ratio": float(target_delta_ratio.item()),
                "background_delta_ratio": float(background_delta_ratio.item()),
                "target_mask_mean": float(target_mask.detach().float().mean().item()),
                "context_mask_mean": float(context_mask.detach().float().mean().item()),
                "background_mask_mean": float(background_mask.detach().float().mean().item()),
                "object_token_target_mask_mean": float(object_target_mask.detach().float().mean().item()),
                "object_token_context_mask_mean": float(object_context_mask.detach().float().mean().item()),
                "object_token_background_mask_mean": float(object_background_mask.detach().float().mean().item()),
                "object_token_reliability_mean": float(object_reliability.detach().float().mean().item()),
                "mask_area_mean": float(mask_area.detach().float().mean().item()),
                "reliability_mean": float(reliability.detach().float().mean().item()),
                "reliability_min": float(reliability.detach().float().min().item()),
                "reliability_max": float(reliability.detach().float().max().item()),
                "mask_noise_applied_ratio": float(mask_noise_applied.detach().float().mean().item()),
            }
            if isinstance(mask_geometry_values, torch.Tensor):
                geometry_float = mask_geometry_values.detach().float()
                self.latest_debug["mask_geometry_center_x"] = float(geometry_float[:, 0].mean().item())
                self.latest_debug["mask_geometry_center_y"] = float(geometry_float[:, 1].mean().item())
                self.latest_debug["mask_geometry_spread_x"] = float(geometry_float[:, 2].mean().item())
                self.latest_debug["mask_geometry_spread_y"] = float(geometry_float[:, 3].mean().item())
                self.latest_debug["mask_geometry_area"] = float(geometry_float[:, 4].mean().item())
                self.latest_debug["mask_geometry_confidence"] = float(geometry_float[:, 5].mean().item())
                self.latest_debug["mask_geometry_reliability"] = float(geometry_float[:, 6].mean().item())
                self.latest_debug["mask_geometry_context_area"] = float(geometry_float[:, 7].mean().item())
            if self.mask_geometry_token_scale is not None:
                self.latest_debug["mask_geometry_token_gate"] = float(
                    self.mask_geometry_token_scale.detach().item()
                )
            if self.region_attention_scale is not None:
                self.latest_debug["region_attention_gate"] = float(
                    self.region_attention_scale.detach().item()
                )
            if attn_weights is not None:
                region_attn_mean = attn_weights.detach().float().mean(dim=(0, 1))
                self.latest_debug["region_attention_target_mean"] = float(region_attn_mean[0].item())
                self.latest_debug["region_attention_context_mean"] = float(region_attn_mean[1].item())
                self.latest_debug["region_attention_background_mean"] = float(region_attn_mean[2].item())
            if self.target_token_attention_scale is not None:
                gate_value = (
                    self._target_token_attention_gate()
                    if not isinstance(target_token_attention_gate, torch.Tensor)
                    else target_token_attention_gate
                )
                self.latest_debug["target_token_attention_gate"] = float(
                    gate_value.detach().item()
                )
                self.latest_debug["object_perceiver_gate"] = float(
                    gate_value.detach().item()
                )
                self.latest_debug["target_token_attention_gate_raw"] = float(
                    target_token_attention_gate_raw.detach().item()
                    if isinstance(target_token_attention_gate_raw, torch.Tensor)
                    else self.target_token_attention_scale.detach().item()
                )
                if isinstance(target_token_attention_limiter, torch.Tensor):
                    limiter_float = target_token_attention_limiter.detach().float()
                    self.latest_debug["target_token_delta_limiter_mean"] = float(
                        limiter_float.mean().item()
                    )
                    self.latest_debug["target_token_delta_limiter_min"] = float(
                        limiter_float.min().item()
                    )
            target_token_attention_weights = target_token_debug.get("attention_weights")
            if isinstance(target_token_attention_weights, torch.Tensor):
                attn_float = target_token_attention_weights.detach().float()
                target_flat = target_mask.detach().float().flatten(2).squeeze(1)
                context_flat = context_mask.detach().float().flatten(2).squeeze(1)
                if attn_float.shape[1] > 0:
                    self.latest_debug["target_token_target_attn_mean"] = float(
                        (attn_float[:, 0, :] * target_flat).sum(dim=1).mean().item()
                    )
                if attn_float.shape[1] > 1:
                    self.latest_debug["target_token_context_attn_mean"] = float(
                        (attn_float[:, 1, :] * context_flat).sum(dim=1).mean().item()
                    )
        return guided_features, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed
