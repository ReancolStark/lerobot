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

        self.use_region_attention = config.mode == "region_attention" and bool(config.use_region_attention)
        self.use_reliability_gate = config.mode == "region_attention" and bool(config.use_reliability_gate)
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
        else:
            self.target_token_proj = None
            self.register_parameter("target_token_pos_embed", None)

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
            reliability = floor + (1.0 - floor) * reliability
        return reliability, confidence_score, area

    def _build_guidance(
        self,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_mask = torch.clamp(mask, 0.0, 1.0)

        if self.training and self.config.mask_dropout_p > 0:
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

        target_mask, mask_noise_applied = self._apply_mask_noise(target_mask)
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

    def _make_target_tokens(
        self,
        features: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        background_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.target_token_proj is None:
            return None, None

        global_mask = torch.ones_like(target_mask)
        pool_masks = [target_mask, context_mask, target_mask + context_mask, background_mask, global_mask]
        pooled_tokens = []
        for i in range(self.num_target_tokens):
            pool_mask = pool_masks[i] if i < len(pool_masks) else global_mask
            pooled_tokens.append(self._masked_pool(features, torch.clamp(pool_mask, 0.0, 1.0)))
        target_tokens = torch.stack(pooled_tokens, dim=0)
        target_tokens = target_tokens + self.target_token_proj(target_tokens)
        target_pos_embed = self.target_token_pos_embed.to(dtype=features.dtype, device=features.device)
        return target_tokens, target_pos_embed

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
        record_debug: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        mask = mask.to(dtype=visual_features.dtype, device=visual_features.device)
        (
            target_mask,
            context_mask,
            background_mask,
            confidence_map,
            reliability,
            mask_area,
            mask_noise_applied,
        ) = self._build_guidance(mask)
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

        target_tokens, target_pos_embed = self._make_target_tokens(
            guided_features,
            target_mask,
            context_mask,
            background_mask,
        )
        if not record_debug:
            self.latest_debug = {}
            return guided_features, target_tokens, target_pos_embed

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
            self.latest_debug = {
                "visual_rms": float(visual_rms.item()),
                "spatial_delta_rms": float(spatial_delta_rms.item()),
                "spatial_delta_ratio": float((spatial_delta_rms / visual_rms).item()),
                "residual_delta_rms": float(residual_delta_rms.item()),
                "residual_delta_ratio": float((residual_delta_rms / visual_rms).item()),
                "region_attention_delta_rms": float(region_delta_rms.item()),
                "region_attention_delta_ratio": float((region_delta_rms / visual_rms).item()),
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
                "mask_area_mean": float(mask_area.detach().float().mean().item()),
                "reliability_mean": float(reliability.detach().float().mean().item()),
                "reliability_min": float(reliability.detach().float().min().item()),
                "reliability_max": float(reliability.detach().float().max().item()),
                "mask_noise_applied_ratio": float(mask_noise_applied.detach().float().mean().item()),
            }
            if self.region_attention_scale is not None:
                self.latest_debug["region_attention_gate"] = float(
                    self.region_attention_scale.detach().item()
                )
            if attn_weights is not None:
                region_attn_mean = attn_weights.detach().float().mean(dim=(0, 1))
                self.latest_debug["region_attention_target_mean"] = float(region_attn_mean[0].item())
                self.latest_debug["region_attention_context_mean"] = float(region_attn_mean[1].item())
                self.latest_debug["region_attention_background_mean"] = float(region_attn_mean[2].item())
        return guided_features, target_tokens, target_pos_embed
