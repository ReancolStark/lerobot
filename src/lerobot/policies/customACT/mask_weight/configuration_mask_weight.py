from dataclasses import dataclass


@dataclass
class MaskWeightConfig:
    # V4 mask-guided adapter switches.
    use_spatial_embedding: bool = True
    use_residual_gate: bool = False
    use_target_tokens: bool = True
    use_mask_geometry_token: bool = True
    use_region_attention: bool = True
    use_reliability_gate: bool = True

    # Mask-guided adapter hyperparameters.
    adapter_hidden_dim: int = 128
    gate_init: float = 0.2
    context_dilation: int = 3
    mask_dropout_p: float = 0.1
    num_target_tokens: int = 2
    use_target_token_attention: bool = True
    target_token_attention_heads: int = 4
    target_token_attention_dropout: float = 0.0
    target_token_attention_gate_init: float = 0.2
    target_token_mask_bias_scale: float = 2.0
    target_object_perceiver_layers: int = 2
    target_object_perceiver_ffn_dim: int = 1024
    target_object_perceiver_dropout: float = 0.0
    mask_geometry_token_gate_init: float = 0.2

    # Instance-level object tokens. The union mask remains the stable dense
    # guidance/background-consistency signal; these tokens expose individual
    # YOLO instances to the ACT encoder.
    use_instance_object_tokens: bool = True
    max_instance_object_tokens: int = 4
    instance_object_token_gate_init: float = 0.1
    instance_object_token_min_area: float = 0.001
    instance_object_token_mask_bias_scale: float = 2.0

    # YOLO mask post-processing.
    mask_blur_kernel_size: int = 7
    mask_blur_sigma: float = 2.0

    # V4 reliability gate. These values are measured on the feature-grid mask
    # area, not raw pixels. If YOLO is missing, too tiny, or too large, the
    # mask-guided residual path is weakened and the policy falls back toward ACT.
    reliability_min_area: float = 0.002
    reliability_max_area: float = 0.65
    reliability_floor: float = 0.6

    # V4 region cross-attention.
    region_attention_heads: int = 4
    region_attention_dropout: float = 0.0
    region_attention_gate_init: float = 0.2
    region_attention_context_weight: float = 0.3
    region_attention_background_weight: float = 0.0

    # V4 mask corruption during training. This teaches the policy not to trust
    # YOLO as a perfect sensor while keeping the original RGB path intact.
    mask_noise_p: float = 0.0
    mask_noise_jitter_px: int = 1
    mask_noise_confidence_min: float = 0.5
    mask_noise_dropout_p: float = 0.0

    # Training-time background counterfactuals. The auxiliary forward keeps the
    # original RGB inference path intact while making background dependence costly.
    use_background_augmentation: bool = True
    use_background_consistency: bool = True
    background_aug_p: float = 0.5
    background_consistency_loss_weight: float = 0.1
    background_feature_consistency_loss_weight: float = 0.03
    background_token_consistency_loss_weight: float = 0.03
    target_background_contrastive_loss_weight: float = 0.02
    target_background_contrastive_margin: float = 0.2
    background_aug_context_dilation: int = 21
    background_aug_keep_threshold: float = 0.05
    background_aug_mode: str = "mixed"  # "random_color", "noise", "mixed", or "shuffle".
    background_aug_noise_p: float = 0.5

    # Default strength range for direct/background augmentation helper calls.
    background_aug_min_strength: float = 1.0
    background_aug_max_strength: float = 1.0

    # Strength range used by the background-consistency auxiliary forward.
    # Use 0.75~1.0 for the v2-style auxiliary path; set both to 1.0 to match the original v3 behavior.
    background_consistency_aug_min_strength: float = 0.75
    background_consistency_aug_max_strength: float = 1.0

    # Central controls for mask_weight training/debug metrics.
    record_debug_log: bool = True

    def __post_init__(self):
        if not 0.0 <= self.mask_dropout_p <= 1.0:
            raise ValueError("mask_dropout_p must be in [0, 1].")
        if self.target_token_attention_heads <= 0:
            raise ValueError("target_token_attention_heads must be positive.")
        if not 0.0 <= self.target_token_attention_dropout <= 1.0:
            raise ValueError("target_token_attention_dropout must be in [0, 1].")
        if self.target_token_mask_bias_scale < 0.0:
            raise ValueError("target_token_mask_bias_scale must be non-negative.")
        if self.target_object_perceiver_layers <= 0:
            raise ValueError("target_object_perceiver_layers must be positive.")
        if self.target_object_perceiver_ffn_dim <= 0:
            raise ValueError("target_object_perceiver_ffn_dim must be positive.")
        if not 0.0 <= self.target_object_perceiver_dropout <= 1.0:
            raise ValueError("target_object_perceiver_dropout must be in [0, 1].")
        if not 0.0 <= self.reliability_min_area <= 1.0:
            raise ValueError("reliability_min_area must be in [0, 1].")
        if not 0.0 <= self.reliability_max_area <= 1.0:
            raise ValueError("reliability_max_area must be in [0, 1].")
        if self.reliability_min_area > self.reliability_max_area:
            raise ValueError("reliability_min_area must be <= reliability_max_area.")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0, 1].")
        if self.region_attention_heads <= 0:
            raise ValueError("region_attention_heads must be positive.")
        if not 0.0 <= self.region_attention_dropout <= 1.0:
            raise ValueError("region_attention_dropout must be in [0, 1].")
        if not 0.0 <= self.region_attention_context_weight <= 1.0:
            raise ValueError("region_attention_context_weight must be in [0, 1].")
        if not 0.0 <= self.region_attention_background_weight <= 1.0:
            raise ValueError("region_attention_background_weight must be in [0, 1].")
        if not 0.0 <= self.mask_noise_p <= 1.0:
            raise ValueError("mask_noise_p must be in [0, 1].")
        if self.mask_noise_jitter_px < 0:
            raise ValueError("mask_noise_jitter_px must be non-negative.")
        if not 0.0 <= self.mask_noise_confidence_min <= 1.0:
            raise ValueError("mask_noise_confidence_min must be in [0, 1].")
        if not 0.0 <= self.mask_noise_dropout_p <= 1.0:
            raise ValueError("mask_noise_dropout_p must be in [0, 1].")
        if not 0.0 <= self.background_aug_p <= 1.0:
            raise ValueError("background_aug_p must be in [0, 1].")
        if self.background_consistency_loss_weight < 0.0:
            raise ValueError("background_consistency_loss_weight must be non-negative.")
        if self.background_feature_consistency_loss_weight < 0.0:
            raise ValueError("background_feature_consistency_loss_weight must be non-negative.")
        if self.background_token_consistency_loss_weight < 0.0:
            raise ValueError("background_token_consistency_loss_weight must be non-negative.")
        if self.target_background_contrastive_loss_weight < 0.0:
            raise ValueError("target_background_contrastive_loss_weight must be non-negative.")
        if self.target_background_contrastive_margin < 0.0:
            raise ValueError("target_background_contrastive_margin must be non-negative.")
        if self.background_aug_context_dilation < 0:
            raise ValueError("background_aug_context_dilation must be non-negative.")
        if not 0.0 <= self.background_aug_keep_threshold <= 1.0:
            raise ValueError("background_aug_keep_threshold must be in [0, 1].")
        if self.background_aug_mode not in {"random_color", "noise", "mixed", "shuffle"}:
            raise ValueError(f"Unknown background_aug_mode={self.background_aug_mode!r}.")
        if not 0.0 <= self.background_aug_noise_p <= 1.0:
            raise ValueError("background_aug_noise_p must be in [0, 1].")
        if not 0.0 <= self.background_aug_min_strength <= 1.0:
            raise ValueError("background_aug_min_strength must be in [0, 1].")
        if not 0.0 <= self.background_aug_max_strength <= 1.0:
            raise ValueError("background_aug_max_strength must be in [0, 1].")
        if self.background_aug_min_strength > self.background_aug_max_strength:
            raise ValueError("background_aug_min_strength must be <= background_aug_max_strength.")
        if not 0.0 <= self.background_consistency_aug_min_strength <= 1.0:
            raise ValueError("background_consistency_aug_min_strength must be in [0, 1].")
        if not 0.0 <= self.background_consistency_aug_max_strength <= 1.0:
            raise ValueError("background_consistency_aug_max_strength must be in [0, 1].")
        if self.background_consistency_aug_min_strength > self.background_consistency_aug_max_strength:
            raise ValueError(
                "background_consistency_aug_min_strength must be <= background_consistency_aug_max_strength."
            )
        if self.adapter_hidden_dim <= 0:
            raise ValueError("adapter_hidden_dim must be positive.")
        if self.context_dilation < 0:
            raise ValueError("context_dilation must be non-negative.")
        if self.num_target_tokens < 0:
            raise ValueError("num_target_tokens must be non-negative.")
        if self.mask_geometry_token_gate_init < 0.0:
            raise ValueError("mask_geometry_token_gate_init must be non-negative.")
        if self.max_instance_object_tokens < 0:
            raise ValueError("max_instance_object_tokens must be non-negative.")
        if self.instance_object_token_gate_init < 0.0:
            raise ValueError("instance_object_token_gate_init must be non-negative.")
        if not 0.0 <= self.instance_object_token_min_area <= 1.0:
            raise ValueError("instance_object_token_min_area must be in [0, 1].")
        if self.instance_object_token_mask_bias_scale < 0.0:
            raise ValueError("instance_object_token_mask_bias_scale must be non-negative.")
