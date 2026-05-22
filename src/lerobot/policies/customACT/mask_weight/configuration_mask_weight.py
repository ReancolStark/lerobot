from dataclasses import dataclass


@dataclass
class MaskWeightConfig:
    # "adapter" injects YOLO masks as visual-token guidance.
    # "legacy_multiply" keeps the old fixed feature multiplication for ablations.
    mode: str = "adapter"

    # Legacy multiply: features = features * (beta + alpha * mask).
    alpha: float = 1.0
    beta: float = 0.5

    # Mask-guided adapter switches.
    use_spatial_embedding: bool = True
    use_residual_gate: bool = True
    use_region_modulation: bool = True
    use_target_tokens: bool = False

    # Mask-guided adapter hyperparameters.
    adapter_hidden_dim: int = 128
    gate_init: float = 0.1
    context_dilation: int = 3
    mask_dropout_p: float = 0.1
    num_target_tokens: int = 2
    target_boost_init: float = 0.15
    context_boost_init: float = 0.05
    background_suppress_init: float = 0.18
    region_modulation_max: float = 0.5

    # YOLO mask post-processing.
    mask_blur_kernel_size: int = 7
    mask_blur_sigma: float = 2.0

    # Training-time background counterfactuals.
    use_background_augmentation: bool = True
    # Adds a second, background-augmented forward pass and penalizes action drift.
    use_background_consistency: bool = True
    # Used for primary-forward random augmentation only when background consistency is off.
    background_aug_p: float = 0.5
    background_consistency_loss_weight: float = 0.1
    background_aug_context_dilation: int = 21
    background_aug_keep_threshold: float = 0.05
    background_aug_mode: str = "mixed"  # "random_color", "noise", "mixed", or "shuffle".
    background_aug_noise_p: float = 0.5
    background_aug_min_strength: float = 0.3
    background_aug_max_strength: float = 1.0
    background_consistency_aug_min_strength: float = 0.75
    background_consistency_aug_max_strength: float = 1.0

    def __post_init__(self):
        if self.mode not in {"adapter", "legacy_multiply"}:
            raise ValueError(f"Unknown MaskWeightConfig.mode={self.mode!r}.")
        if not 0.0 <= self.mask_dropout_p <= 1.0:
            raise ValueError("mask_dropout_p must be in [0, 1].")
        if not 0.0 <= self.background_aug_p <= 1.0:
            raise ValueError("background_aug_p must be in [0, 1].")
        if self.background_consistency_loss_weight < 0.0:
            raise ValueError("background_consistency_loss_weight must be non-negative.")
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
        for name in ("target_boost_init", "context_boost_init", "background_suppress_init"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.region_modulation_max <= 0.0:
            raise ValueError("region_modulation_max must be positive.")
