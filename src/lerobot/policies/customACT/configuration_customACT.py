#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.customACT.history_obs_state.configuration_history_obs import HistoryObsConfig, HistoryLSTMConfig, HistoryConv1dConfig
from lerobot.policies.customACT.key_history_state.configuration_key_history import KeyHistoryTokenConfig
from lerobot.policies.customACT.model_adaptive_chunk.configuration_model_adaptive_chunk import (
    ReplanScoreAdaptiveChunkingConfig,
)
from lerobot.policies.customACT.recovery_adaptive_chunking.configuration_recovery_adaptive_chunking import (
    RecoveryAdaptiveChunkingConfig,
)
from lerobot.policies.customACT.segment_understanding.configuration_segment_understanding import SegmentUnderstandingConfig

AdaptiveActionChunkingConfig = RecoveryAdaptiveChunkingConfig

@PreTrainedConfig.register_subclass("customACT")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """
# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # recovery_token+自适应动作块统一配置。
    # Unified config for the recovery token, recovery score pseudo-label, and adaptive chunk controller.
    # Existing top-level fields remain supported; __post_init__ syncs them into this nested config.
    recovery_adaptive_chunking: RecoveryAdaptiveChunkingConfig = field(
        default_factory=RecoveryAdaptiveChunkingConfig
    )
    # False keeps legacy top-level fields as the source of truth. Set True for new
    # experiments that configure recovery token + AAC only through the nested config.
    use_recovery_adaptive_chunking_config: bool = True

    # 是否开启计算状态平滑度指标（state smoothness），用于评估动作块内状态变化的平滑程度。这个指标可以帮助分析自适应动作块的效果，尤其是在运动突变发生时状态的变化情况。
    compute_state_smoothness: bool = True


 # —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # 历史状态序列
    # 自定义历史信息参数。
    # 注意：如果你要开关历史模块，建议只改这个配置文件，不要在训练命令里覆盖这些参数。

    # 旧版 history_obs_states 模块的历史帧数。0 表示完全关闭旧模块；>0 表示使用旧的历史状态 token。
    # 这个旧模块只看 observation.state 历史，不使用 action.history；新实验建议保持 0。
    n_history_obs_states: int = 0

    # 旧版 history_obs_states 模块类型。"conv1d" 使用一维卷积历史编码；"lstm" 使用 LSTM 历史编码。
    ho_type: str = 'conv1d'
    # his_obs_config: HistoryObsConfig | None = None

    # 旧版 LSTM 历史模块的输入维度。通常应等于 observation.state 的维度。
    ho_input_size: int = 6
    # 旧版 LSTM 历史模块的隐藏层维度，越大容量越强，但参数和过拟合风险也越高。
    ho_hidden_size: int = 64
    # 旧版 LSTM 历史模块层数。1 层最轻量，更多层通常需要更多数据。
    ho_num_layers: int = 1
    # 旧版 Conv1d 历史模块输出多少个历史 token，也就是把历史序列压缩成几段。
    ho_history_segment_num: int = 4
    # 旧版 Conv1d 分段的非均匀程度。越大越偏向保留靠近当前时刻的细节。
    ho_history_segment_alpha: float = 0.2
    # 旧版 Conv1d 分段权重衰减方式。"linear" 线性衰减；"exponential" 指数衰减；None 表示不使用。
    ho_history_segment_decay: str = 'linear'
    # 旧版事件先验权重，用于控制运动突变先验在旧 history token pooling 中的影响。
    ho_event_prior_weight: float = 1.0
    # 旧版历史模块辅助动作预测损失权重。只有 n_history_obs_states > 0 时才会生效。
    ho_aux_loss_weight: float = 0.05



# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # 是否启用新的“失败感知本体历史 Recovery Token 模块”。
    # False：完全走原始 ACT 路径，不需要历史 state/action 字段。
    # True：训练时数据集自动构造 observation.state.history、action.history、history_mask；
    #       推理时 policy 内部维护 state/action 历史 buffer。
    use_recovery_history_token: bool = False
    # Recovery 模块使用的历史窗口长度 H。
    # state.history = [s_{t-H+1}, ..., s_t]，action.history = [a_{t-H}, ..., a_{t-1}]。
    history_len: int = 64
    # 把 H 个历史时刻压缩成多少个 recovery tokens。4 表示输出 [B, 4, dim_model]。
    # 如果 history_len 不能整除该值，最后一段会自动包含剩余历史帧。
    history_num_segments: int = 4
    # Recovery 模块内部因果卷积的隐藏维度。只影响新增历史模块，不改变 ACT 主干 dim_model。
    history_hidden_dim: int = 256
    # 因果卷积核大小。3 表示每层卷积最多看当前和左侧两个位置，再由 dilation 扩大感受野。
    history_conv_kernel_size: int = 3
    # 多层膨胀因果卷积的 dilation 设置。[1, 2, 4, 8] 能覆盖短期到较长期的历史变化。
    # 所有卷积都只做左侧 padding，保证不会泄露未来信息。
    history_conv_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    # Recovery 模块内部 dropout，用于减少小数据集上历史 token 的过拟合。
    history_dropout: float = 0.1
    # 是否使用动作-状态执行偏差 exec_error = projected_action - state。
    # True：让模块显式感知“动作发出后状态没有按预期变化”的失败线索。
    # False：exec_error 置零，只使用相对位移、速度和加速度。
    use_action_state_error: bool = True
    # 历史动作预测辅助损失权重。目标是 batch["action"][:, 0, :]，用于让 recovery token 保留动作相关信息。
    history_action_loss_weight: float = 0.05
    # 事件先验对齐辅助损失权重。该损失把 learned event score 弱约束到运动突变先验附近，不使用人工阶段标签。
    event_prior_loss_weight: float = 0.01
    # Recovery tokens 的位置编码类型。目前只实现 "learned"，即每个 recovery segment 一个可学习位置向量。
    recovery_token_pos_embed: str = "learned"



# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # 关键帧历史状态 Token 模块（No-failure Key History Token Module）。
    # Event-guided key historical state token module.
    # This no-failure history module only uses observation.state.history and history_mask.
    # It does not use action.history, execution error, failure labels, or recovery scores.
    use_key_history_token: bool = False
    key_history_len: int = 64
    key_history_num_segments: int = 4
    key_history_hidden_dim: int = 256
    key_history_conv_kernel_size: int = 3
    key_history_conv_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    key_history_dropout: float = 0.1
    key_history_prior_scale_init: float = 0.5
    key_history_selection_temperature: float = 1.0
    key_history_action_loss_weight: float = 0.05
    key_history_event_loss_weight: float = 0.01
    key_history_token_pos_embed: str = "learned"
    # —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————



    # Segment understanding config
    use_segment_understanding: bool = False
    seg_config: SegmentUnderstandingConfig = field(default_factory=SegmentUnderstandingConfig)



# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # 自适应动作块配置（Adaptive Action Chunking Config）。
    # 是否启用推理阶段的历史感知自适应动作块。False 时保持原始 ACT 固定执行长度。
    use_adaptive_action_chunking: bool = False
    # 自适应动作块的详细配置，包括 chunk 长度阈值、历史状态阈值、动态加权和调试输出。
    adaptive_action_chunking: AdaptiveActionChunkingConfig = field(
        default_factory=AdaptiveActionChunkingConfig
    )

    # Replan-score-only adaptive chunking. This replaces the three-regime
    # controller at inference time and maps recovery/replan score directly to
    # the number of actions inserted into the execution queue.
    # 启用replan-score-only的自适应动作块。这在推理阶段替换原来的三阶段控制器，直接将recovery/replan分数映射到执行队列中插入的动作数量。
    use_replan_score_adaptive_chunking: bool = False
    replan_score_adaptive_chunking: ReplanScoreAdaptiveChunkingConfig = field(
        default_factory=ReplanScoreAdaptiveChunkingConfig
    )




# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————






    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

 # ————————————————————————————————————————————————————————————————————————————————————

    # Architecture.
    # Vision backbone.
    #  "dino", "resnet18", "convnext"
    vision_backbone: str = "resnet18"
    # vision_backbone: str = "convnext"  
    # vision_backbone: str = "dino"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None



    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.use_recovery_adaptive_chunking_config:
            self._apply_recovery_adaptive_chunking_config()
        else:
            self.recovery_adaptive_chunking = self.get_recovery_adaptive_chunking_config()

        """Input validation (not exhaustive)."""
        # if not self.vision_backbone.startswith("resnet"):
        #     raise ValueError(
        #         f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
        #     )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.use_recovery_history_token and self.use_key_history_token:
            raise ValueError(
                "`use_recovery_history_token` and `use_key_history_token` cannot both be enabled. "
                "Disable recovery history when running the no-failure key-history experiment."
            )
        if self.use_recovery_history_token:
            if self.history_len <= 0:
                raise ValueError("`history_len` must be positive when recovery history token is enabled.")
            if self.history_num_segments <= 0:
                raise ValueError("`history_num_segments` must be positive.")
            if self.history_len < self.history_num_segments:
                raise ValueError("`history_len` must be >= `history_num_segments`.")
            if self.history_hidden_dim <= 0:
                raise ValueError("`history_hidden_dim` must be positive.")
            if self.history_conv_kernel_size <= 0:
                raise ValueError("`history_conv_kernel_size` must be positive.")
            if not self.history_conv_dilations:
                raise ValueError("`history_conv_dilations` cannot be empty.")
            if self.recovery_token_pos_embed != "learned":
                raise ValueError("Only `recovery_token_pos_embed='learned'` is currently supported.")
        if self.use_key_history_token:
            if self.key_history_len <= 0:
                raise ValueError("`key_history_len` must be positive when key history token is enabled.")
            if self.key_history_num_segments <= 0:
                raise ValueError("`key_history_num_segments` must be positive.")
            if self.key_history_len < self.key_history_num_segments:
                raise ValueError("`key_history_len` must be >= `key_history_num_segments`.")
            if self.key_history_hidden_dim <= 0:
                raise ValueError("`key_history_hidden_dim` must be positive.")
            if self.key_history_conv_kernel_size <= 0:
                raise ValueError("`key_history_conv_kernel_size` must be positive.")
            if not self.key_history_conv_dilations:
                raise ValueError("`key_history_conv_dilations` cannot be empty.")
            if self.key_history_selection_temperature <= 0:
                raise ValueError("`key_history_selection_temperature` must be positive.")
            if self.key_history_token_pos_embed != "learned":
                raise ValueError("Only `key_history_token_pos_embed='learned'` is currently supported.")
        if self.use_adaptive_action_chunking:
            aac = self.adaptive_action_chunking
            if aac.state_history_len < 2:
                raise ValueError("`adaptive_action_chunking.state_history_len` must be >= 2.")
            if aac.min_chunk_size <= 0:
                raise ValueError("`adaptive_action_chunking.min_chunk_size` must be positive.")
            if aac.max_chunk_size < 0:
                raise ValueError("`adaptive_action_chunking.max_chunk_size` cannot be negative.")
            if aac.max_chunk_size and aac.max_chunk_size < aac.min_chunk_size:
                raise ValueError(
                    "`adaptive_action_chunking.max_chunk_size` must be >= min_chunk_size when set."
                )
            if aac.stable_chunk_multiplier <= 0 or aac.unstable_chunk_multiplier <= 0:
                raise ValueError("Adaptive chunk multipliers must be positive.")
            if aac.volatility_low < 0 or aac.volatility_high < 0 or aac.acceleration_high < 0:
                raise ValueError("Adaptive chunk motion thresholds cannot be negative.")
            if aac.volatility_low > aac.volatility_high:
                raise ValueError("`adaptive_action_chunking.volatility_low` must be <= volatility_high.")
            if aac.action_uncertainty_low < 0 or aac.action_uncertainty_high < 0:
                raise ValueError("Adaptive chunk action uncertainty thresholds cannot be negative.")
            if aac.action_uncertainty_low > aac.action_uncertainty_high:
                raise ValueError(
                    "`adaptive_action_chunking.action_uncertainty_low` must be <= action_uncertainty_high."
                )
            if aac.debug_print_every <= 0:
                raise ValueError("`adaptive_action_chunking.debug_print_every` must be positive.")
            if aac.debug_print_num_actions < 0 or aac.debug_print_action_dims < 0:
                raise ValueError("Adaptive chunk debug preview sizes cannot be negative.")
        if self.use_replan_score_adaptive_chunking:
            if self.temporal_ensemble_coeff is not None:
                raise ValueError(
                    "`use_replan_score_adaptive_chunking` is only supported by the queued inference path. "
                    "Disable temporal ensembling for this controller."
                )
            if self.use_adaptive_action_chunking:
                raise ValueError(
                    "`use_replan_score_adaptive_chunking` and `use_adaptive_action_chunking` are mutually "
                    "exclusive. Disable the old three-regime adaptive chunk controller first."
                )
            if (
                not self.use_recovery_history_token
                and self.replan_score_adaptive_chunking.fallback_replan_score is None
            ):
                raise ValueError(
                    "`use_replan_score_adaptive_chunking` requires recovery history token scoring, or a "
                    "`replan_score_adaptive_chunking.fallback_replan_score`."
                )
            self.replan_score_adaptive_chunking.validate()
        self.recovery_adaptive_chunking.validate()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.use_recovery_history_token and not self.robot_state_feature:
            raise ValueError("`observation.state` is required when recovery history token is enabled.")

    def get_HistoryObsConfig(self) -> HistoryObsConfig:
        if self.ho_type == 'lstm':
            return HistoryLSTMConfig(
                input_size=self.ho_input_size,
                hidden_size=self.ho_hidden_size,
                num_layers=self.ho_num_layers
            )
        elif self.ho_type == 'conv1d':
            return HistoryConv1dConfig(
                history_segment_num=self.ho_history_segment_num,
                history_segment_alpha=self.ho_history_segment_alpha,
                history_segment_decay=self.ho_history_segment_decay,
                event_prior_weight=self.ho_event_prior_weight
            )
        else:
            raise ValueError(f"Unknown ho_type: {self.ho_type}")

    def get_KeyHistoryTokenConfig(self) -> KeyHistoryTokenConfig:
        return KeyHistoryTokenConfig(
            enabled=self.use_key_history_token,
            history_len=self.key_history_len,
            num_segments=self.key_history_num_segments,
            hidden_dim=self.key_history_hidden_dim,
            conv_kernel_size=self.key_history_conv_kernel_size,
            conv_dilations=self.key_history_conv_dilations,
            dropout=self.key_history_dropout,
            prior_scale_init=self.key_history_prior_scale_init,
            selection_temperature=self.key_history_selection_temperature,
            action_loss_weight=self.key_history_action_loss_weight,
            event_loss_weight=self.key_history_event_loss_weight,
            token_pos_embed=self.key_history_token_pos_embed,
        )

    def get_recovery_adaptive_chunking_config(self) -> RecoveryAdaptiveChunkingConfig:
        return RecoveryAdaptiveChunkingConfig.from_legacy_config(self)

    def _apply_recovery_adaptive_chunking_config(self) -> None:
        rac = self.recovery_adaptive_chunking
        self.use_recovery_history_token = rac.enabled and rac.use_recovery_token
        self.use_adaptive_action_chunking = (
            rac.enabled
            and rac.use_adaptive_action_chunking
            and not self.use_replan_score_adaptive_chunking
        )

        self.history_len = rac.history_len
        self.history_num_segments = rac.num_segments
        self.history_hidden_dim = rac.hidden_dim
        self.history_conv_kernel_size = rac.conv_kernel_size
        self.history_conv_dilations = list(rac.conv_dilations)
        self.history_dropout = rac.dropout
        self.use_action_state_error = rac.use_action_state_error
        self.history_action_loss_weight = rac.action_loss_weight
        self.event_prior_loss_weight = rac.event_prior_loss_weight
        self.recovery_token_pos_embed = rac.token_pos_embed
        for name in (
            "state_history_len",
            "min_chunk_size",
            "max_chunk_size",
            "stable_chunk_multiplier",
            "unstable_chunk_multiplier",
            "chunk_smoothing",
            "max_chunk_delta",
            "volatility_low",
            "volatility_high",
            "acceleration_high",
            "recovery_score_high",
            "action_uncertainty_low",
            "action_uncertainty_high",
            "action_curvature_weight",
            "stable_old_action_weight",
            "nominal_old_action_weight",
            "unstable_old_action_weight",
            "min_old_action_weight",
            "max_old_action_weight",
            "transition_blend_steps",
            "transition_blend_old_action_weight",
            "debug_print_chunks",
            "debug_print_every",
            "debug_print_num_actions",
            "debug_print_action_dims",
        ):
            setattr(self.adaptive_action_chunking, name, getattr(rac, name))

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size)) # 0,1,2,3,4,5,6...chunk_size-1

    @property
    def history_obs_state_delta_indices(self) -> list: # -steps+1...-2,-1,0  [-steps+1, 1)
        return list(range(-self.n_history_obs_states+1, 1)) if( self.n_history_obs_states > 0 ) else None 

    @property
    def recovery_state_history_delta_indices(self) -> list | None:
        return list(range(-self.history_len + 1, 1)) if self.use_recovery_history_token else None

    @property
    def recovery_action_history_delta_indices(self) -> list | None:
        return list(range(-self.history_len, 0)) if self.use_recovery_history_token else None

    @property
    def key_state_history_delta_indices(self) -> list | None:
        return list(range(-self.key_history_len + 1, 1)) if self.use_key_history_token else None

    @property
    def reward_delta_indices(self) -> None:
        return None


