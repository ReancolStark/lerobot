import torch
import torch.nn as nn

from lerobot.policies.customACT.segment_understanding.configuration_segment_understanding import (
    SegmentUnderstandingConfig,
)


class SegmentUnderstandingEmbedding(nn.Module):
    """Object-kinematics token builder for ACT.

    Output token order is:
        [fk_token, object_1, ..., object_K, fk_conditioned_target_token]
    """

    def __init__(self, config: SegmentUnderstandingConfig):
        super().__init__()
        self.config = config
        self.object_encoder = ObjectTokenEncoder(config)
        self.fk_encoder = FKEncoder(config)

        self.object_pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_yolo_objects, config.output_dim)
        )

        if config.object_refine_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.output_dim,
                nhead=config.fusion_num_heads,
                dim_feedforward=config.output_dim * 4,
                dropout=config.fusion_dropout,
                batch_first=True,
                norm_first=True,
            )
            self.object_refiner = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.object_refine_layers,
            )
        else:
            self.object_refiner = None

        self.fk_to_object_attention = nn.MultiheadAttention(
            config.output_dim,
            config.fusion_num_heads,
            dropout=config.fusion_dropout,
            batch_first=True,
        )
        self.target_fusion = nn.Sequential(
            nn.LayerNorm(config.output_dim * 2),
            nn.Linear(config.output_dim * 2, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    @staticmethod
    def masked_visual_pool(visual_features: torch.Tensor, object_masks: torch.Tensor) -> torch.Tensor:
        # visual_features: [B, D, H, W], object_masks: [B, K, 1, H, W]
        weights = object_masks.to(dtype=visual_features.dtype, device=visual_features.device).clamp(0.0, 1.0)
        features = visual_features.unsqueeze(1)
        denom = weights.sum(dim=(-2, -1)).clamp(min=1e-6)
        pooled = (features * weights).sum(dim=(-2, -1)) / denom
        return pooled

    def forward(
        self,
        object_features: torch.Tensor,
        object_mask: torch.Tensor,
        fk: torch.Tensor,
        visual_features: torch.Tensor,
        object_visual_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        object_features = object_features.to(device=visual_features.device, dtype=visual_features.dtype)
        object_mask = object_mask.to(device=visual_features.device)
        fk = fk.to(device=visual_features.device, dtype=visual_features.dtype)
        object_visual = self.masked_visual_pool(visual_features, object_visual_masks)
        object_tokens = self.object_encoder(object_features, object_mask, object_visual)
        object_tokens = object_tokens + self.object_pos_embedding.to(
            dtype=object_tokens.dtype,
            device=object_tokens.device,
        )
        object_tokens = object_tokens.masked_fill(~object_mask.unsqueeze(-1), 0.0)

        safe_padding_mask = ~object_mask
        no_object = ~object_mask.any(dim=1)
        if no_object.any():
            safe_padding_mask = safe_padding_mask.clone()
            safe_padding_mask[no_object] = False

        if self.object_refiner is not None:
            object_tokens = self.object_refiner(
                object_tokens,
                src_key_padding_mask=safe_padding_mask,
            )
            object_tokens = object_tokens.masked_fill(~object_mask.unsqueeze(-1), 0.0)

        fk_token = self.fk_encoder(fk)
        attended_objects = self.fk_to_object_attention(
            query=fk_token.unsqueeze(1),
            key=object_tokens,
            value=object_tokens,
            key_padding_mask=safe_padding_mask,
            need_weights=False,
        )[0].squeeze(1)
        target_token = self.target_fusion(torch.cat([fk_token, attended_objects], dim=-1))

        tokens = torch.cat(
            [
                fk_token.unsqueeze(1),
                object_tokens,
                target_token.unsqueeze(1),
            ],
            dim=1,
        )
        valid_mask = torch.cat(
            [
                torch.ones(object_mask.shape[0], 1, dtype=torch.bool, device=object_mask.device),
                object_mask,
                torch.ones(object_mask.shape[0], 1, dtype=torch.bool, device=object_mask.device),
            ],
            dim=1,
        )
        return tokens, valid_mask, target_token


class ObjectTokenEncoder(nn.Module):
    def __init__(self, config: SegmentUnderstandingConfig):
        super().__init__()
        self.config = config
        self.cls_embed = nn.Embedding(config.num_classes, config.cls_embed_dim)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(config.object_numeric_dim, config.r_hidden_dim),
            nn.GELU(),
            nn.Linear(config.r_hidden_dim, config.r_hidden_dim),
            nn.GELU(),
        )
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(config.output_dim),
            nn.Linear(config.output_dim, config.object_hidden_dim),
            nn.GELU(),
        )
        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(config.r_hidden_dim + config.object_hidden_dim),
            nn.Linear(config.r_hidden_dim + config.object_hidden_dim, config.reliability_hidden_dim),
            nn.GELU(),
            nn.Linear(config.reliability_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(config.cls_embed_dim + config.r_hidden_dim + config.object_hidden_dim),
            nn.Linear(
                config.cls_embed_dim + config.r_hidden_dim + config.object_hidden_dim,
                config.output_dim,
            ),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def forward(
        self,
        object_features: torch.Tensor,
        object_mask: torch.Tensor,
        object_visual: torch.Tensor,
    ) -> torch.Tensor:
        cls_ids = object_features[..., 0].long().clamp(0, self.config.num_classes - 1)
        numeric = object_features[..., 1:]

        cls_embed = self.cls_embed(cls_ids)
        numeric_embed = self.numeric_encoder(numeric)
        visual_embed = self.visual_encoder(object_visual)
        reliability = self.reliability_gate(torch.cat([numeric_embed, visual_embed], dim=-1))

        tokens = self.token_proj(torch.cat([cls_embed, numeric_embed, visual_embed], dim=-1))
        tokens = tokens * (0.1 + 0.9 * reliability)
        return tokens.masked_fill(~object_mask.unsqueeze(-1), 0.0)


class FKEncoder(nn.Module):
    def __init__(self, config: SegmentUnderstandingConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, config.fk_hidden_dim),
            nn.GELU(),
            nn.Linear(config.fk_hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def forward(self, fk: torch.Tensor) -> torch.Tensor:
        # Raw FK is [x, y, z, flattened 3x3 rotation]. Use the first two rotation columns as 6D rotation.
        position = fk[:, :3]
        rotation = fk[:, 3:].reshape(fk.shape[0], 3, 3)
        rotation_6d = rotation[:, :, :2].transpose(1, 2).reshape(fk.shape[0], 6)
        return self.mlp(torch.cat([position, rotation_6d], dim=-1))
