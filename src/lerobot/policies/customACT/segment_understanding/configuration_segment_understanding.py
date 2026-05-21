from dataclasses import dataclass, field


@dataclass
class SegmentUnderstandingConfig:
    # YOLO
    yolo_path: str = "runs/segment/grab_block/weights/best.pt"
    tracker_path: str = "custom/scripts/yolo/botsort.yaml"
    camera_name: str = "robot1"
    max_yolo_objects: int = 5
    num_classes: int = 4

    # Wrist-camera end-effector prior in normalized image coordinates.
    ee_anchor: list[float] = field(default_factory=lambda: [0.5, 1.0])
    ee_anchor_box_size: list[float] = field(default_factory=lambda: [0.28, 0.22])
    anchor_distance_weight: float = 0.25
    anchor_area_weight: float = 0.05
    overlap_ambiguity_weight: float = 0.05
    use_tracking_in_eval: bool = True
    track_age_weight: float = 0.05
    track_jump_weight: float = 0.35
    track_memory_max_age: int = 30

    # FK
    urdf_path: str = "custom/config/SO101/so101_new_calib.urdf"
    ee_frame_name: str = "gripper_frame_link"

    # Object token encoder.
    # [cx, cy, w, h, aspect, dx, dy, dist, sin_theta, cos_theta,
    #  conf, mask_area, mask_bbox_ratio, ee_anchor_iou, max_peer_iou]
    object_numeric_dim: int = 15
    cls_embed_dim: int = 64
    r_hidden_dim: int = 128
    object_hidden_dim: int = 256
    object_refine_layers: int = 1
    reliability_hidden_dim: int = 64

    # FK encoder. The raw FK input is [xyz, 3x3 rotation], internally reduced to xyz + 6D rotation.
    fk_input_dim: int = 12
    fk_hidden_dim: int = 128

    # Fusion.
    output_dim: int = 512
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1

    # V3 servo residual branch. The head predicts a short-horizon action from segment/FK context,
    # and the main ACT action receives a small residual correction from it.
    use_servo_residual: bool = True
    servo_hidden_dim: int = 512
    servo_residual_scale: float = 0.1
    servo_aux_loss_weight: float = 0.05
    servo_target_horizon: int = 10
