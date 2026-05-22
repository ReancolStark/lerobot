#!/usr/bin/env python
"""Preview YOLO-guided background replacement on one image.

Run:
    python custom/scripts/yolo_background_replace_preview.py

The script opens a file chooser for an image, asks for a replacement strength
in [0, 1], runs the same YOLO mask and background replacement code used by
customACT mask_weight, then shows a side-by-side preview without saving images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.policies.customACT.mask_weight.configuration_mask_weight import MaskWeightConfig
from lerobot.policies.customACT.mask_weight.mask_weight import (
    make_mask_guided_background_augmentation,
    yolo_result_to_soft_mask,
)
from lerobot.policies.customACT.segment_understanding.configuration_segment_understanding import (
    SegmentUnderstandingConfig,
)
from lerobot.policies.customACT.segment_understanding.utils.yolo_data_processer import (
    YoloDataProcessor,
)


def _choose_image_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    filename = filedialog.askopenfilename(
        title="Choose an image to preview",
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(filename) if filename else None


def _choose_model_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    filename = filedialog.askopenfilename(
        title="Default YOLO weight was not found. Choose a YOLO .pt weight",
        filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(filename) if filename else None


def _ask_strength(default: float = 0.9) -> float | None:
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    value = simpledialog.askfloat(
        "Background replacement strength",
        "Enter background replacement strength in [0, 1]:",
        initialvalue=default,
        minvalue=0.0,
        maxvalue=1.0,
    )
    root.destroy()
    return None if value is None else float(value)


def _read_rgb_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode output image: {path}")
    encoded.tofile(str(path))


def _show_rgb_image(window_name: str, image_rgb: np.ndarray, max_width: int = 1800, max_height: int = 950) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, bgr)
    print("Preview window opened. Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def _to_image_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = torch.from_numpy(image_rgb).float() / 255.0
    return image.permute(2, 0, 1).unsqueeze(0).contiguous()


def _tensor_to_rgb_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return (image.numpy() * 255.0).round().astype(np.uint8)


def _mask_to_rgb_image(mask: torch.Tensor) -> np.ndarray:
    mask_2d = mask.detach().cpu().squeeze().clamp(0, 1).numpy()
    mask_u8 = (mask_2d * 255.0).round().astype(np.uint8)
    heat_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _add_label(image_rgb: np.ndarray, label: str) -> np.ndarray:
    label_h = 36
    out = np.full((image_rgb.shape[0] + label_h, image_rgb.shape[1], 3), 245, dtype=np.uint8)
    out[label_h:] = image_rgb
    cv2.putText(
        out,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return out


def _make_preview(
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    replaced_rgb: np.ndarray,
    strength: float,
) -> np.ndarray:
    panels = [
        _add_label(original_rgb, "original"),
        _add_label(overlay_rgb, "YOLO overlay"),
        _add_label(mask_rgb, "soft mask"),
        _add_label(replaced_rgb, f"replaced strength={strength:.2f}"),
    ]
    return np.concatenate(panels, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="Image path. If omitted, opens a file chooser.")
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Background replacement strength in [0, 1]. If omitted, opens an input dialog.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO .pt path. Defaults to SegmentUnderstandingConfig.yolo_path.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda:0, etc. Defaults to auto.")
    parser.add_argument("--save", action="store_true", help="Save preview images. Default is display only.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory used only with --save.")
    parser.add_argument(
        "--mode",
        choices=["random_color", "noise", "mixed", "shuffle"],
        default="mixed",
        help="Background replacement mode.",
    )
    parser.add_argument("--context-dilation", type=int, default=21, help="Pixels to preserve around YOLO mask.")
    parser.add_argument("--keep-threshold", type=float, default=0.05, help="Mask threshold for preserved area.")
    parser.add_argument("--mask-blur-kernel-size", type=int, default=7)
    parser.add_argument("--mask-blur-sigma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = args.image or _choose_image_path()
    if image_path is None:
        print("Canceled: no image selected.")
        return

    strength = args.strength
    if strength is None:
        strength = _ask_strength()
    if strength is None:
        print("Canceled: no strength entered.")
        return
    if not 0.0 <= strength <= 1.0:
        raise ValueError("--strength must be in [0, 1].")

    seg_config = SegmentUnderstandingConfig()
    if args.model is not None:
        seg_config.yolo_path = args.model
    elif not Path(seg_config.yolo_path).exists():
        model_path = _choose_model_path()
        if model_path is None:
            print(f"Canceled: default YOLO weight was not found: {seg_config.yolo_path}")
            return
        seg_config.yolo_path = str(model_path)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = YoloDataProcessor(seg_config, device=device)

    image_rgb = _read_rgb_image(image_path)
    results = processor.get_yolo_results([image_rgb])
    yolo_mask = yolo_result_to_soft_mask(
        results,
        kernel_size=args.mask_blur_kernel_size,
        sigma=args.mask_blur_sigma,
    ).cpu()

    image_tensor = _to_image_tensor(image_rgb)
    config = MaskWeightConfig(
        background_aug_p=1.0,
        background_aug_context_dilation=args.context_dilation,
        background_aug_keep_threshold=args.keep_threshold,
        background_aug_mode=args.mode,
    )
    apply_mask = torch.ones(image_tensor.shape[0], 1, 1, 1, dtype=torch.bool)
    full_replaced_tensor, debug = make_mask_guided_background_augmentation(
        image_tensor,
        yolo_mask,
        config,
        apply_mask=apply_mask,
    )
    replaced_tensor = image_tensor * (1.0 - strength) + full_replaced_tensor * strength
    debug = dict(debug)
    debug["strength_mean"] = torch.tensor(float(strength))
    debug["image_delta_l1_full_strength"] = debug["image_delta_l1"]
    debug["image_delta_l1"] = (replaced_tensor.detach() - image_tensor.detach()).abs().float().mean()

    overlay_rgb = processor.draw_results_on_frame(image_rgb, results[0])
    mask_rgb = _mask_to_rgb_image(yolo_mask[0])
    replaced_rgb = _tensor_to_rgb_image(replaced_tensor)
    preview_rgb = _make_preview(image_rgb, overlay_rgb, mask_rgb, replaced_rgb, strength)

    num_detections = 0 if results[0].boxes is None else len(results[0].boxes)
    print(f"YOLO model: {seg_config.yolo_path}")
    print(f"Image: {image_path}")
    print(f"Detections: {num_detections}")
    print(f"Strength: {strength:.3f}")
    for key, value in debug.items():
        print(f"{key}: {float(value.detach().cpu().item()):.6f}")
    if args.save:
        output_dir = args.output_dir or image_path.parent
        safe_strength = f"{strength:.2f}".replace(".", "p")
        preview_path = output_dir / f"{image_path.stem}_mask_weight_preview_s{safe_strength}.jpg"
        replaced_path = output_dir / f"{image_path.stem}_background_replaced_s{safe_strength}.png"
        mask_path = output_dir / f"{image_path.stem}_soft_mask.png"
        overlay_path = output_dir / f"{image_path.stem}_yolo_overlay.jpg"

        _write_rgb_image(preview_path, preview_rgb)
        _write_rgb_image(replaced_path, replaced_rgb)
        _write_rgb_image(mask_path, mask_rgb)
        _write_rgb_image(overlay_path, overlay_rgb)

        print(f"Preview saved: {preview_path}")
        print(f"Replaced image saved: {replaced_path}")
        print(f"Soft mask saved: {mask_path}")
        print(f"YOLO overlay saved: {overlay_path}")

    _show_rgb_image("YOLO background replacement preview", preview_rgb)


if __name__ == "__main__":
    main()
