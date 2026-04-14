"""Load VisDrone real frame + bboxes for power_benchmark.py"""
import os
import numpy as np
import cv2
from pathlib import Path

VISDRONE_DIR = os.path.expanduser("~/datasets/visdrone/VisDrone2019-DET-val")

def load_visdrone_pool(max_frames=50, min_box=16, max_box=320):
    """
    Load 1 representative frame for pixel data + pool of real bboxes.
    Bboxes from other frames are rescaled into canonical frame coordinates.
    """
    img_dir = Path(VISDRONE_DIR) / "images"
    ann_dir = Path(VISDRONE_DIR) / "annotations"
    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images in {img_dir}")

    # Canonical frame = real drone scene pixel data
    canonical_img = cv2.imread(str(img_paths[0]))
    canonical_img = cv2.cvtColor(canonical_img, cv2.COLOR_BGR2RGB)
    H, W = canonical_img.shape[:2]
    print(f"  Canonical frame: {img_paths[0].name} ({W}x{H})")

    # Pool real bboxes from many frames, rescale to canonical coords
    bbox_pool = []
    for img_path in img_paths[:max_frames]:
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue
        this_img = cv2.imread(str(img_path))
        if this_img is None:
            continue
        ih, iw = this_img.shape[:2]
        sx, sy = W / iw, H / ih
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    x, y, w, h = (int(parts[0]), int(parts[1]),
                                  int(parts[2]), int(parts[3]))
                except ValueError:
                    continue
                x, y = int(x * sx), int(y * sy)
                w, h = int(w * sx), int(h * sy)
                if w < min_box or h < min_box:
                    continue
                if w > max_box or h > max_box:
                    continue
                if x + w >= W or y + h >= H or x < 1 or y < 1:
                    continue
                bbox_pool.append((x, y, w, h))

    print(f"  Pool: {len(bbox_pool)} real bboxes from {max_frames} frames")
    if bbox_pool:
        widths = [b[2] for b in bbox_pool]
        heights = [b[3] for b in bbox_pool]
        print(f"  Real distribution: w={min(widths)}-{max(widths)}, "
              f"h={min(heights)}-{max(heights)}")
        print(f"  Median: {int(np.median(widths))}x{int(np.median(heights))}, "
              f"P95: {int(np.percentile(widths,95))}x{int(np.percentile(heights,95))}")
    return canonical_img, bbox_pool


def sample_real_detections(bbox_pool, n_objects, seed=42):
    """Sample n_objects from real pool (with replacement if needed)."""
    rng = np.random.RandomState(seed)
    if n_objects <= len(bbox_pool):
        idx = rng.choice(len(bbox_pool), n_objects, replace=False)
    else:
        idx = rng.choice(len(bbox_pool), n_objects, replace=True)
    return [bbox_pool[i] for i in idx]


if __name__ == "__main__":
    frame, pool = load_visdrone_pool(max_frames=30)
    print(f"\nFrame shape: {frame.shape}")
    print(f"Pool size: {len(pool)}")
    sampled = sample_real_detections(pool, 100)
    print(f"Sample 3: {sampled[:3]}")