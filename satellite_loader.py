"""Load Sentinel-2 real satellite tile + generate variable ROI tiles."""
import os
import numpy as np
import rasterio

SAT_PATH = os.path.expanduser(
    "~/datasets/sentinel2/sentinel2_sample_b02.tif"
)


def load_satellite_canonical(crop_size=4096, offset=(0, 0)):
    """
    Load a canonical region from the full Sentinel-2 tile.
    Returns 2D uint16 array (single band, like grayscale satellite).
    
    Default 4096x4096 = a realistic onboard processing chunk
    (smaller than full 10980² to fit Jetson GPU memory comfortably).
    """
    with rasterio.open(SAT_PATH) as src:
        H_full, W_full = src.shape
        oy, ox = offset
        ch = min(crop_size, H_full - oy)
        cw = min(crop_size, W_full - ox)
        arr = src.read(1, window=((oy, oy + ch), (ox, ox + cw)))
    print(f"  Sentinel-2 region: {arr.shape}, "
          f"range=[{arr.min()}, {arr.max()}], "
          f"mean={arr.mean():.0f}")
    return arr


def generate_satellite_rois(frame_h, frame_w, n_rois,
                             min_size=64, max_size=512,
                             distribution="long_tail", seed=42):
    """
    Generate variable-size ROI bboxes mimicking real satellite ROI extraction
    (e.g., cloud-mask-driven scene analysis, ship detection, agricultural plots).
    
    distribution:
      - "long_tail": most small ROIs (clouds, small objects), few large
      - "uniform": uniform random sizes
      - "bimodal": cluster around two scales (small features + large ag fields)
    """
    rng = np.random.RandomState(seed)
    rois = []
    
    if distribution == "long_tail":
        # Sample sizes from log-normal-ish: most small, tail of large
        # 70% tiny (64-128), 20% medium (128-256), 10% large (256-512)
        weights = [0.70, 0.20, 0.10]
        bins = [(min_size, 128), (128, 256), (256, max_size)]
    elif distribution == "bimodal":
        # 50% small features (~64-150), 50% large ag plots (~300-512)
        weights = [0.50, 0.50]
        bins = [(min_size, 150), (300, max_size)]
    else:  # uniform
        weights = [1.0]
        bins = [(min_size, max_size)]
    
    bin_idx = rng.choice(len(weights), n_rois, p=weights)
    for bi in bin_idx:
        lo, hi = bins[bi]
        w = rng.randint(lo, hi + 1)
        h = rng.randint(lo, hi + 1)
        x = rng.randint(1, max(2, frame_w - w - 2))
        y = rng.randint(1, max(2, frame_h - h - 2))
        rois.append((x, y, w, h))
    
    return rois


def load_satellite_workload(crop_size=4096, n_rois=1000,
                             distribution="long_tail", seed=42):
    """One-stop: returns (canonical_image_uint8_3ch, list_of_rois)."""
    arr = load_satellite_canonical(crop_size=crop_size)
    
    # Convert uint16 → uint8 3ch (matches existing kernel signatures)
    arr_norm = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    arr_3ch = np.stack([arr_norm] * 3, axis=-1)  # H × W × 3
    
    H, W = arr.shape
    rois = generate_satellite_rois(H, W, n_rois,
                                     distribution=distribution, seed=seed)
    
    sizes_w = [r[2] for r in rois]
    sizes_h = [r[3] for r in rois]
    print(f"  ROIs: {len(rois)} ({distribution})")
    print(f"  Size range: w={min(sizes_w)}-{max(sizes_w)}, "
          f"h={min(sizes_h)}-{max(sizes_h)}")
    print(f"  Median: {int(np.median(sizes_w))}x{int(np.median(sizes_h))}, "
          f"P95: {int(np.percentile(sizes_w, 95))}x{int(np.percentile(sizes_h, 95))}")
    
    return arr_3ch, rois


if __name__ == "__main__":
    print("\n=== Smoke test: Sentinel-2 satellite loader ===\n")
    frame, rois = load_satellite_workload(crop_size=4096, n_rois=500)
    print(f"\nFrame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"ROI count: {len(rois)}")
    print(f"First 3 ROIs: {rois[:3]}")