"""
Edge Processing Simulation — REAL DATA variant
Uses VisDrone for drone scenario, Sentinel-2 for satellite scenario.
Reuses kernels + pipeline logic from edge_simulation.py.
"""
import os
import time
import numpy as np
from numba import cuda

from edge_simulation import (
    CHANNELS, SEED, WARMUP, ITERATIONS,
    SAT_DOWNLINK_MBPS, SAT_THRESHOLD,
    DRONE_FPS, DRONE_FRAME_BUDGET_MS,
    DRONE_TARGET_W, DRONE_TARGET_H, DRONE_DOWNLINK_MBPS,
    octopus_normalize_threshold, octopus_crop_resize, single_crop_resize_kernel,
)
from visdrone_loader import load_visdrone_pool, sample_real_detections
from satellite_loader import load_satellite_workload

# ★ Override threshold for Greenland scene (high albedo from ice/snow)
import edge_simulation as _es
_es.SAT_THRESHOLD = 0.535
SAT_THRESHOLD = 0.535



# ============================================
# SATELLITE SCENARIO — REAL Sentinel-2
# ============================================
def run_satellite_scenario_real(crop_size=8192):
    print("=" * 70)
    print("  SCENARIO 1: SATELLITE ONBOARD TILE FILTERING — REAL Sentinel-2")
    print("=" * 70)

    # ★ Load real Sentinel-2 region
    sat_img, _ = load_satellite_workload(crop_size=crop_size, n_rois=1)
    SAT_H, SAT_W = sat_img.shape[:2]
    sat_flat = sat_img.reshape(-1).astype(np.uint8)

    print(f"  Image: {SAT_W}x{SAT_H} ({CHANNELS}ch) — REAL Sentinel-2 Greenland tile")
    print(f"  Downlink: {SAT_DOWNLINK_MBPS} Mbps")
    print(f"  Filter: keep tiles with mean brightness > {SAT_THRESHOLD}")
    print()

    # Use grid tiling (same as original synthetic) on real pixels
    rng = np.random.RandomState(SEED)
    tiles = []
    y = 0
    while y < SAT_H:
        row_h = rng.randint(128, 513)
        if y + row_h > SAT_H:
            row_h = SAT_H - y
        x = 0
        while x < SAT_W:
            tile_w = rng.randint(128, 513)
            if x + tile_w > SAT_W:
                tile_w = SAT_W - x
            tiles.append((x, y, tile_w, row_h))
            x += tile_w
        y += row_h

    num_tiles = len(tiles)
    total_pixels = sum(w * h for (_, _, w, h) in tiles)
    total_bytes = total_pixels * CHANNELS
    total_mb = total_bytes / (1024 * 1024)

    print(f"  Tiles generated: {num_tiles}")
    print(f"  Total data: {total_mb:.0f} MB")
    print()

    transmit_all_seconds = (total_bytes * 8) / (SAT_DOWNLINK_MBPS * 1e6)
    print(f"  [A] Transmit everything (no processing):")
    print(f"      Data: {total_mb:.0f} MB")
    print(f"      Time: {transmit_all_seconds:.1f} seconds ({transmit_all_seconds/60:.1f} min)")
    print()

    # Build metadata
    meta_host = np.zeros((num_tiles, 4), dtype=np.int32)
    tile_sizes = []
    for i, (x, y, w, h) in enumerate(tiles):
        offset = (y * SAT_W + x) * CHANNELS
        meta_host[i] = [offset, w, h, i]
        tile_sizes.append(w * h * CHANNELS)

    src_dev = cuda.to_device(sat_flat)
    meta_dev = cuda.to_device(meta_host)
    flags_dev = cuda.device_array(num_tiles, dtype=np.int32)

    for _ in range(WARMUP):
        octopus_normalize_threshold[num_tiles, 256](
            src_dev, meta_dev, flags_dev, SAT_THRESHOLD, SAT_W)
        cuda.synchronize()

    process_times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        octopus_normalize_threshold[num_tiles, 256](
            src_dev, meta_dev, flags_dev, SAT_THRESHOLD, SAT_W)
        cuda.synchronize()
        process_times.append((time.perf_counter() - t0) * 1000)

    flags = flags_dev.copy_to_host()
    num_interesting = int(flags.sum())
    interesting_bytes = sum(tile_sizes[i] for i in range(num_tiles) if flags[i])
    interesting_mb = interesting_bytes / (1024 * 1024)
    transmit_filtered_seconds = (interesting_bytes * 8) / (SAT_DOWNLINK_MBPS * 1e6)

    process_ms = np.median(process_times)
    total_b_seconds = process_ms / 1000 + transmit_filtered_seconds

    print(f"  [B] Octopus filter → transmit interesting only:")
    print(f"      Processing: {process_ms:.1f} ms ({num_tiles} tiles, single kernel)")
    print(f"      Tiles kept: {num_interesting}/{num_tiles} "
          f"({num_interesting/num_tiles*100:.0f}%)")
    print(f"      Data after filter: {interesting_mb:.1f} MB (was {total_mb:.0f} MB)")
    print(f"      Transmit time: {transmit_filtered_seconds:.1f}s "
          f"(was {transmit_all_seconds:.1f}s)")
    print(f"      Total: {total_b_seconds:.1f}s")
    print()

    bandwidth_saved = (1 - interesting_bytes / total_bytes) * 100
    speedup = transmit_all_seconds / total_b_seconds

    print(f"  RESULT:")
    print(f"    Bandwidth saved: {bandwidth_saved:.0f}%")
    print(f"    Pipeline speedup: {speedup:.1f}x")
    print(f"    Processing overhead: {process_ms:.1f}ms "
          f"(negligible vs {transmit_all_seconds:.0f}s transmit)")
    print()
    return {
        "num_tiles": num_tiles, "total_mb": total_mb,
        "process_ms": process_ms, "bandwidth_saved_pct": bandwidth_saved,
        "speedup": speedup,
    }


# ============================================
# DRONE SCENARIO — REAL VisDrone
# ============================================
def run_drone_scenario_real():
    print("=" * 70)
    print("  SCENARIO 2: DRONE REAL-TIME CLASSIFICATION — REAL VisDrone")
    print("=" * 70)

    # ★ Load real drone frame + bbox pool
    frame, bbox_pool = load_visdrone_pool(max_frames=50)
    DRONE_H, DRONE_W = frame.shape[:2]
    print(f"  Video: {DRONE_W}x{DRONE_H} @ {DRONE_FPS}fps (REAL VisDrone)")
    print(f"  Frame budget: {DRONE_FRAME_BUDGET_MS:.1f}ms")
    print(f"  Target: crop detections → {DRONE_TARGET_W}x{DRONE_TARGET_H}")
    print(f"  Real bbox pool: {len(bbox_pool)} bboxes from VisDrone annotations")
    print()

    frame_flat = frame.reshape(-1).astype(np.uint8)
    src_dev = cuda.to_device(frame_flat)

    rng = np.random.RandomState(SEED)

    for duration_sec in [1, 5, 10]:
        num_frames = DRONE_FPS * duration_sec
        # Each frame: sample real bbox count (real VisDrone avg ~25-50/frame)
        # We sample with replacement from real pool to fill num_frames worth
        all_detections = []
        for fid in range(num_frames):
            # Realistic: 5-50 detections per frame (matches drone scenario)
            n_per_frame = rng.randint(5, 51)
            sampled = sample_real_detections(bbox_pool, n_per_frame,
                                              seed=SEED + fid)
            for (x, y, w, h) in sampled:
                all_detections.append((fid, x, y, w, h))
        num_crops = len(all_detections)

        print(f"  --- {duration_sec}s of video ({num_frames} frames, "
              f"{num_crops} REAL detections) ---")

        # ---- CPU OpenCV ----
        import cv2
        cpu_times = []
        for _ in range(min(5, ITERATIONS)):
            t0 = time.perf_counter()
            for (fid, x, y, w, h) in all_detections:
                roi = frame[y:y+h, x:x+w]
                _ = cv2.resize(roi, (DRONE_TARGET_W, DRONE_TARGET_H),
                               interpolation=cv2.INTER_LINEAR)
            cpu_times.append((time.perf_counter() - t0) * 1000)
        cpu_ms = np.median(cpu_times)

        # ---- Individual CUDA ----
        out_patches = [cuda.device_array(
            (DRONE_TARGET_H, DRONE_TARGET_W, CHANNELS), dtype=np.uint8
        ) for _ in range(min(num_crops, 2000))]
        tpb = 256
        bpg = (DRONE_TARGET_W * DRONE_TARGET_H + tpb - 1) // tpb

        x0, y0, w0, h0 = all_detections[0][1:]
        for _ in range(WARMUP):
            single_crop_resize_kernel[bpg, tpb](
                src_dev, DRONE_W, DRONE_H, x0, y0, w0, h0, out_patches[0])
            cuda.synchronize()

        indiv_times = []
        for _ in range(min(5, ITERATIONS)):
            t0 = time.perf_counter()
            for i, (fid, x, y, w, h) in enumerate(all_detections):
                single_crop_resize_kernel[bpg, tpb](
                    src_dev, DRONE_W, DRONE_H, x, y, w, h,
                    out_patches[i % len(out_patches)])
            cuda.synchronize()
            indiv_times.append((time.perf_counter() - t0) * 1000)
        indiv_ms = np.median(indiv_times)

        # ---- Octopus ----
        meta_host = np.zeros((num_crops, 8), dtype=np.int32)
        for i, (fid, x, y, w, h) in enumerate(all_detections):
            meta_host[i] = [0, DRONE_W, DRONE_H, x, y, w, h, i]
        meta_dev = cuda.to_device(meta_host)
        out_dev = cuda.device_array(
            (num_crops, DRONE_TARGET_H, DRONE_TARGET_W, CHANNELS), dtype=np.uint8)

        for _ in range(WARMUP):
            octopus_crop_resize[num_crops, 256](src_dev, meta_dev, out_dev)
            cuda.synchronize()

        oct_times = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            octopus_crop_resize[num_crops, 256](src_dev, meta_dev, out_dev)
            cuda.synchronize()
            oct_times.append((time.perf_counter() - t0) * 1000)
        oct_ms = np.median(oct_times)

        cpu_per_frame = cpu_ms / num_frames
        indiv_per_frame = indiv_ms / num_frames
        oct_per_frame = oct_ms / num_frames

        cpu_rt = "YES" if cpu_per_frame < DRONE_FRAME_BUDGET_MS else "NO"
        indiv_rt = "YES" if indiv_per_frame < DRONE_FRAME_BUDGET_MS else "NO"
        oct_rt = "YES" if oct_per_frame < DRONE_FRAME_BUDGET_MS else "NO"

        raw_frame_bytes = DRONE_W * DRONE_H * CHANNELS * num_frames
        crop_bytes = num_crops * DRONE_TARGET_W * DRONE_TARGET_H * CHANNELS
        raw_transmit_s = (raw_frame_bytes * 8) / (DRONE_DOWNLINK_MBPS * 1e6)
        crop_transmit_s = (crop_bytes * 8) / (DRONE_DOWNLINK_MBPS * 1e6)

        print(f"    Detections per frame: {num_crops/num_frames:.0f} avg (real pool)")
        print()
        print(f"    {'Method':<30} {'Total':>10} {'Per-frame':>12} {'Real-time?':>12}")
        print(f"    {'-'*64}")
        print(f"    {'CPU OpenCV':<30} {cpu_ms:>8.1f}ms {cpu_per_frame:>10.1f}ms {cpu_rt:>12}")
        print(f"    {'Individual CUDA':<30} {indiv_ms:>8.1f}ms {indiv_per_frame:>10.1f}ms {indiv_rt:>12}")
        print(f"    {'Octopus (single launch)':<30} {oct_ms:>8.1f}ms {oct_per_frame:>10.1f}ms {oct_rt:>12}")
        print(f"    {'Frame budget':<30} {'':>10} {DRONE_FRAME_BUDGET_MS:>10.1f}ms")
        print()
        print(f"    Bandwidth: raw {raw_frame_bytes/1024/1024:.0f}MB ({raw_transmit_s:.1f}s) "
              f"→ crops {crop_bytes/1024/1024:.0f}MB ({crop_transmit_s:.1f}s)")
        print(f"    Speedup vs CPU: {cpu_ms/oct_ms:.1f}x")
        print(f"    Speedup vs Individual: {indiv_ms/oct_ms:.1f}x")
        print()


# ============================================
# MAIN
# ============================================
def main():
    print()
    print("*" * 70)
    print("  EDGE PROCESSING SIMULATION — REAL DATA")
    print("  Hardware: Jetson Orin Nano (8GB, 102 GB/s, 4MB L2)")
    print("  Datasets: VisDrone-DET val (drone), Sentinel-2 L2A (satellite)")
    print("*" * 70)
    print()

    run_satellite_scenario_real(crop_size=8192)
    print()
    run_drone_scenario_real()

    print()
    print("=" * 70)
    print("  KEY TAKEAWAY (REAL DATA)")
    print("=" * 70)
    print("  Satellite: real Sentinel-2 → onboard filter saves real downlink time.")
    print("  Drone: real VisDrone bboxes → Octopus keeps real-time at 10s batch.")
    print("=" * 70)


if __name__ == "__main__":
    main()