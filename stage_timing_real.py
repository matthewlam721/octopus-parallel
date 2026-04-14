"""
Per-stage timing breakdown — Real VisDrone data.
Proves where Octopus's speedup comes from (and where its overhead is).

For each n_objects, measures:
  - Stage 1: Build metadata array (host)
  - Stage 2: H2D metadata transfer
  - Stage 3: Kernel execution (crop+resize)
  - Stage 4: Kernel execution (normalize)
  - Stage 5: Final sync
vs Individual:
  - Stage 1-2: skipped (no metadata)
  - Stage 3: 1000 individual kernel launches
  - Stage 4: skipped (per-task fused)
  - Stage 5: Final sync (cumulative implicit syncs)
"""
import os
import time
import numpy as np
from datetime import datetime
from numba import cuda

from power_benchmark import (
    CHANNELS, SEED, TARGET_W, TARGET_H, WARMUP,
    octopus_crop_resize, octopus_normalize_float, single_crop_resize_kernel,
)
from visdrone_loader import load_visdrone_pool, sample_real_detections

# Object counts to profile
PROFILE_COUNTS = [100, 500, 1000, 2000]
N_REPS = 30  # repetitions per stage for averaging


def time_stage(fn, n_reps=N_REPS, sync_before=True, sync_after=True):
    """Run fn n_reps times, return (mean_ms, std_ms, all_samples_ms)."""
    if sync_before:
        cuda.synchronize()
    samples = []
    for _ in range(n_reps):
        if sync_before:
            cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_after:
            cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return np.mean(samples), np.std(samples), samples


def profile_octopus(objects, src_dev, n_obj):
    """Time each stage of Octopus pipeline."""
    out_uint8 = cuda.device_array((n_obj, TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)
    out_float = cuda.device_array((n_obj, TARGET_H, TARGET_W, CHANNELS), dtype=np.float32)
    
    
    stages = {}
    
    # ---- Stage 1: Build metadata (host-side numpy) ----
    def build_meta():
        m = np.zeros((n_obj, 8), dtype=np.int32)
        for i, (x, y, w, h) in enumerate(objects):
            m[i] = [0, FRAME_W_REAL, FRAME_H_REAL, x, y, w, h, i]
        return m
    
    # Warmup
    for _ in range(5):
        build_meta()
    
    stages['1_build_meta_host'] = time_stage(
        lambda: build_meta(), n_reps=N_REPS, sync_before=False, sync_after=False
    )
    
    # ---- Stage 2: H2D metadata transfer ----
    meta_host = build_meta()
    
    def h2d_meta():
        cuda.to_device(meta_host)
    
    for _ in range(5):
        h2d_meta()
    
    stages['2_h2d_metadata'] = time_stage(h2d_meta, n_reps=N_REPS)
    
    meta_dev = cuda.to_device(meta_host)
    
    # ---- Stage 3: Crop+resize kernel ----
    def kernel_crop():
        octopus_crop_resize[n_obj, 256](src_dev, meta_dev, out_uint8)
    
    for _ in range(WARMUP):
        kernel_crop()
        cuda.synchronize()
    
    stages['3_kernel_crop_resize'] = time_stage(kernel_crop, n_reps=N_REPS)
    
    # ---- Stage 4: Normalize kernel ----
    def kernel_norm():
        octopus_normalize_float[n_obj, 256](out_uint8, out_float)
    
    for _ in range(WARMUP):
        kernel_norm()
        cuda.synchronize()
    
    stages['4_kernel_normalize'] = time_stage(kernel_norm, n_reps=N_REPS)
    
    # ---- Stage 5: Full pipeline (no per-stage sync) ----
    def full_pipeline():
        m = build_meta()
        md = cuda.to_device(m)
        octopus_crop_resize[n_obj, 256](src_dev, md, out_uint8)
        octopus_normalize_float[n_obj, 256](out_uint8, out_float)
    
    for _ in range(WARMUP):
        full_pipeline()
        cuda.synchronize()
    
    stages['5_e2e_total'] = time_stage(full_pipeline, n_reps=N_REPS)
    
    return stages


def profile_individual(objects, src_dev, n_obj):
    """Time each stage of Individual-kernel pipeline."""
    tpb = 256
    bpg = (TARGET_W * TARGET_H + tpb - 1) // tpb
    out_patches = [cuda.device_array((TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)
                   for _ in range(n_obj)]
    
    stages = {}
    
    # Warmup
    for _ in range(WARMUP):
        for i, (x, y, w, h) in enumerate(objects[:min(50, n_obj)]):
            single_crop_resize_kernel[bpg, tpb](
                src_dev, FRAME_W_REAL, FRAME_H_REAL, x, y, w, h, out_patches[i])
        cuda.synchronize()
    
    # ---- Stage 3: All N kernel launches (no per-launch sync) ----
    def all_launches():
        for i, (x, y, w, h) in enumerate(objects):
            single_crop_resize_kernel[bpg, tpb](
                src_dev, FRAME_W_REAL, FRAME_H_REAL, x, y, w, h, out_patches[i])
    
    stages['3_kernel_launches_total'] = time_stage(all_launches, n_reps=N_REPS)
    
    # ---- Decompose: launch overhead vs work ----
    # Time a single small launch to estimate per-launch overhead
    x, y, w, h = objects[0]
    def single_launch():
        single_crop_resize_kernel[bpg, tpb](
            src_dev, FRAME_W_REAL, FRAME_H_REAL, x, y, w, h, out_patches[0])
    
    for _ in range(WARMUP):
        single_launch()
        cuda.synchronize()
    
    stages['3a_per_launch_overhead'] = time_stage(single_launch, n_reps=N_REPS * 3)
    
    # ---- Stage 5: E2E total ----
    stages['5_e2e_total'] = stages['3_kernel_launches_total']
    
    return stages


def print_stage_table(n_obj, oct_stages, ind_stages):
    print(f"\n  {'='*78}")
    print(f"  PER-STAGE BREAKDOWN — {n_obj} REAL objects")
    print(f"  {'='*78}")
    
    print(f"\n  OCTOPUS pipeline:")
    print(f"  {'Stage':<32} {'Mean (ms)':>12} {'Std':>10} {'% of e2e':>10}")
    print(f"  {'-'*64}")
    e2e_oct = oct_stages['5_e2e_total'][0]
    for stage_name, (mean, std, _) in oct_stages.items():
        pct = (mean / e2e_oct) * 100
        marker = " ★" if stage_name.startswith(('1_', '2_')) else "  "
        print(f"  {marker}{stage_name:<30} {mean:>10.3f} ±{std:>7.3f} {pct:>9.1f}%")
    
    print(f"\n  INDIVIDUAL pipeline:")
    print(f"  {'Stage':<32} {'Mean (ms)':>12} {'Std':>10} {'% of e2e':>10}")
    print(f"  {'-'*64}")
    e2e_ind = ind_stages['5_e2e_total'][0]
    for stage_name, (mean, std, _) in ind_stages.items():
        pct = (mean / e2e_ind) * 100
        print(f"    {stage_name:<30} {mean:>10.3f} ±{std:>7.3f} {pct:>9.1f}%")
    
    # ---- The money table: where does the speedup come from? ----
    print(f"\n  💡 SPEEDUP DECOMPOSITION (Individual − Octopus):")
    octopus_overhead = (oct_stages['1_build_meta_host'][0] + 
                        oct_stages['2_h2d_metadata'][0])
    octopus_kernel = (oct_stages['3_kernel_crop_resize'][0] + 
                      oct_stages['4_kernel_normalize'][0])
    individual_kernel = ind_stages['3_kernel_launches_total'][0]
    
    print(f"     Octopus extra overhead   (stages 1+2): +{octopus_overhead:7.3f} ms")
    print(f"     Octopus kernel           (stages 3+4):  {octopus_kernel:7.3f} ms")
    print(f"     Individual N launches    (stage 3):     {individual_kernel:7.3f} ms")
    print(f"     ─────────────────────────────────────────────────────")
    print(f"     Net saving (kernel side):   {individual_kernel - octopus_kernel:+7.3f} ms")
    print(f"     Net saving (after overhead):{individual_kernel - octopus_kernel - octopus_overhead:+7.3f} ms")
    print(f"     E2E speedup: {e2e_ind / e2e_oct:.2f}x")
    print(f"     Octopus overhead is {(octopus_overhead/e2e_oct)*100:.1f}% of e2e — "
          f"{'negligible ✓' if octopus_overhead < 1.0 else 'NON-trivial ⚠'}")


def main():
    global FRAME_W_REAL, FRAME_H_REAL
    
    print("\n" + "*" * 78)
    print("  🔬 PER-STAGE TIMING — REAL VISDRONE DATA")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("*" * 78)
    
    # Load real data
    frame, bbox_pool = load_visdrone_pool(max_frames=50)
    FRAME_H_REAL, FRAME_W_REAL = frame.shape[:2]
    print(f"\n  Real frame: {FRAME_W_REAL}x{FRAME_H_REAL}, pool: {len(bbox_pool)} bboxes\n")
    
    frame_flat = frame.reshape(-1).astype(np.uint8)
    src_dev = cuda.to_device(frame_flat)
    
    all_results = []
    for n_obj in PROFILE_COUNTS:
        objects = sample_real_detections(bbox_pool, n_obj, seed=SEED)
        
        oct_stages = profile_octopus(objects, src_dev, n_obj)
        ind_stages = profile_individual(objects, src_dev, n_obj)
        
        print_stage_table(n_obj, oct_stages, ind_stages)
        
        all_results.append({
            'n_obj': n_obj,
            'octopus': {k: v[0] for k, v in oct_stages.items()},
            'individual': {k: v[0] for k, v in ind_stages.items()},
        })
    
    # ---- Save CSV ----
    import csv
    csv_path = "stage_timing_real.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_obj', 'method', 'stage', 'mean_ms'])
        for r in all_results:
            for stage, ms in r['octopus'].items():
                w.writerow([r['n_obj'], 'octopus', stage, f"{ms:.4f}"])
            for stage, ms in r['individual'].items():
                w.writerow([r['n_obj'], 'individual', stage, f"{ms:.4f}"])
    print(f"\n\n  ✅ CSV: {csv_path}")


if __name__ == "__main__":
    main()