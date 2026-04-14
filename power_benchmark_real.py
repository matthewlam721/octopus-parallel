"""
Real-data variant: VisDrone frames + real bboxes.
Re-uses ALL kernels/sampler/timing logic from power_benchmark.py.
Only swaps synthetic frame + bboxes with real ones.
"""
import os
import numpy as np
from datetime import datetime
from numba import cuda

from power_benchmark import (
    CHANNELS, SEED, TARGET_W, TARGET_H,
    WARMUP, OBJECT_COUNTS, get_iterations,
    JetsonPowerMonitor,
    octopus_crop_resize, octopus_normalize_float, single_crop_resize_kernel,
    measure_idle, benchmark_method, save_csv,
)
from visdrone_loader import load_visdrone_pool, sample_real_detections


def run_sweep_real(monitor, idle_stats):
    print(f"\n  {'='*70}")
    print(f"  POWER SWEEP — REAL VISDRONE DATA")
    print(f"  {'='*70}")
    idle_mw = idle_stats['mean_mw'] if idle_stats else 0

    # ★ REAL DATA
    frame, bbox_pool = load_visdrone_pool(max_frames=50)
    FRAME_H, FRAME_W = frame.shape[:2]
    print(f"\n  Real frame: {FRAME_W}x{FRAME_H}, pool: {len(bbox_pool)} bboxes")

    frame_flat = frame.reshape(-1).astype(np.uint8)
    src_dev = cuda.to_device(frame_flat)
    tpb = 256
    bpg = (TARGET_W * TARGET_H + tpb - 1) // tpb

    all_results = []
    for n_obj in OBJECT_COUNTS:
        iterations = get_iterations(n_obj)
        objects = sample_real_detections(bbox_pool, n_obj, seed=SEED)

        widths = [w for (_, _, w, _) in objects]
        heights = [h for (_, _, _, h) in objects]
        print(f"\n  ── {n_obj} REAL objects ── ({iterations} iters)")
        print(f"     Real range: {min(widths)}x{min(heights)} → {max(widths)}x{max(heights)}")
        print(f"     Median: {int(np.median(widths))}x{int(np.median(heights))}")

        out_uint8 = cuda.device_array((n_obj, TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)
        out_float = cuda.device_array((n_obj, TARGET_H, TARGET_W, CHANNELS), dtype=np.float32)

        meta_host = np.zeros((n_obj, 8), dtype=np.int32)
        for i, (x, y, w, h) in enumerate(objects):
            meta_host[i] = [0, FRAME_W, FRAME_H, x, y, w, h, i]
        meta_dev = cuda.to_device(meta_host)

        out_patches = [cuda.device_array((TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)
                       for _ in range(n_obj)]

        for _ in range(WARMUP):
            octopus_crop_resize[n_obj, 256](src_dev, meta_dev, out_uint8)
            octopus_normalize_float[n_obj, 256](out_uint8, out_float)
            cuda.synchronize()
        for _ in range(min(5, WARMUP)):
            for i, (x, y, w, h) in enumerate(objects[:min(50, n_obj)]):
                single_crop_resize_kernel[bpg, tpb](
                    src_dev, FRAME_W, FRAME_H, x, y, w, h, out_patches[i])
            cuda.synchronize()

        def run_octopus_kernel():
            octopus_crop_resize[n_obj, 256](src_dev, meta_dev, out_uint8)
            octopus_normalize_float[n_obj, 256](out_uint8, out_float)

        r_oct_k = benchmark_method("Octopus kernel", run_octopus_kernel, monitor, iterations)

        def run_octopus_e2e():
            m = np.zeros((n_obj, 8), dtype=np.int32)
            for i, (x, y, w, h) in enumerate(objects):
                m[i] = [0, FRAME_W, FRAME_H, x, y, w, h, i]
            md = cuda.to_device(m)
            octopus_crop_resize[n_obj, 256](src_dev, md, out_uint8)
            octopus_normalize_float[n_obj, 256](out_uint8, out_float)

        r_oct_e2e = benchmark_method("Octopus e2e", run_octopus_e2e, monitor, iterations)

        def run_individual():
            for i, (x, y, w, h) in enumerate(objects):
                single_crop_resize_kernel[bpg, tpb](
                    src_dev, FRAME_W, FRAME_H, x, y, w, h, out_patches[i])

        r_indiv = benchmark_method("Individual CUDA", run_individual, monitor, iterations)

        print(f"\n     {'Method':<22} {'ms/frame':>9} {'Power mW':>10} {'mJ/frame':>10}")
        print(f"     {'-'*55}")
        for r in [r_oct_k, r_oct_e2e, r_indiv]:
            p = r['power_stats']
            power = p['mean_mw'] if p else 0
            mj = r['energy_per_frame_mj'] or 0
            print(f"     {r['name']:<22} {r['per_frame_ms']:>7.2f}ms {power:>8.0f} {mj:>9.3f}")

        if r_oct_k['energy_per_frame_mj'] and r_indiv['energy_per_frame_mj']:
            sp_k = r_indiv['per_frame_ms'] / r_oct_k['per_frame_ms']
            sp_e = r_indiv['per_frame_ms'] / r_oct_e2e['per_frame_ms']
            es_k = (1 - r_oct_k['energy_per_frame_mj'] / r_indiv['energy_per_frame_mj']) * 100
            es_e = (1 - r_oct_e2e['energy_per_frame_mj'] / r_indiv['energy_per_frame_mj']) * 100
            print(f"\n     Speed: kernel {sp_k:.2f}x | e2e {sp_e:.2f}x")
            print(f"     Energy: kernel {es_k:+.1f}% | e2e {es_e:+.1f}%")

        all_results.append({
            "n_objects": n_obj,
            "octopus_kernel": r_oct_k,
            "octopus_e2e": r_oct_e2e,
            "individual": r_indiv,
        })
        del out_uint8, out_float, out_patches
    return all_results


def main():
    print()
    print("*" * 70)
    print("  ⚡ OCTOPUS POWER — REAL VISDRONE DATA")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("*" * 70)

    if os.geteuid() != 0:
        print("\n  ⚠️  Not root, power readings may fail. Run with sudo.")

    monitor = JetsonPowerMonitor()
    idle = measure_idle(monitor, duration=3.0)
    if idle:
        print(f"  Idle: {idle['mean_mw']:.0f} mW (±{idle['std_mw']:.0f})")

    results = run_sweep_real(monitor, idle)
    save_csv(results, idle, filename="power_results_real.csv")
    print("\n  ✅ Done. CSV: power_results_real.csv")


if __name__ == "__main__":
    main()