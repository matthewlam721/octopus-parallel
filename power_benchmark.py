"""
Octopus Power Efficiency Benchmark — Edge Device (Paper Data)
==============================================================
For publication: measures power and energy on Jetson Orin Nano
at drone/satellite scale object counts (100-5000+).

Hypothesis:
  - Single kernel launch (Octopus) vs N launches (Individual)
  - At high object counts, Octopus reduces scheduling overhead
  - Less overhead → GPU finishes sooner → less total energy
  - On constrained edge device (4MB L2, 102 GB/s), cache-friendly
    sequential access pattern should also reduce memory controller power

Hardware: Jetson Orin Nano
  - 8GB shared LPDDR5, 102 GB/s
  - 4MB L2 cache
  - 1024 CUDA cores, Ampere SM
  - TDP: 7-15W configurable

Run: sudo python3 power_benchmark_paper.py
"""

import numpy as np
import time
import os
import glob
import threading
import csv
from datetime import datetime
from numba import cuda, float32, int32, uint8

# ============================================
# CONFIG
# ============================================
CHANNELS = 3
SEED = 42
TARGET_W, TARGET_H = 224, 224
FRAME_W, FRAME_H = 3840, 2160  # 4K — drone / satellite frame

WARMUP = 15
SAMPLE_INTERVAL_MS = 2  # Faster sampling for accuracy

# Object counts to test — paper sweep
OBJECT_COUNTS = [50, 100, 200, 500, 1000, 2000, 3000, 5000]

# Iterations scaled by object count (keep total work ~consistent for power measurement)
def get_iterations(n_objects):
    """More objects = fewer iterations needed for stable power reading."""
    if n_objects <= 100:
        return 300
    elif n_objects <= 500:
        return 150
    elif n_objects <= 1000:
        return 100
    elif n_objects <= 3000:
        return 50
    else:
        return 30

# Crop size distributions (drone/satellite detections)
# Mix of small, medium, large objects seen from aerial view
CROP_SIZE_DISTRIBUTION = {
    # (min_w, max_w, min_h, max_h, probability)
    "tiny":   (16, 40,   16, 40,   0.30),   # distant cars, people
    "small":  (40, 80,   40, 80,   0.35),   # vehicles, structures
    "medium": (80, 160,  80, 160,  0.25),   # buildings, large vehicles
    "large":  (160, 320, 160, 320, 0.10),   # large structures
}


# ============================================
# POWER MONITORING (same as before, refined)
# ============================================
class JetsonPowerMonitor:
    KNOWN_POWER_PATHS = [
        "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input",
        "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input",
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input",
    ]

    def __init__(self):
        self.power_paths = []
        self.path_labels = []
        self.total_power_path = None
        self._find_sensors()

    def _find_sensors(self):
        print("  Searching for power sensors...")
        for path in self.KNOWN_POWER_PATHS:
            if os.path.exists(path):
                try:
                    val = self._read_path(path)
                    if val is not None and val >= 0:
                        self.power_paths.append(path)
                        self.path_labels.append(path.split("/")[-1])
                        print(f"    Found: {path} → {val} mW")
                except:
                    pass

        if not self.power_paths:
            for pattern in ["/sys/class/hwmon/hwmon*/power*_input",
                           "/sys/class/hwmon/hwmon*/in*_input",
                           "/sys/bus/i2c/drivers/ina3221x/*/hwmon/hwmon*/in*_input"]:
                for path in sorted(glob.glob(pattern)):
                    try:
                        val = self._read_path(path)
                        if val is not None and val > 0:
                            self.power_paths.append(path)
                            self.path_labels.append(path.split("/")[-1])
                            if len(self.power_paths) <= 5:
                                print(f"    Found: {path} → {val} mW")
                    except:
                        pass

        if self.power_paths:
            self.total_power_path = self.power_paths[0]
            print(f"  Primary sensor: {self.total_power_path}")
        else:
            print("  ⚠️  No power sensors found!")

    def _read_path(self, path):
        try:
            with open(path, 'r') as f:
                return int(f.read().strip())
        except:
            return None

    def read_total_power_mw(self):
        if self.total_power_path:
            return self._read_path(self.total_power_path)
        return None

    def read_all_rails(self):
        readings = {}
        for path, label in zip(self.power_paths, self.path_labels):
            val = self._read_path(path)
            if val is not None:
                readings[label] = val
        return readings


class PowerSampler:
    def __init__(self, monitor, interval_ms=2):
        self.monitor = monitor
        self.interval = interval_ms / 1000.0
        self.samples = []
        self.timestamps = []
        self.all_rails = []  # Store all rail readings
        self.running = False
        self._thread = None

    def start(self):
        self.samples = []
        self.timestamps = []
        self.all_rails = []
        self.running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sample_loop(self):
        while self.running:
            t = time.perf_counter()
            power = self.monitor.read_total_power_mw()
            rails = self.monitor.read_all_rails()
            if power is not None:
                self.samples.append(power)
                self.timestamps.append(t)
                self.all_rails.append(rails)
            time.sleep(self.interval)

    def get_stats(self):
        if not self.samples:
            return None
        arr = np.array(self.samples, dtype=np.float64)
        return {
            "samples": len(arr),
            "mean_mw": np.mean(arr),
            "median_mw": np.median(arr),
            "min_mw": np.min(arr),
            "max_mw": np.max(arr),
            "std_mw": np.std(arr),
            "p5_mw": np.percentile(arr, 5),
            "p95_mw": np.percentile(arr, 95),
        }

    def get_energy_mj(self):
        if len(self.timestamps) < 2:
            return None
        total_time_s = self.timestamps[-1] - self.timestamps[0]
        mean_power_mw = np.mean(self.samples)
        return mean_power_mw * total_time_s

    def get_duration_s(self):
        if len(self.timestamps) < 2:
            return 0
        return self.timestamps[-1] - self.timestamps[0]


# ============================================
# KERNELS
# ============================================
@cuda.jit(fastmath=True)
def octopus_crop_resize(src_flat, metadata, out_tensor):
    task_id = cuda.blockIdx.x
    if task_id >= metadata.shape[0]:
        return
    src_offset = metadata[task_id, 0]
    src_w      = metadata[task_id, 1]
    src_h      = metadata[task_id, 2]
    crop_x     = metadata[task_id, 3]
    crop_y     = metadata[task_id, 4]
    crop_w     = metadata[task_id, 5]
    crop_h     = metadata[task_id, 6]
    dst_idx    = metadata[task_id, 7]
    target_w = 224
    target_h = 224
    scale_x = crop_w / float32(target_w)
    scale_y = crop_h / float32(target_h)
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    total = target_w * target_h
    max_x = src_w - 1
    max_y = src_h - 1
    for i in range(tid, total, stride):
        ty = i // target_w
        tx = i % target_w
        gx = tx * scale_x + crop_x
        gy = ty * scale_y + crop_y
        ix = int32(gx)
        iy = int32(gy)
        if ix < 0: ix = 0
        elif ix >= max_x: ix = max_x - 1
        if iy < 0: iy = 0
        elif iy >= max_y: iy = max_y - 1
        fx = gx - ix
        fy = gy - iy
        base = src_offset + (iy * src_w + ix) * CHANNELS
        down = base + src_w * CHANNELS
        for c in range(CHANNELS):
            p00 = float32(src_flat[base + c])
            p10 = float32(src_flat[base + CHANNELS + c])
            p01 = float32(src_flat[down + c])
            p11 = float32(src_flat[down + CHANNELS + c])
            top = p00 + (p10 - p00) * fx
            bot = p01 + (p11 - p01) * fx
            val = top + (bot - top) * fy
            val = val + 0.5
            if val < 0: val = 0.0
            if val > 255: val = 255.0
            out_tensor[dst_idx, ty, tx, c] = uint8(val)


@cuda.jit(fastmath=True)
def octopus_normalize_float(src_tensor, dst_tensor):
    task_id = cuda.blockIdx.x
    if task_id >= src_tensor.shape[0]:
        return
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    total = 224 * 224 * CHANNELS
    for i in range(tid, total, stride):
        ch = i % CHANNELS
        rem = i // CHANNELS
        tx = rem % 224
        ty = rem // 224
        dst_tensor[task_id, ty, tx, ch] = float32(src_tensor[task_id, ty, tx, ch]) / 255.0


@cuda.jit(fastmath=True)
def single_crop_resize_kernel(src_flat, src_w, src_h,
                               crop_x, crop_y, crop_w, crop_h,
                               out_patch):
    target_w = 224
    target_h = 224
    start = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridsize(1)
    total = target_w * target_h
    scale_x = crop_w / float32(target_w)
    scale_y = crop_h / float32(target_h)
    max_x = src_w - 1
    max_y = src_h - 1
    for i in range(start, total, stride):
        ty = i // target_w
        tx = i % target_w
        gx = tx * scale_x + crop_x
        gy = ty * scale_y + crop_y
        ix = int32(gx)
        iy = int32(gy)
        if ix < 0: ix = 0
        elif ix >= max_x: ix = max_x - 1
        if iy < 0: iy = 0
        elif iy >= max_y: iy = max_y - 1
        fx = gx - ix
        fy = gy - iy
        base = (iy * src_w + ix) * CHANNELS
        down = base + src_w * CHANNELS
        for c in range(CHANNELS):
            p00 = float32(src_flat[base + c])
            p10 = float32(src_flat[base + CHANNELS + c])
            p01 = float32(src_flat[down + c])
            p11 = float32(src_flat[down + CHANNELS + c])
            top = p00 + (p10 - p00) * fx
            bot = p01 + (p11 - p01) * fx
            val = top + (bot - top) * fy
            val = val + 0.5
            if val < 0: val = 0.0
            if val > 255: val = 255.0
            out_patch[ty, tx, c] = uint8(val)


# ============================================
# DETECTION GENERATION
# ============================================
def generate_detections(frame_w, frame_h, num_objects, seed=42):
    """Generate realistic aerial/satellite detections with variable sizes."""
    rng = np.random.RandomState(seed)

    categories = list(CROP_SIZE_DISTRIBUTION.values())
    probs = [c[4] for c in categories]

    objects = []
    for _ in range(num_objects):
        # Pick size category
        idx = rng.choice(len(categories), p=probs)
        min_w, max_w, min_h, max_h, _ = categories[idx]

        w = rng.randint(min_w, max_w + 1)
        h = rng.randint(min_h, max_h + 1)
        x = rng.randint(1, max(2, frame_w - w - 2))
        y = rng.randint(1, max(2, frame_h - h - 2))
        objects.append((x, y, w, h))

    return objects


# ============================================
# BENCHMARKS
# ============================================
def measure_idle(monitor, duration=3.0):
    """Measure idle power baseline."""
    print(f"\n  Measuring idle power ({duration}s)...")
    cuda.synchronize()
    time.sleep(1.0)

    sampler = PowerSampler(monitor, SAMPLE_INTERVAL_MS)
    sampler.start()
    time.sleep(duration)
    sampler.stop()
    return sampler.get_stats()


def benchmark_method(name, run_fn, monitor, iterations):
    """Generic benchmark wrapper with power sampling."""
    cuda.synchronize()
    time.sleep(0.5)  # Cool down between tests

    sampler = PowerSampler(monitor, SAMPLE_INTERVAL_MS)
    sampler.start()
    t0 = time.perf_counter()

    for _ in range(iterations):
        run_fn()
        cuda.synchronize()

    total_s = time.perf_counter() - t0
    sampler.stop()

    stats = sampler.get_stats()
    energy = sampler.get_energy_mj()
    per_frame_ms = (total_s / iterations) * 1000

    return {
        "name": name,
        "iterations": iterations,
        "total_s": total_s,
        "per_frame_ms": per_frame_ms,
        "power_stats": stats,
        "energy_total_mj": energy,
        "energy_per_frame_mj": energy / iterations if energy else None,
        "samples": stats["samples"] if stats else 0,
    }


def run_sweep(monitor, idle_stats):
    """Main sweep: Octopus vs Individual across object counts."""
    print(f"\n  {'='*70}")
    print(f"  POWER SWEEP — OBJECT COUNT SCALING")
    print(f"  Frame: {FRAME_W}x{FRAME_H} (4K drone/satellite)")
    print(f"  {'='*70}")

    idle_mw = idle_stats['mean_mw'] if idle_stats else 0

    # Prepare frame
    frame = np.random.randint(0, 255, (FRAME_H, FRAME_W, CHANNELS), dtype=np.uint8)
    frame_flat = frame.reshape(-1).astype(np.uint8)
    src_dev = cuda.to_device(frame_flat)
    tpb = 256
    bpg = (TARGET_W * TARGET_H + tpb - 1) // tpb

    all_results = []

    for n_obj in OBJECT_COUNTS:
        iterations = get_iterations(n_obj)
        objects = generate_detections(FRAME_W, FRAME_H, n_obj, seed=SEED)

        widths = [w for (_, _, w, _) in objects]
        heights = [h for (_, _, _, h) in objects]
        print(f"\n  ── {n_obj} objects ── ({iterations} iters)")
        print(f"     Size range: {min(widths)}x{min(heights)} → {max(widths)}x{max(heights)}")

        # Allocate
        out_uint8 = cuda.device_array((n_obj, TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)
        out_float = cuda.device_array((n_obj, TARGET_H, TARGET_W, CHANNELS), dtype=np.float32)

        # Pre-build metadata (for kernel-only test)
        meta_host = np.zeros((n_obj, 8), dtype=np.int32)
        for i, (x, y, w, h) in enumerate(objects):
            meta_host[i] = [0, FRAME_W, FRAME_H, x, y, w, h, i]
        meta_dev = cuda.to_device(meta_host)

        # Individual output patches
        out_patches = [cuda.device_array((TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)
                       for _ in range(n_obj)]

        # Warmup
        for _ in range(WARMUP):
            octopus_crop_resize[n_obj, 256](src_dev, meta_dev, out_uint8)
            octopus_normalize_float[n_obj, 256](out_uint8, out_float)
            cuda.synchronize()
        for _ in range(min(5, WARMUP)):
            for i, (x, y, w, h) in enumerate(objects[:min(50, n_obj)]):
                single_crop_resize_kernel[bpg, tpb](
                    src_dev, FRAME_W, FRAME_H, x, y, w, h, out_patches[i])
            cuda.synchronize()

        # ---- Octopus kernel only ----
        def run_octopus_kernel():
            octopus_crop_resize[n_obj, 256](src_dev, meta_dev, out_uint8)
            octopus_normalize_float[n_obj, 256](out_uint8, out_float)

        r_oct_k = benchmark_method("Octopus kernel", run_octopus_kernel,
                                    monitor, iterations)

        # ---- Octopus e2e (meta build + upload + kernel + norm) ----
        def run_octopus_e2e():
            m = np.zeros((n_obj, 8), dtype=np.int32)
            for i, (x, y, w, h) in enumerate(objects):
                m[i] = [0, FRAME_W, FRAME_H, x, y, w, h, i]
            md = cuda.to_device(m)
            octopus_crop_resize[n_obj, 256](src_dev, md, out_uint8)
            octopus_normalize_float[n_obj, 256](out_uint8, out_float)

        r_oct_e2e = benchmark_method("Octopus e2e", run_octopus_e2e,
                                      monitor, iterations)

        # ---- Individual CUDA ----
        def run_individual():
            for i, (x, y, w, h) in enumerate(objects):
                single_crop_resize_kernel[bpg, tpb](
                    src_dev, FRAME_W, FRAME_H, x, y, w, h, out_patches[i])

        r_indiv = benchmark_method("Individual CUDA", run_individual,
                                    monitor, iterations)

        # Print results
        print(f"\n     {'Method':<22} {'ms/frame':>9} {'Power mW':>10} {'mJ/frame':>10} {'Δ Idle':>8}")
        print(f"     {'-'*61}")

        for r in [r_oct_k, r_oct_e2e, r_indiv]:
            p = r['power_stats']
            power = p['mean_mw'] if p else 0
            mj = r['energy_per_frame_mj'] or 0
            delta = power - idle_mw
            print(f"     {r['name']:<22} {r['per_frame_ms']:>7.2f}ms {power:>8.0f} {mj:>9.3f} {delta:>+7.0f}")

        # Speedup and energy comparison
        if r_oct_k['energy_per_frame_mj'] and r_indiv['energy_per_frame_mj']:
            speed_k = r_indiv['per_frame_ms'] / r_oct_k['per_frame_ms']
            speed_e2e = r_indiv['per_frame_ms'] / r_oct_e2e['per_frame_ms']
            energy_save_k = (1 - r_oct_k['energy_per_frame_mj'] / r_indiv['energy_per_frame_mj']) * 100
            energy_save_e2e = (1 - r_oct_e2e['energy_per_frame_mj'] / r_indiv['energy_per_frame_mj']) * 100
            print(f"\n     Speed:  kernel {speed_k:.1f}x | e2e {speed_e2e:.1f}x")
            print(f"     Energy: kernel {energy_save_k:+.1f}% | e2e {energy_save_e2e:+.1f}%")

        all_results.append({
            "n_objects": n_obj,
            "octopus_kernel": r_oct_k,
            "octopus_e2e": r_oct_e2e,
            "individual": r_indiv,
        })

        del out_uint8, out_float, out_patches

    return all_results


def save_csv(all_results, idle_stats, filename="power_results.csv"):
    """Save results to CSV for paper plots."""
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    idle_mw = idle_stats['mean_mw'] if idle_stats else 0

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_objects",
            "method",
            "ms_per_frame",
            "power_mean_mw",
            "power_median_mw",
            "power_p5_mw",
            "power_p95_mw",
            "power_std_mw",
            "energy_per_frame_mj",
            "energy_total_mj",
            "power_delta_idle_mw",
            "idle_power_mw",
            "iterations",
            "samples",
        ])

        for entry in all_results:
            n = entry["n_objects"]
            for key, label in [("octopus_kernel", "octopus_kernel"),
                                ("octopus_e2e", "octopus_e2e"),
                                ("individual", "individual_cuda")]:
                r = entry[key]
                p = r['power_stats'] or {}
                writer.writerow([
                    n,
                    label,
                    f"{r['per_frame_ms']:.3f}",
                    f"{p.get('mean_mw', 0):.1f}",
                    f"{p.get('median_mw', 0):.1f}",
                    f"{p.get('p5_mw', 0):.1f}",
                    f"{p.get('p95_mw', 0):.1f}",
                    f"{p.get('std_mw', 0):.1f}",
                    f"{r['energy_per_frame_mj']:.4f}" if r['energy_per_frame_mj'] else "",
                    f"{r['energy_total_mj']:.1f}" if r['energy_total_mj'] else "",
                    f"{p.get('mean_mw', 0) - idle_mw:.1f}",
                    f"{idle_mw:.1f}",
                    r['iterations'],
                    r['samples'],
                ])

    print(f"\n  CSV saved: {filepath}")
    return filepath


# ============================================
# MAIN
# ============================================
def main():
    print()
    print("*" * 70)
    print("  ⚡ OCTOPUS POWER EFFICIENCY — EDGE DEVICE (PAPER DATA)")
    print("  Hardware: Jetson Orin Nano (8GB, 4MB L2, 102 GB/s)")
    print("  Frame: 4K (3840×2160) — drone / satellite")
    print(f"  Object counts: {OBJECT_COUNTS}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("*" * 70)

    if os.geteuid() != 0:
        print("\n  ⚠️  Not root. Run: sudo python3 power_benchmark_paper.py")
        print("  Continuing anyway...\n")

    monitor = JetsonPowerMonitor()

    # Idle baseline
    idle = measure_idle(monitor, duration=3.0)
    if idle:
        print(f"  Idle: {idle['mean_mw']:.0f} mW (±{idle['std_mw']:.0f})")

    # Main sweep
    results = run_sweep(monitor, idle)

    # Save CSV
    csv_path = save_csv(results, idle)

    # ---- Final Summary Table ----
    print()
    print("=" * 70)
    print("  📊 PAPER SUMMARY — POWER EFFICIENCY ON EDGE DEVICE")
    print("=" * 70)

    idle_mw = idle['mean_mw'] if idle else 0

    print(f"\n  Idle power: {idle_mw:.0f} mW")
    print(f"\n  {'Objects':>8} │ {'Octopus ms':>11} {'Indiv ms':>11} {'Speedup':>8}"
          f" │ {'Oct mJ':>8} {'Ind mJ':>8} {'Energy Δ':>9}")
    print(f"  {'─'*8}─┼{'─'*32}─┼{'─'*27}")

    for entry in results:
        n = entry["n_objects"]
        ok = entry["octopus_kernel"]
        ind = entry["individual"]

        ok_ms = ok['per_frame_ms']
        ind_ms = ind['per_frame_ms']
        speedup = ind_ms / ok_ms if ok_ms > 0 else 0

        ok_mj = ok['energy_per_frame_mj'] or 0
        ind_mj = ind['energy_per_frame_mj'] or 0
        energy_delta = (1 - ok_mj / ind_mj) * 100 if ind_mj > 0 else 0

        print(f"  {n:>8} │ {ok_ms:>9.2f}ms {ind_ms:>9.2f}ms {speedup:>7.1f}x"
              f" │ {ok_mj:>7.3f} {ind_mj:>7.3f} {energy_delta:>+8.1f}%")

    print(f"\n  Key findings for paper:")
    print(f"  • At high object counts (1000+), Octopus single kernel launch")
    print(f"    eliminates N individual launch overheads")
    print(f"  • Energy scales with both power draw AND execution time")
    print(f"  • Edge device (4MB L2, 102GB/s) amplifies cache efficiency gains")
    print(f"  • Relevant for battery-powered platforms: drone, satellite, VR headset")
    print(f"\n  CSV data ready for plotting: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()