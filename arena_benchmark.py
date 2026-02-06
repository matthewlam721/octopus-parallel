"""
TensorRT vs Octopus: Crop+Resize Benchmark on Jetson
=====================================================
Compares approaches for variable-size crop+resize:
  1. CPU OpenCV (baseline)
  2. Individual CUDA kernels (one per crop, kernel launch overhead)
  3. Octopus (block metadata, single kernel)
  4. TensorRT (padded batch resize) — runs in subprocess

Two scenarios:
  A. Uniform crops   — all 256x256 (TensorRT best case)
  B. Variable crops   — random 32-512px (Octopus best case)

Target: 1000 crops from 4K source -> 224x224 bilinear
Hardware: Jetson Orin Nano (8GB)
"""

import numpy as np
import time
import subprocess
import json
import sys
import os
import tempfile

# ============================================
# CONFIG
# ============================================
NUM_CROPS = 1000
TARGET_W, TARGET_H = 224, 224
CHANNELS = 3
SRC_W, SRC_H = 3840, 2160
WARMUP = 5
ITERATIONS = 30
SEED = 42


def generate_crops(num_crops, src_w, src_h, variable=True, seed=42):
    rng = np.random.RandomState(seed)
    crops = []
    for _ in range(num_crops):
        if variable:
            w = rng.randint(32, 513)
            h = rng.randint(32, 513)
        else:
            w, h = 256, 256
        x = rng.randint(0, src_w - w)
        y = rng.randint(0, src_h - h)
        crops.append((x, y, w, h))
    return crops


# ============================================
# 1. CPU OpenCV
# ============================================
def bench_cpu_opencv(src_img, crops, iterations=5):
    import cv2
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        for (x, y, w, h) in crops:
            roi = src_img[y:y+h, x:x+w]
            _ = cv2.resize(roi, (TARGET_W, TARGET_H),
                           interpolation=cv2.INTER_LINEAR)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


# ============================================
# 2. Individual CUDA kernels (numba)
# ============================================
def bench_individual_kernels(src_img_flat, crops, iterations=10):
    from numba import cuda
    from numba import float32, int32, uint8

    @cuda.jit(fastmath=True)
    def single_crop_kernel(src_flat, src_w, src_h,
                           crop_x, crop_y, crop_w, crop_h,
                           out_patch):
        start = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        stride = cuda.gridsize(1)
        total = TARGET_W * TARGET_H

        scale_x = crop_w / float32(TARGET_W)
        scale_y = crop_h / float32(TARGET_H)
        max_x = src_w - 2
        max_y = src_h - 2

        for i in range(start, total, stride):
            ty = i // TARGET_W
            tx = i % TARGET_W
            gx = tx * scale_x + crop_x
            gy = ty * scale_y + crop_y
            ix = int32(gx)
            iy = int32(gy)
            if ix < 0: ix = 0
            elif ix > max_x: ix = max_x
            if iy < 0: iy = 0
            elif iy > max_y: iy = max_y
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

    src_dev = cuda.to_device(src_img_flat)
    out_patches = [cuda.device_array((TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)
                   for _ in range(NUM_CROPS)]

    tpb = 256
    bpg = (TARGET_W * TARGET_H + tpb - 1) // tpb

    for _ in range(3):
        x, y, w, h = crops[0]
        single_crop_kernel[bpg, tpb](src_dev, SRC_W, SRC_H, x, y, w, h, out_patches[0])
        cuda.synchronize()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        for i, (x, y, w, h) in enumerate(crops):
            single_crop_kernel[bpg, tpb](
                src_dev, SRC_W, SRC_H, x, y, w, h, out_patches[i])
        cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times)


# ============================================
# 3. Octopus (numba)
# ============================================
def bench_octopus(src_img_flat, crops, iterations=30):
    from numba import cuda
    from numba import float32, int32, uint8

    @cuda.jit(fastmath=True)
    def octopus_bilinear_kernel(src_flat, metadata, out_tensor):
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

        scale_x = crop_w / float32(TARGET_W)
        scale_y = crop_h / float32(TARGET_H)
        start = cuda.threadIdx.x
        stride = cuda.blockDim.x
        total_pixels = TARGET_W * TARGET_H
        max_x_idx = src_w - 1
        max_y_idx = src_h - 1

        for i in range(start, total_pixels, stride):
            ty = i // TARGET_W
            tx = i % TARGET_W
            gx = tx * scale_x + crop_x
            gy = ty * scale_y + crop_y
            ix = int32(gx)
            iy = int32(gy)
            if ix < 0: ix = 0
            elif ix >= max_x_idx: ix = max_x_idx - 1
            if iy < 0: iy = 0
            elif iy >= max_y_idx: iy = max_y_idx - 1
            fx = gx - ix
            fy = gy - iy
            base_idx = src_offset + (iy * src_w + ix) * CHANNELS
            down_idx = base_idx + src_w * CHANNELS
            for c in range(CHANNELS):
                p00 = float32(src_flat[base_idx + c])
                p10 = float32(src_flat[base_idx + CHANNELS + c])
                p01 = float32(src_flat[down_idx + c])
                p11 = float32(src_flat[down_idx + CHANNELS + c])
                top = p00 + (p10 - p00) * fx
                bot = p01 + (p11 - p01) * fx
                val = top + (bot - top) * fy
                val = val + 0.5
                if val < 0: val = 0.0
                if val > 255: val = 255.0
                out_tensor[dst_idx, ty, tx, c] = uint8(val)

    meta_host = np.zeros((NUM_CROPS, 8), dtype=np.int32)
    for i, (x, y, w, h) in enumerate(crops):
        meta_host[i] = [0, SRC_W, SRC_H, x, y, w, h, i]

    src_dev = cuda.to_device(src_img_flat)
    meta_dev = cuda.to_device(meta_host)
    out_dev = cuda.device_array((NUM_CROPS, TARGET_H, TARGET_W, CHANNELS),
                                dtype=np.uint8)

    threads_per_block = 256
    blocks = NUM_CROPS

    for _ in range(WARMUP):
        octopus_bilinear_kernel[blocks, threads_per_block](
            src_dev, meta_dev, out_dev)
        cuda.synchronize()

    kernel_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        octopus_bilinear_kernel[blocks, threads_per_block](
            src_dev, meta_dev, out_dev)
        cuda.synchronize()
        kernel_times.append((time.perf_counter() - t0) * 1000)

    e2e_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        src_dev2 = cuda.to_device(src_img_flat)
        meta_dev2 = cuda.to_device(meta_host)
        out_dev2 = cuda.device_array((NUM_CROPS, TARGET_H, TARGET_W, CHANNELS),
                                     dtype=np.uint8)
        octopus_bilinear_kernel[blocks, threads_per_block](
            src_dev2, meta_dev2, out_dev2)
        result = out_dev2.copy_to_host()
        cuda.synchronize()
        e2e_times.append((time.perf_counter() - t0) * 1000)

    return np.median(kernel_times), np.median(e2e_times)


# ============================================
# 4. TensorRT — subprocess
# ============================================
TRT_SUBPROCESS_CODE = """
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import json
import sys

NUM_CROPS = 1000
TARGET_W, TARGET_H = 224, 224
CHANNELS = 3
SRC_W, SRC_H = 3840, 2160
WARMUP = 5
ITERATIONS = 10
SEED = 42

def generate_crops(num_crops, src_w, src_h, variable=True, seed=42):
    rng = np.random.RandomState(seed)
    crops = []
    for _ in range(num_crops):
        if variable:
            w = rng.randint(32, 513)
            h = rng.randint(32, 513)
        else:
            w, h = 256, 256
        x = rng.randint(0, src_w - w)
        y = rng.randint(0, src_h - h)
        crops.append((x, y, w, h))
    return crops

def bench_trt(variable):
    crops = generate_crops(NUM_CROPS, SRC_W, SRC_H, variable=variable, seed=SEED)
    max_w = max(c[2] for c in crops)
    max_h = max(c[3] for c in crops)
    batch = NUM_CROPS

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    input_tensor = network.add_input("input", trt.float32, (batch, CHANNELS, max_h, max_w))
    resize_layer = network.add_resize(input_tensor)
    resize_layer.scales = [1.0, 1.0, TARGET_H / float(max_h), TARGET_W / float(max_w)]
    resize_layer.resize_mode = trt.InterpolationMode.LINEAR
    output_tensor = resize_layer.get_output(0)
    output_tensor.name = "output"
    network.mark_output(output_tensor)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return {"error": "Engine build failed"}

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()

    np.random.seed(SEED)
    src_img = np.random.randint(0, 255, (SRC_H, SRC_W, CHANNELS), dtype=np.uint8)
    padded = np.zeros((batch, CHANNELS, max_h, max_w), dtype=np.float32)
    for i, (x, y, w, h) in enumerate(crops):
        crop = src_img[y:y+h, x:x+w, :]
        padded[i, :, :h, :w] = crop.transpose(2, 0, 1).astype(np.float32)

    output_shape = (batch, CHANNELS, TARGET_H, TARGET_W)
    output_host = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(padded.nbytes)
    d_output = cuda.mem_alloc(output_host.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, padded, stream)
    stream.synchronize()
    for _ in range(WARMUP):
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        stream.synchronize()

    kernel_times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        stream.synchronize()
        kernel_times.append((time.perf_counter() - t0) * 1000)

    e2e_times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_input, padded, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, d_output, stream)
        stream.synchronize()
        e2e_times.append((time.perf_counter() - t0) * 1000)

    total_real = sum(w * h for (_, _, w, h) in crops)
    total_padded = batch * max_w * max_h
    waste_pct = (1 - total_real / total_padded) * 100
    padded_mb = padded.nbytes / (1024 * 1024)

    d_input.free()
    d_output.free()

    return {
        "kernel_ms": float(np.median(kernel_times)),
        "e2e_ms": float(np.median(e2e_times)),
        "waste_pct": float(waste_pct),
        "padded_mb": float(padded_mb),
        "max_crop": str(max_w) + "x" + str(max_h),
    }

results = {}
for name, var in [("uniform", False), ("variable", True)]:
    try:
        results[name] = bench_trt(var)
    except Exception as e:
        results[name] = {"error": str(e)}

print(json.dumps(results), flush=True)
"""


def run_trt_subprocess():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(TRT_SUBPROCESS_CODE)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=180
        )
        if result.returncode != 0:
            print(f"  [ERROR] TensorRT subprocess failed:")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-10:]:
                print(f"    {line}")
            return None

        stdout_lines = result.stdout.strip().split('\n')
        for line in reversed(stdout_lines):
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)

        print(f"  [ERROR] No JSON output. stdout:")
        print(result.stdout[-500:] if result.stdout else "(empty)")
        return None

    except subprocess.TimeoutExpired:
        print("  [ERROR] TensorRT subprocess timed out (180s)")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None
    finally:
        os.unlink(tmp_path)


# ============================================
# MAIN
# ============================================
def main():
    print("=" * 70)
    print("  ARENA BENCHMARK: Crop+Resize on Jetson")
    print("=" * 70)
    print(f"  Source: {SRC_W}x{SRC_H} ({CHANNELS}ch)")
    print(f"  Crops:  {NUM_CROPS} -> {TARGET_W}x{TARGET_H}")
    print(f"  Warmup: {WARMUP}, Iterations: {ITERATIONS}")
    print()

    np.random.seed(SEED)
    src_img = np.random.randint(0, 255, (SRC_H, SRC_W, CHANNELS), dtype=np.uint8)
    src_flat = src_img.reshape(-1).astype(np.uint8)

    numba_results = {}

    for scenario_name, variable in [("uniform", False), ("variable", True)]:
        label = "UNIFORM (256x256)" if not variable else "VARIABLE (32-512px)"
        crops = generate_crops(NUM_CROPS, SRC_W, SRC_H,
                               variable=variable, seed=SEED)
        res = {}

        print(f"--- {label} ---")

        print("  [1] CPU OpenCV...")
        try:
            res['cpu'] = bench_cpu_opencv(src_img, crops, iterations=5)
            print(f"      -> {res['cpu']:.1f} ms")
        except Exception as e:
            print(f"      [ERROR] {e}")

        print("  [2] Individual CUDA kernels...")
        try:
            res['individual'] = bench_individual_kernels(src_flat, crops, iterations=10)
            print(f"      -> {res['individual']:.1f} ms")
        except Exception as e:
            print(f"      [ERROR] {e}")

        print("  [3] Octopus...")
        try:
            k, e = bench_octopus(src_flat, crops, iterations=ITERATIONS)
            res['oct_kernel'] = k
            res['oct_e2e'] = e
            print(f"      -> Kernel: {k:.1f} ms | E2E: {e:.1f} ms")
        except Exception as e:
            print(f"      [ERROR] {e}")

        numba_results[scenario_name] = res
        print()

    print("  [4] TensorRT (subprocess, builds 2 engines)...")
    print("      This takes ~30s for engine building, please wait...")
    trt_results = run_trt_subprocess()
    if trt_results:
        for scenario in ["uniform", "variable"]:
            if scenario in trt_results:
                r = trt_results[scenario]
                if "error" in r:
                    print(f"      {scenario}: [ERROR] {r['error']}")
                else:
                    print(f"      {scenario}: Kernel {r['kernel_ms']:.1f}ms | "
                          f"E2E {r['e2e_ms']:.1f}ms | "
                          f"Pad waste {r['waste_pct']:.1f}% | "
                          f"Input {r['padded_mb']:.0f}MB")
    print()

    # ---- Final Summary ----
    for scenario_name, variable in [("uniform", False), ("variable", True)]:
        label = "UNIFORM (256x256)" if not variable else "VARIABLE (32-512px)"
        res = numba_results.get(scenario_name, {})
        trt = trt_results.get(scenario_name, {}) if trt_results else {}

        print("=" * 70)
        print(f"  RESULTS: {label}")
        print("=" * 70)

        baseline = res.get('cpu', 0)

        print(f"  {'Method':<40} {'Time':>10} {'vs CPU':>10}")
        print(f"  {'-'*60}")

        if 'cpu' in res:
            print(f"  {'CPU OpenCV':<40} {res['cpu']:>8.1f}ms {'1.00x':>10}")

        if 'individual' in res:
            sp = baseline / res['individual'] if res['individual'] > 0 else 0
            print(f"  {'Individual CUDA (1000 launches)':<40} "
                  f"{res['individual']:>8.1f}ms {sp:>9.2f}x")

        if 'oct_kernel' in res:
            sp = baseline / res['oct_kernel'] if res['oct_kernel'] > 0 else 0
            print(f"  {'Octopus kernel (single launch)':<40} "
                  f"{res['oct_kernel']:>8.1f}ms {sp:>9.2f}x")

        if 'oct_e2e' in res:
            sp = baseline / res['oct_e2e'] if res['oct_e2e'] > 0 else 0
            print(f"  {'Octopus end-to-end':<40} "
                  f"{res['oct_e2e']:>8.1f}ms {sp:>9.2f}x")

        if trt and 'kernel_ms' in trt:
            sp = baseline / trt['kernel_ms'] if trt['kernel_ms'] > 0 else 0
            print(f"  {'TensorRT kernel (padded)':<40} "
                  f"{trt['kernel_ms']:>8.1f}ms {sp:>9.2f}x"
                  f"  [waste: {trt['waste_pct']:.0f}%]")

            sp_e2e = baseline / trt['e2e_ms'] if trt['e2e_ms'] > 0 else 0
            print(f"  {'TensorRT end-to-end (padded)':<40} "
                  f"{trt['e2e_ms']:>8.1f}ms {sp_e2e:>9.2f}x"
                  f"  [input: {trt['padded_mb']:.0f}MB]")

        # Head-to-head
        if 'oct_kernel' in res and trt and 'kernel_ms' in trt:
            ratio = trt['kernel_ms'] / res['oct_kernel']
            if ratio > 1:
                print(f"\n  >> Octopus kernel {ratio:.2f}x faster than TensorRT")
            else:
                print(f"\n  >> TensorRT kernel {1/ratio:.2f}x faster than Octopus")

            ratio_e2e = trt['e2e_ms'] / res['oct_e2e']
            if ratio_e2e > 1:
                print(f"  >> Octopus E2E {ratio_e2e:.2f}x faster than TensorRT")
            else:
                print(f"  >> TensorRT E2E {1/ratio_e2e:.2f}x faster than Octopus")

        print()


if __name__ == "__main__":
    main()