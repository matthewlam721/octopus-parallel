"""
TRT vs Octopus arena — REAL VisDrone data + VRAM measurement.
"""
import os
import sys
import json
import time
import tempfile
import subprocess
import numpy as np

from arena_benchmark import (
    TARGET_W, TARGET_H, CHANNELS,
    WARMUP, ITERATIONS, SEED,
    bench_cpu_opencv, bench_individual_kernels, bench_octopus,
)
import arena_benchmark as ab
from visdrone_loader import load_visdrone_pool, sample_real_detections


def get_real_setup(num_crops):
    frame, bbox_pool = load_visdrone_pool(max_frames=50)
    src_h, src_w = frame.shape[:2]
    crops = sample_real_detections(bbox_pool, num_crops, seed=SEED)
    src_flat = frame.reshape(-1).astype(np.uint8)
    return frame, src_flat, crops, src_w, src_h


# ============================================
# TRT subprocess — feed real data + VRAM tracking
# ============================================
TRT_REAL_SUBPROCESS_CODE = r"""
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import json
import sys
import os

NPY_FRAME = os.environ['REAL_FRAME_NPY']
NPY_CROPS = os.environ['REAL_CROPS_NPY']
TARGET_W, TARGET_H = 224, 224
CHANNELS = 3
WARMUP = 5
ITERATIONS = 10

src_img = np.load(NPY_FRAME)
crops_arr = np.load(NPY_CROPS)
SRC_H, SRC_W = src_img.shape[:2]
NUM_CROPS = len(crops_arr)
crops = [tuple(c) for c in crops_arr]


def get_vram_mb():
    free, total = cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def bench_trt(uniform):
    if uniform:
        local_crops = []
        rng = np.random.RandomState(42)
        for _ in range(NUM_CROPS):
            w = h = 256
            x = rng.randint(0, SRC_W - w)
            y = rng.randint(0, SRC_H - h)
            local_crops.append((x, y, w, h))
    else:
        local_crops = crops

    max_w = max(c[2] for c in local_crops)
    max_h = max(c[3] for c in local_crops)
    batch = NUM_CROPS

    baseline_vram = get_vram_mb()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    input_tensor = network.add_input("input", trt.float32,
                                       (batch, CHANNELS, max_h, max_w))
    resize_layer = network.add_resize(input_tensor)
    resize_layer.scales = [1.0, 1.0, TARGET_H / float(max_h),
                            TARGET_W / float(max_w)]
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

    padded = np.zeros((batch, CHANNELS, max_h, max_w), dtype=np.float32)
    for i, (x, y, w, h) in enumerate(local_crops):
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
        context.execute_async_v2(bindings=[int(d_input), int(d_output)],
                                  stream_handle=stream.handle)
        stream.synchronize()

    kernel_times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        context.execute_async_v2(bindings=[int(d_input), int(d_output)],
                                  stream_handle=stream.handle)
        stream.synchronize()
        kernel_times.append((time.perf_counter() - t0) * 1000)

    e2e_times = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_input, padded, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)],
                                  stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, d_output, stream)
        stream.synchronize()
        e2e_times.append((time.perf_counter() - t0) * 1000)

    # ★ Peak VRAM while everything allocated
    peak_vram = get_vram_mb()

    total_real = sum(w * h for (_, _, w, h) in local_crops)
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
        "peak_vram_mb": float(peak_vram),
        "baseline_vram_mb": float(baseline_vram),
        "vram_used_mb": float(peak_vram - baseline_vram),
    }


results = {}
for name, uni in [("uniform", True), ("variable", False)]:
    try:
        results[name] = bench_trt(uni)
    except Exception as e:
        results[name] = {"error": str(e)}
print(json.dumps(results), flush=True)
"""


def run_trt_subprocess_real(src_img, real_crops):
    tmpdir = tempfile.mkdtemp(prefix='arena_real_')
    frame_npy = os.path.join(tmpdir, 'frame.npy')
    crops_npy = os.path.join(tmpdir, 'crops.npy')
    code_py = os.path.join(tmpdir, 'trt_run.py')

    np.save(frame_npy, src_img)
    np.save(crops_npy, np.array(real_crops, dtype=np.int32))
    with open(code_py, 'w') as f:
        f.write(TRT_REAL_SUBPROCESS_CODE)

    env = os.environ.copy()
    env['REAL_FRAME_NPY'] = frame_npy
    env['REAL_CROPS_NPY'] = crops_npy

    try:
        result = subprocess.run(
            [sys.executable, code_py],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if result.returncode != 0:
            print("  [TRT ERROR]")
            for line in result.stderr.strip().split('\n')[-15:]:
                print(f"    {line}")
            return None
        for line in reversed(result.stdout.strip().split('\n')):
            if line.strip().startswith('{'):
                return json.loads(line)
        print("  [TRT ERROR] no JSON, stdout tail:")
        print(result.stdout[-500:])
        return None
    except subprocess.TimeoutExpired:
        print("  [TRT ERROR] subprocess timeout (300s)")
        return None
    finally:
        try:
            os.unlink(frame_npy)
            os.unlink(crops_npy)
        except:
            pass


def patch_arena_constants(num_crops, src_w, src_h):
    ab.NUM_CROPS = num_crops
    ab.SRC_W = src_w
    ab.SRC_H = src_h


def main():
    NUM_CROPS = 1000
    print("=" * 70)
    print("  ARENA BENCHMARK — REAL DATA (VisDrone) + VRAM")
    print("=" * 70)
    print(f"  Crops: {NUM_CROPS} -> {TARGET_W}x{TARGET_H}")
    print()

    src_img, src_flat, real_crops, SRC_W, SRC_H = get_real_setup(NUM_CROPS)
    print(f"  Source: {SRC_W}x{SRC_H} REAL VisDrone")
    ws = [c[2] for c in real_crops]
    hs = [c[3] for c in real_crops]
    print(f"    bbox range: w={min(ws)}-{max(ws)}, h={min(hs)}-{max(hs)}")
    print(f"    bbox median: {int(np.median(ws))}x{int(np.median(hs))}")
    print()

    patch_arena_constants(NUM_CROPS, SRC_W, SRC_H)

    numba_results = {}

    rng = np.random.RandomState(SEED)
    uniform_crops = []
    for _ in range(NUM_CROPS):
        w = h = 256
        x = rng.randint(0, SRC_W - w)
        y = rng.randint(0, SRC_H - h)
        uniform_crops.append((x, y, w, h))

    for scenario, crops in [("uniform", uniform_crops),
                             ("variable_real", real_crops)]:
        label = ("UNIFORM 256x256 (TRT best case)" if scenario == "uniform"
                 else "VARIABLE — REAL VisDrone bboxes")
        print(f"--- {label} ---")
        res = {}

        print("  [1] CPU OpenCV...")
        try:
            res['cpu'] = bench_cpu_opencv(src_img, crops, iterations=5)
            print(f"      -> {res['cpu']:.1f} ms")
        except Exception as e:
            print(f"      [ERROR] {e}")

        print("  [2] Individual CUDA kernels...")
        try:
            res['individual'] = bench_individual_kernels(src_flat, crops,
                                                          iterations=10)
            print(f"      -> {res['individual']:.1f} ms")
        except Exception as e:
            print(f"      [ERROR] {e}")

        print("  [3] Octopus (with VRAM tracking)...")
        try:
            from numba import cuda as nb_cuda
            ctx = nb_cuda.current_context()
            free_before, total = ctx.get_memory_info()

            k, e = bench_octopus(src_flat, crops, iterations=ITERATIONS)

            free_after, _ = ctx.get_memory_info()
            oct_vram_mb = (free_before - free_after) / (1024 * 1024)

            res['oct_kernel'] = k
            res['oct_e2e'] = e
            res['oct_vram_mb'] = oct_vram_mb
            print(f"      -> Kernel: {k:.1f} ms | E2E: {e:.1f} ms | "
                  f"VRAM: {oct_vram_mb:.1f} MB")
        except Exception as e:
            print(f"      [ERROR] {e}")

        numba_results[scenario] = res
        print()

    print("  [4] TensorRT (real data subprocess)...")
    print("      Building 2 engines (uniform + variable_real)...")
    trt_results = run_trt_subprocess_real(src_img, real_crops)
    if trt_results:
        for k in ["uniform", "variable"]:
            r = trt_results.get(k, {})
            if "error" in r:
                print(f"      {k}: [ERROR] {r['error']}")
            else:
                print(f"      {k}: kernel {r['kernel_ms']:.1f}ms | "
                      f"e2e {r['e2e_ms']:.1f}ms | "
                      f"VRAM {r.get('vram_used_mb', 0):.0f}MB | "
                      f"input {r['padded_mb']:.0f}MB")
    print()

    for scenario in ["uniform", "variable_real"]:
        label = ("UNIFORM 256x256" if scenario == "uniform"
                 else "VARIABLE REAL VisDrone")
        res = numba_results.get(scenario, {})
        trt_key = "uniform" if scenario == "uniform" else "variable"
        trt = trt_results.get(trt_key, {}) if trt_results else {}

        print("=" * 70)
        print(f"  RESULTS: {label}")
        print("=" * 70)

        baseline = res.get('cpu', 0)
        print(f"  {'Method':<40} {'Time':>10} {'vs CPU':>10}")
        print(f"  {'-'*60}")

        if 'cpu' in res:
            print(f"  {'CPU OpenCV':<40} {res['cpu']:>8.1f}ms {'1.00x':>10}")
        if 'individual' in res:
            sp = baseline / res['individual'] if res['individual'] else 0
            print(f"  {'Individual CUDA':<40} {res['individual']:>8.1f}ms {sp:>9.2f}x")
        if 'oct_kernel' in res:
            sp = baseline / res['oct_kernel']
            vram = res.get('oct_vram_mb', 0)
            print(f"  {'Octopus kernel':<40} {res['oct_kernel']:>8.1f}ms {sp:>9.2f}x"
                  f"  [VRAM: {vram:.1f}MB]")
        if 'oct_e2e' in res:
            sp = baseline / res['oct_e2e']
            print(f"  {'Octopus e2e':<40} {res['oct_e2e']:>8.1f}ms {sp:>9.2f}x")

        if trt and 'kernel_ms' in trt:
            sp = baseline / trt['kernel_ms']
            vram = trt.get('vram_used_mb', 0)
            print(f"  {'TensorRT kernel':<40} {trt['kernel_ms']:>8.1f}ms {sp:>9.2f}x"
                  f"  [waste: {trt['waste_pct']:.0f}%, VRAM: {vram:.0f}MB]")
            sp = baseline / trt['e2e_ms']
            print(f"  {'TensorRT e2e':<40} {trt['e2e_ms']:>8.1f}ms {sp:>9.2f}x"
                  f"  [input: {trt['padded_mb']:.0f}MB]")

        if 'oct_vram_mb' in res and trt and 'vram_used_mb' in trt:
            ratio = trt['vram_used_mb'] / max(res['oct_vram_mb'], 0.1)
            print(f"\n  >> VRAM:   TRT {trt['vram_used_mb']:.0f}MB vs "
                  f"Octopus {res['oct_vram_mb']:.1f}MB ({ratio:.0f}x more for TRT)")

        if 'oct_kernel' in res and trt and 'kernel_ms' in trt:
            r = trt['kernel_ms'] / res['oct_kernel']
            print(f"  >> Kernel: " + (f"Octopus {r:.2f}x vs TRT" if r > 1
                  else f"TRT {1/r:.2f}x faster than Octopus"))
            r = trt['e2e_ms'] / res['oct_e2e']
            print(f"  >> E2E:    " + (f"Octopus {r:.2f}x vs TRT" if r > 1
                  else f"TRT {1/r:.2f}x faster than Octopus"))
        print()


if __name__ == "__main__":
    main()