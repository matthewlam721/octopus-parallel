"""
TensorRT Variable-Size Benchmark (Memory-Efficient)
====================================================
Processes ONE batch at a time, frees everything before next batch.
Accumulates timing across all batches.
"""
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import json
import gc

NUM_CROPS = 1000
TARGET_W, TARGET_H = 224, 224
CHANNELS = 3
SRC_W, SRC_H = 3840, 2160
WARMUP = 2
ITERATIONS = 5
SEED = 42

def generate_crops(num_crops, src_w, src_h, seed=42):
    rng = np.random.RandomState(seed)
    crops = []
    for _ in range(num_crops):
        w = rng.randint(32, 513)
        h = rng.randint(32, 513)
        x = rng.randint(0, src_w - w)
        y = rng.randint(0, src_h - h)
        crops.append((x, y, w, h))
    return crops

def main():
    crops = generate_crops(NUM_CROPS, SRC_W, SRC_H, seed=SEED)
    np.random.seed(SEED)
    src_img = np.random.randint(0, 255, (SRC_H, SRC_W, CHANNELS), dtype=np.uint8)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    for BATCH_SIZE in [50, 100]:
        num_batches = (NUM_CROPS + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n--- batch_size={BATCH_SIZE} ({num_batches} batches) ---")

        total_waste_pixels = 0
        total_real_pixels = 0
        total_padded_bytes = 0
        total_build_time = 0

        # Collect per-iteration total times
        kernel_totals = [0.0] * ITERATIONS
        e2e_totals = [0.0] * ITERATIONS

        for b in range(num_batches):
            start_idx = b * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, NUM_CROPS)
            b_crops = crops[start_idx:end_idx]
            bs = len(b_crops)
            max_w = max(c[2] for c in b_crops)
            max_h = max(c[3] for c in b_crops)

            # Build engine
            t_build = time.perf_counter()
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 26)  # 64MB

            inp = network.add_input("input", trt.float32, (bs, CHANNELS, max_h, max_w))
            resize = network.add_resize(inp)
            resize.scales = [1.0, 1.0, TARGET_H / float(max_h), TARGET_W / float(max_w)]
            resize.resize_mode = trt.InterpolationMode.LINEAR
            out = resize.get_output(0)
            out.name = "output"
            network.mark_output(out)

            serialized = builder.build_serialized_network(network, config)
            total_build_time += time.perf_counter() - t_build

            if serialized is None:
                print(f"    Batch {b}: engine build FAILED (max {max_w}x{max_h})")
                continue

            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized)
            context = engine.create_execution_context()

            # Prepare padded input
            padded = np.zeros((bs, CHANNELS, max_h, max_w), dtype=np.float32)
            for i, (x, y, w, h) in enumerate(b_crops):
                crop = src_img[y:y+h, x:x+w, :]
                padded[i, :, :h, :w] = crop.transpose(2, 0, 1).astype(np.float32)

            out_shape = (bs, CHANNELS, TARGET_H, TARGET_W)
            out_host = np.empty(out_shape, dtype=np.float32)

            real_px = sum(w * h for (_, _, w, h) in b_crops)
            padded_px = bs * max_w * max_h
            total_real_pixels += real_px
            total_waste_pixels += (padded_px - real_px)
            total_padded_bytes += padded.nbytes

            d_input = cuda.mem_alloc(padded.nbytes)
            d_output = cuda.mem_alloc(out_host.nbytes)
            stream = cuda.Stream()

            # Warmup
            cuda.memcpy_htod_async(d_input, padded, stream)
            stream.synchronize()
            for _ in range(WARMUP):
                context.execute_async_v2(
                    bindings=[int(d_input), int(d_output)],
                    stream_handle=stream.handle)
                stream.synchronize()

            # Kernel timing
            for it in range(ITERATIONS):
                t0 = time.perf_counter()
                context.execute_async_v2(
                    bindings=[int(d_input), int(d_output)],
                    stream_handle=stream.handle)
                stream.synchronize()
                kernel_totals[it] += (time.perf_counter() - t0) * 1000

            # E2E timing
            for it in range(ITERATIONS):
                t0 = time.perf_counter()
                cuda.memcpy_htod_async(d_input, padded, stream)
                context.execute_async_v2(
                    bindings=[int(d_input), int(d_output)],
                    stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(out_host, d_output, stream)
                stream.synchronize()
                e2e_totals[it] += (time.perf_counter() - t0) * 1000

            # FREE everything before next batch
            d_input.free()
            d_output.free()
            del context, engine, runtime, serialized
            del builder, network, config
            del padded, out_host
            gc.collect()

        waste_pct = total_waste_pixels / (total_real_pixels + total_waste_pixels) * 100
        padded_mb = total_padded_bytes / (1024 * 1024)

        kernel_median = np.median(kernel_totals)
        e2e_median = np.median(e2e_totals)

        print(f"    Kernel (sum all batches): {kernel_median:.1f} ms")
        print(f"    E2E (sum all batches):    {e2e_median:.1f} ms")
        print(f"    Pad waste:                {waste_pct:.1f}%")
        print(f"    Total padded size:        {padded_mb:.0f} MB (sequential)")
        print(f"    Engine build time:        {total_build_time:.1f}s total")
        print(f"    vs Octopus kernel (172ms): {kernel_median/172:.2f}x slower")
        print(f"    vs Octopus E2E (243ms):    {e2e_median/243:.2f}x slower")

main()