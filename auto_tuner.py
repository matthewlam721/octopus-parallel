import numpy as np
from numba import cuda
import time

@cuda.jit
def kernel_b_multiply(images, offsets, num_images, output):
    idx = cuda.grid(1)
    if idx >= len(images):
        return
    lo, hi = 0, num_images
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid + 1] <= idx:
            lo = mid + 1
        else:
            hi = mid
    output[idx] = images[idx] * 2.0

@cuda.jit
def kernel_c_multiply(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = images[i] * 2.0

@cuda.jit(device=True)
def binary_search(offsets, num_images, idx):
    lo, hi = 0, num_images
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid + 1] <= idx:
            lo = mid + 1
        else:
            hi = mid
    return lo

@cuda.jit
def kernel_b_blur(images, offsets, widths, heights, num_images, output):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = len(images)
    for pixel_idx in range(tid, n, stride):
        img_id = binary_search(offsets, num_images, pixel_idx)
        offset = offsets[img_id]
        w = widths[img_id]
        h = heights[img_id]
        local_idx = pixel_idx - offset
        y = local_idx // w
        x = local_idx % w
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[pixel_idx] = images[pixel_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images[neighbor_idx]
            output[pixel_idx] = total / 9.0

@cuda.jit
def kernel_c_blur(images, offsets, widths, heights, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    img_id = block_to_image[block_id]
    offset = offsets[img_id]
    w = widths[img_id]
    h = heights[img_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for local_idx in range(local_start + tid, local_end, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[global_idx] = images[global_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images[neighbor_idx]
            output[global_idx] = total / 9.0

def auto_tune(num_images=10000, threshold=1.5):
    """
    Run micro-probe to decide B vs C.
    Returns 'C' if block metadata is faster, 'B' otherwise.
    """
    print("=" * 50)
    print("AUTO-TUNER: Micro-probe")
    print("=" * 50)
    
    np.random.seed(42)
    sizes = np.random.randint(100, 500, num_images).astype(np.int64)
    widths = np.sqrt(sizes).astype(np.int32)
    widths = np.maximum(widths, 3)
    heights = (sizes // widths).astype(np.int32)
    heights = np.maximum(heights, 3)
    sizes = (widths * heights).astype(np.int64)
    
    offsets = np.zeros(num_images + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    total_pixels = int(offsets[-1])
    images_flat = np.random.rand(total_pixels).astype(np.float32)

    # Setup C metadata
    THREADS = 256
    block_to_image = []
    block_start = []
    block_end = []
    for img_id in range(num_images):
        s = int(sizes[img_id])
        for b in range((s + THREADS - 1) // THREADS):
            block_to_image.append(img_id)
            block_start.append(b * THREADS)
            block_end.append(min((b + 1) * THREADS, s))

    block_to_image = np.array(block_to_image, dtype=np.int32)
    block_start = np.array(block_start, dtype=np.int64)
    block_end = np.array(block_end, dtype=np.int64)

    # GPU
    d_images = cuda.to_device(images_flat)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    d_offsets = cuda.to_device(offsets)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_b2i = cuda.to_device(block_to_image)
    d_bs = cuda.to_device(block_start)
    d_be = cuda.to_device(block_end)

    blocks_b = (total_pixels + 255) // 256

    # Warmup
    kernel_b_multiply[blocks_b, 256](d_images, d_offsets, num_images, d_output)
    kernel_c_multiply[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
    cuda.synchronize()

    # Test 1: Multiply (memory-bound)
    times_b_mul = []
    for _ in range(3):
        start = time.perf_counter()
        kernel_b_multiply[blocks_b, 256](d_images, d_offsets, num_images, d_output)
        cuda.synchronize()
        times_b_mul.append(time.perf_counter() - start)

    times_c_mul = []
    for _ in range(3):
        start = time.perf_counter()
        kernel_c_multiply[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
        cuda.synchronize()
        times_c_mul.append(time.perf_counter() - start)

    mul_speedup = np.median(times_b_mul) / np.median(times_c_mul)
    print(f"\n[Probe 1: Multiply (memory-bound)]")
    print(f"  B: {np.median(times_b_mul)*1000:.2f} ms")
    print(f"  C: {np.median(times_c_mul)*1000:.2f} ms")
    print(f"  Speedup: {mul_speedup:.2f}x")

    # Test 2: Blur (compute-bound)
    kernel_b_blur[256, 256](d_images, d_offsets[:-1], d_widths, d_heights, num_images, d_output)
    kernel_c_blur[len(block_to_image), 256](d_images, d_offsets[:-1], d_widths, d_heights, d_b2i, d_bs, d_be, d_output)
    cuda.synchronize()

    times_b_blur = []
    for _ in range(3):
        start = time.perf_counter()
        kernel_b_blur[256, 256](d_images, d_offsets[:-1], d_widths, d_heights, num_images, d_output)
        cuda.synchronize()
        times_b_blur.append(time.perf_counter() - start)

    times_c_blur = []
    for _ in range(3):
        start = time.perf_counter()
        kernel_c_blur[len(block_to_image), 256](d_images, d_offsets[:-1], d_widths, d_heights, d_b2i, d_bs, d_be, d_output)
        cuda.synchronize()
        times_c_blur.append(time.perf_counter() - start)

    blur_speedup = np.median(times_b_blur) / np.median(times_c_blur)
    print(f"\n[Probe 2: Blur (compute-bound)]")
    print(f"  B: {np.median(times_b_blur)*1000:.2f} ms")
    print(f"  C: {np.median(times_c_blur)*1000:.2f} ms")
    print(f"  Speedup: {blur_speedup:.2f}x")

    # Decision
    print(f"\n{'=' * 50}")
    print("DECISION")
    print(f"{'=' * 50}")
    
    if mul_speedup > threshold and blur_speedup < threshold:
        decision = 'C'
        reason = f"Multiply speedup ({mul_speedup:.2f}x) > {threshold}x, Blur speedup ({blur_speedup:.2f}x) < {threshold}x"
        print(f"→ Use BLOCK METADATA (C)")
        print(f"  Reason: Memory-bound operations benefit significantly")
    elif mul_speedup > threshold and blur_speedup > threshold:
        decision = 'C'
        reason = f"Both speedups > {threshold}x"
        print(f"→ Use BLOCK METADATA (C)")
        print(f"  Reason: C wins across workload types")
    else:
        decision = 'B'
        reason = f"Multiply speedup ({mul_speedup:.2f}x) <= {threshold}x"
        print(f"→ Use BINARY SEARCH (B)")
        print(f"  Reason: Overhead doesn't justify block metadata")
    
    print(f"\n{'=' * 50}")
    
    return decision, mul_speedup, blur_speedup

if __name__ == "__main__":
    decision, mul_sp, blur_sp = auto_tune(num_images=50000)
