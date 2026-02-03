import numpy as np
from numba import cuda
import time

@cuda.jit(device=True)
def binary_search_image(offsets, num_images, pixel_idx):
    left = 0
    right = num_images - 1
    while left < right:
        mid = (left + right + 1) // 2
        if offsets[mid] <= pixel_idx:
            left = mid
        else:
            right = mid - 1
    return left

@cuda.jit
def kernel_b_blur(images, offsets, widths, heights, num_images, output):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = len(images)
    
    for pixel_idx in range(tid, n, stride):
        img_id = binary_search_image(offsets, num_images, pixel_idx)
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

print("=" * 60)
print("JETSON BLUR KERNEL BENCHMARK (B vs C only)")
print("=" * 60)

for num_images in [50000, 100000, 150000]:
    np.random.seed(42)
    
    # Generate square-ish images for blur
    sizes = np.random.randint(100, 500, num_images).astype(np.int64)
    widths = np.sqrt(sizes).astype(np.int32)
    widths = np.maximum(widths, 3)  # min width 3 for blur
    heights = (sizes // widths).astype(np.int32)
    heights = np.maximum(heights, 3)  # min height 3 for blur
    sizes = (widths * heights).astype(np.int64)
    
    offsets = np.zeros(num_images + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    total_pixels = int(offsets[-1])
    images_flat = np.random.rand(total_pixels).astype(np.float32)

    # Setup C
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

    # GPU transfer
    d_images = cuda.to_device(images_flat)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    d_offsets = cuda.to_device(offsets[:-1])  # exclude last
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_b2i = cuda.to_device(block_to_image)
    d_bs = cuda.to_device(block_start)
    d_be = cuda.to_device(block_end)

    blocks_b = 256
    threads = 256
    
    # Warmup
    kernel_b_blur[blocks_b, threads](d_images, d_offsets, d_widths, d_heights, num_images, d_output)
    kernel_c_blur[len(block_to_image), threads](d_images, d_offsets, d_widths, d_heights, d_b2i, d_bs, d_be, d_output)
    cuda.synchronize()

    # Benchmark B
    times_b = []
    for _ in range(5):
        start = time.perf_counter()
        kernel_b_blur[blocks_b, threads](d_images, d_offsets, d_widths, d_heights, num_images, d_output)
        cuda.synchronize()
        times_b.append(time.perf_counter() - start)

    # Benchmark C
    times_c = []
    for _ in range(5):
        start = time.perf_counter()
        kernel_c_blur[len(block_to_image), threads](d_images, d_offsets, d_widths, d_heights, d_b2i, d_bs, d_be, d_output)
        cuda.synchronize()
        times_c.append(time.perf_counter() - start)

    b_ms = np.median(times_b) * 1000
    c_ms = np.median(times_c) * 1000
    speedup = b_ms / c_ms
    
    print(f"{num_images//1000}K imgs | B: {b_ms:.1f}ms | C: {c_ms:.1f}ms | Speedup: {speedup:.2f}x")
    
    del d_images, d_output, d_offsets, d_widths, d_heights, d_b2i, d_bs, d_be

print("=" * 60)
