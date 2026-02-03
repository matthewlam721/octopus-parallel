import numpy as np
from numba import cuda
from PIL import Image
import time
from pathlib import Path

@cuda.jit
def kernel_b(images, offsets, num_images, output):
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
def kernel_c(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = images[i] * 2.0

def load_real_images_resized(folder, target_sizes):
    """Load real images, resize to smaller for testing"""
    images = []
    sizes = []
    
    folder = Path(folder)
    files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    files = sorted(files)
    
    for f in files:
        img = Image.open(f).convert('L')
        for target in target_sizes:
            # Resize to target
            resized = img.resize(target, Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32).flatten() / 255.0
            images.append(arr)
            sizes.append(len(arr))
    
    return images, sizes

def benchmark(num_images, sizes_range, name):
    np.random.seed(42)
    sizes = np.random.randint(sizes_range[0], sizes_range[1], num_images).astype(np.int64)
    
    offsets = np.zeros(num_images + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    total_pixels = int(offsets[-1])
    
    # Check memory
    mem_mb = total_pixels * 4 / 1e6
    if mem_mb > 1500:
        print(f"{name}: SKIP (too large: {mem_mb:.0f} MB)")
        return None, None
    
    images_flat = np.random.rand(total_pixels).astype(np.float32)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Images: {num_images:,}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Size range: {sizes_range[0]:,} - {sizes_range[1]:,}")
    print(f"Binary search depth: log2({num_images}) = {int(np.log2(num_images))}")

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

    d_images = cuda.to_device(images_flat)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    d_offsets = cuda.to_device(offsets)
    d_b2i = cuda.to_device(block_to_image)
    d_bs = cuda.to_device(block_start)
    d_be = cuda.to_device(block_end)

    blocks_b = (total_pixels + 255) // 256

    kernel_b[blocks_b, 256](d_images, d_offsets, num_images, d_output)
    kernel_c[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
    cuda.synchronize()

    times_b = []
    for _ in range(5):
        start = time.perf_counter()
        kernel_b[blocks_b, 256](d_images, d_offsets, num_images, d_output)
        cuda.synchronize()
        times_b.append(time.perf_counter() - start)

    times_c = []
    for _ in range(5):
        start = time.perf_counter()
        kernel_c[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
        cuda.synchronize()
        times_c.append(time.perf_counter() - start)

    b_ms = np.median(times_b) * 1000
    c_ms = np.median(times_c) * 1000
    print(f"B (Binary Search): {b_ms:.2f} ms")
    print(f"C (Block Metadata): {c_ms:.2f} ms")
    print(f"Speedup: {b_ms/c_ms:.2f}x")
    
    del d_images, d_output, d_offsets, d_b2i, d_bs, d_be

print("="*60)
print("JETSON REAL-WORLD SCALE TEST")
print("="*60)

# Simulate realistic scenarios
configs = [
    (100, (14400, 2073600), "100 Real Photos (mixed sizes)"),
    (500, (5000, 500000), "500 Video Frames (varied res)"),
    (1000, (1000, 100000), "1K Thumbnails + HD mix"),
    (5000, (500, 50000), "5K Drone/Satellite tiles"),
    (10000, (100, 10000), "10K Small patches"),
    (50000, (100, 1000), "50K Tiny patches"),
]

for num, sizes_range, name in configs:
    benchmark(num, sizes_range, name)

print("\n" + "="*60)
