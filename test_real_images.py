import numpy as np
from numba import cuda
from PIL import Image
import time
import os
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

@cuda.jit
def kernel_c_normalize(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = (images[i] - 0.5) * 2.0  # Normalize to [-1, 1]

@cuda.jit
def kernel_c_threshold(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = 1.0 if images[i] > 0.5 else 0.0

def load_real_images(folder):
    """Load real images from folder"""
    images = []
    sizes = []
    widths = []
    heights = []
    
    folder = Path(folder)
    files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    files = sorted(files)[:100]  # Max 100 images
    
    print(f"Loading {len(files)} images from {folder}...")
    
    for f in files:
        img = Image.open(f).convert('L')  # Grayscale
        w, h = img.size
        arr = np.array(img, dtype=np.float32).flatten() / 255.0
        images.append(arr)
        sizes.append(len(arr))
        widths.append(w)
        heights.append(h)
    
    return images, sizes, widths, heights

def benchmark_real_images(folder):
    images, sizes, widths, heights = load_real_images(folder)
    
    if len(images) == 0:
        print("No images found!")
        return
    
    num_images = len(images)
    sizes = np.array(sizes, dtype=np.int64)
    widths = np.array(widths, dtype=np.int32)
    heights = np.array(heights, dtype=np.int32)
    
    offsets = np.zeros(num_images + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    total_pixels = int(offsets[-1])
    images_flat = np.concatenate(images).astype(np.float32)
    
    print(f"\n{'='*60}")
    print(f"REAL IMAGE BENCHMARK")
    print(f"{'='*60}")
    print(f"Images: {num_images}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Size range: {sizes.min():,} - {sizes.max():,} pixels")
    print(f"Size imbalance: {sizes.max() / sizes.min():.1f}x")
    
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

    # GPU
    d_images = cuda.to_device(images_flat)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    d_offsets = cuda.to_device(offsets)
    d_b2i = cuda.to_device(block_to_image)
    d_bs = cuda.to_device(block_start)
    d_be = cuda.to_device(block_end)

    blocks_b = (total_pixels + 255) // 256

    # Warmup
    kernel_b[blocks_b, 256](d_images, d_offsets, num_images, d_output)
    kernel_c[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
    cuda.synchronize()

    print(f"\n[Multiply (brightness adjustment)]")
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
    print(f"  B (Binary Search): {b_ms:.2f} ms")
    print(f"  C (Block Metadata): {c_ms:.2f} ms")
    print(f"  Speedup: {b_ms/c_ms:.2f}x")

    print(f"\n[Normalize (for ML preprocessing)]")
    times_c_norm = []
    for _ in range(5):
        start = time.perf_counter()
        kernel_c_normalize[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
        cuda.synchronize()
        times_c_norm.append(time.perf_counter() - start)
    print(f"  C (Block Metadata): {np.median(times_c_norm)*1000:.2f} ms")

    print(f"\n[Threshold (binarization)]")
    times_c_thresh = []
    for _ in range(5):
        start = time.perf_counter()
        kernel_c_threshold[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
        cuda.synchronize()
        times_c_thresh.append(time.perf_counter() - start)
    print(f"  C (Block Metadata): {np.median(times_c_thresh)*1000:.2f} ms")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "frames"
    benchmark_real_images(folder)
