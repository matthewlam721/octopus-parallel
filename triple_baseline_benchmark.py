"""
Triple baseline benchmark:
- A: O(1) lookup table (huge memory)
- B: Binary search (no extra memory, O(log M) kernel)
- C: Block metadata (small memory, O(1) kernel)

--flush-cache to simulate real workload
--heavy for 10x iteration stress test
"""
from numba import njit
from numba import cuda
import numpy as np
import time
import math
from pathlib import Path
import argparse

# ============================================
# CACHE FLUSH UTILITY
# ============================================

_flush_garbage = None
_flush_output = None


@cuda.jit
def _touch_memory_kernel(arr, out):
    """Touch all memory to fill cache"""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(tid, arr.shape[0], stride):
        out[i] = arr[i] * 1.0001


def init_cache_flush(size_mb=100):
    """Initialize cache flush arrays"""
    global _flush_garbage, _flush_output
    num_floats = size_mb * 1024 * 1024 // 4
    _flush_garbage = cuda.to_device(np.random.rand(num_floats).astype(np.float32))
    _flush_output = cuda.device_array(num_floats, dtype=np.float32)
    print(f"  Cache flush initialized: {size_mb} MB")


def flush_l2_cache():
    """
    Fill L2 cache with garbage to simulate real workload.
    RTX 4090 has 72 MB L2, so we touch 100 MB to ensure full flush.
    """
    global _flush_garbage, _flush_output
    if _flush_garbage is None:
        init_cache_flush()
    
    _touch_memory_kernel[512, 256](_flush_garbage, _flush_output)
    cuda.synchronize()


def cleanup_cache_flush():
    """Free cache flush arrays"""
    global _flush_garbage, _flush_output
    if _flush_garbage is not None:
        del _flush_garbage, _flush_output
        _flush_garbage = None
        _flush_output = None


from PIL import Image

# ============================================
# IMAGE LOADING
# ============================================

def load_images(data_dir, max_images=500):
    """Load images from directory."""
    data_path = Path(data_dir)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(data_path.glob(ext))
    
    image_files = sorted(image_files)[:max_images]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loading {len(image_files)} images...")
    
    all_pixels = []
    widths = []
    heights = []
    offsets = []
    sizes = []
    
    current_offset = 0
    
    for img_path in image_files:
        with Image.open(img_path) as img:
            img = img.convert('L')
            w, h = img.size
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0
            
            all_pixels.append(pixels)
            widths.append(w)
            heights.append(h)
            offsets.append(current_offset)
            sizes.append(w * h)
            current_offset += w * h
    
    images_flat = np.concatenate(all_pixels)
    
    return (images_flat, 
            np.array(offsets, dtype=np.int64),
            np.array(sizes, dtype=np.int64),
            np.array(widths, dtype=np.int32),
            np.array(heights, dtype=np.int32))


def add_large_image(images_flat, offsets, sizes, widths, heights, size=(3840, 2160)):
    """Add large synthetic image for imbalance."""
    w, h = size
    large_pixels = np.random.rand(w * h).astype(np.float32)
    
    new_flat = np.concatenate([images_flat, large_pixels])
    new_offsets = np.concatenate([offsets, [len(images_flat)]])
    new_sizes = np.concatenate([sizes, [w * h]])
    new_widths = np.concatenate([widths, [w]])
    new_heights = np.concatenate([heights, [h]])
    
    return new_flat, new_offsets, new_sizes, new_widths, new_heights


# ============================================
# SETUP FUNCTIONS
# ============================================

@njit(cache=True)
def setup_baseline_a(offsets, sizes, total_pixels):
    """
    Baseline A: Build pixel-to-image mapping.
    Even optimized, filling 2GB is physically slow.
    """
    pixel_to_image = np.empty(total_pixels, dtype=np.int32)
    n_images = len(sizes)
    
    for img_id in range(n_images):
        start = offsets[img_id]
        size = sizes[img_id]
        for i in range(start, start + size):
            pixel_to_image[i] = img_id
            
    return pixel_to_image


def setup_baseline_b(offsets):
    """
    Baseline B: Binary search needs only offsets.
    O(1) setup, O(M) memory (already have offsets).
    """
    return offsets.astype(np.int64)


@njit(cache=True)
def setup_baseline_c(sizes, threshold=65536):
    """
    Baseline C: Build block metadata.
    Pre-allocate arrays instead of list.append for speed.
    """
    n_images = sizes.shape[0]
    
    total_blocks = 0
    for i in range(n_images):
        size = sizes[i]
        if size <= threshold:
            total_blocks += 1
        else:
            blocks = (size + threshold - 1) // threshold
            total_blocks += blocks
            
    block_to_image = np.empty(total_blocks, dtype=np.int32)
    block_start = np.empty(total_blocks, dtype=np.int64)
    block_end = np.empty(total_blocks, dtype=np.int64)
    
    current_block = 0
    for img_id in range(n_images):
        size = sizes[img_id]
        
        if size <= threshold:
            block_to_image[current_block] = img_id
            block_start[current_block] = 0
            block_end[current_block] = size
            current_block += 1
        else:
            num_blocks_this_img = (size + threshold - 1) // threshold
            pixels_per_block = (size + num_blocks_this_img - 1) // num_blocks_this_img
            
            for b in range(num_blocks_this_img):
                start = b * pixels_per_block
                end = start + pixels_per_block
                if end > size:
                    end = size
                
                block_to_image[current_block] = img_id
                block_start[current_block] = start
                block_end[current_block] = end
                current_block += 1
                
    return block_to_image, block_start, block_end


# ============================================
# KERNELS - LIGHT (3x3 blur, 1 iteration)
# ============================================

@cuda.jit
def kernel_baseline_a(images_flat, offsets, widths, heights,
                      pixel_to_image, output):
    """Baseline A: O(1) lookup - light kernel"""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        img_id = pixel_to_image[pixel_idx]
        
        offset = offsets[img_id]
        w = widths[img_id]
        h = heights[img_id]
        
        local_idx = pixel_idx - offset
        y = local_idx // w
        x = local_idx % w
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[pixel_idx] = images_flat[pixel_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[pixel_idx] = total / 9.0


@cuda.jit
def kernel_baseline_a_heavy(images_flat, offsets, widths, heights,
                            pixel_to_image, output):
    """Baseline A: O(1) lookup - heavy kernel (10x iterations)"""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        img_id = pixel_to_image[pixel_idx]
        
        offset = offsets[img_id]
        w = widths[img_id]
        h = heights[img_id]
        
        local_idx = pixel_idx - offset
        y = local_idx // w
        x = local_idx % w
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[pixel_idx] = images_flat[pixel_idx]
        else:
            result = 0.0
            for iteration in range(10):
                total = 0.0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        neighbor_idx = offset + (y + dy) * w + (x + dx)
                        total += images_flat[neighbor_idx]
                result += total / 9.0
            output[pixel_idx] = result / 10.0


@cuda.jit(device=True)
def binary_search_image(offsets, num_images, pixel_idx):
    """Binary search to find which image a pixel belongs to."""
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
def kernel_baseline_b(images_flat, offsets, sizes, widths, heights,
                      num_images, output):
    """Baseline B: Binary search - light kernel"""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        img_id = binary_search_image(offsets, num_images, pixel_idx)
        
        offset = offsets[img_id]
        w = widths[img_id]
        h = heights[img_id]
        
        local_idx = pixel_idx - offset
        y = local_idx // w
        x = local_idx % w
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[pixel_idx] = images_flat[pixel_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[pixel_idx] = total / 9.0


@cuda.jit
def kernel_baseline_b_heavy(images_flat, offsets, sizes, widths, heights,
                            num_images, output):
    """Baseline B: Binary search - heavy kernel (10x iterations)"""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        img_id = binary_search_image(offsets, num_images, pixel_idx)
        
        offset = offsets[img_id]
        w = widths[img_id]
        h = heights[img_id]
        
        local_idx = pixel_idx - offset
        y = local_idx // w
        x = local_idx % w
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[pixel_idx] = images_flat[pixel_idx]
        else:
            result = 0.0
            for iteration in range(10):
                total = 0.0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        neighbor_idx = offset + (y + dy) * w + (x + dx)
                        total += images_flat[neighbor_idx]
                result += total / 9.0
            output[pixel_idx] = result / 10.0


@cuda.jit
def kernel_baseline_c(images_flat, offsets, widths, heights,
                      block_to_image, block_start, block_end, output):
    """Baseline C: Block metadata - light kernel"""
    block_id = cuda.blockIdx.x
    
    if block_id >= block_to_image.shape[0]:
        return
    
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    offset = offsets[img_id]
    w = widths[img_id]
    h = heights[img_id]
    
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[global_idx] = images_flat[global_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[global_idx] = total / 9.0


@cuda.jit
def kernel_baseline_c_heavy(images_flat, offsets, widths, heights,
                            block_to_image, block_start, block_end, output):
    """Baseline C: Block metadata - heavy kernel (10x iterations)"""
    block_id = cuda.blockIdx.x
    
    if block_id >= block_to_image.shape[0]:
        return
    
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    offset = offsets[img_id]
    w = widths[img_id]
    h = heights[img_id]
    
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[global_idx] = images_flat[global_idx]
        else:
            result = 0.0
            for iteration in range(10):
                total = 0.0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        neighbor_idx = offset + (y + dy) * w + (x + dx)
                        total += images_flat[neighbor_idx]
                result += total / 9.0
            output[global_idx] = result / 10.0


# ============================================
# STATISTICS
# ============================================

def robust_stats(times_ms):
    """Compute median and IQR."""
    times = np.array(times_ms)
    median = np.median(times)
    q1 = np.percentile(times, 25)
    q3 = np.percentile(times, 75)
    iqr = q3 - q1
    return median, iqr, q1, q3


# ============================================
# BENCHMARK
# ============================================

def run_triple_baseline(images_flat, offsets, sizes, widths, heights, name,
                        threshold=65536, warmup=3, runs=15, flush_cache=False, heavy=False):
    """Compare three baselines."""
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*70}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    
    print(f"\nDataset: {num_images} images, {total_pixels:,} pixels")
    
    threads_per_block = 256
    grid_blocks = 256
    
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    
    results = {}
    
    # Baseline A
    print(f"\n[A: O(1) Lookup Table]")
    
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        pixel_to_image = setup_baseline_a(offsets, sizes, total_pixels)
        setup_times.append((time.perf_counter() - start) * 1000)
    
    a_setup_med, a_setup_iqr, _, _ = robust_stats(setup_times)
    a_memory_mb = pixel_to_image.nbytes / (1024 * 1024)
    print(f"  Setup: {a_setup_med:.2f} ms, Memory: {a_memory_mb:.2f} MB")
    
    h2d_times = []
    for _ in range(runs):
        start = time.perf_counter()
        d_pixel_to_image = cuda.to_device(pixel_to_image)
        cuda.synchronize()
        h2d_times.append((time.perf_counter() - start) * 1000)
    
    a_h2d_med, _, _, _ = robust_stats(h2d_times)
    print(f"  H2D: {a_h2d_med:.2f} ms")
    
    kernel_a = kernel_baseline_a_heavy if heavy else kernel_baseline_a
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        kernel_a[grid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights, d_pixel_to_image, d_output)
    cuda.synchronize()
    
    kernel_times = []
    for _ in range(runs):
        if flush_cache:
            flush_l2_cache()
        start = time.perf_counter()
        kernel_a[grid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights, d_pixel_to_image, d_output)
        cuda.synchronize()
        kernel_times.append((time.perf_counter() - start) * 1000)
    
    a_kernel_med, _, _, _ = robust_stats(kernel_times)
    print(f"  Kernel: {a_kernel_med:.2f} ms")
    
    d2h_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = d_output.copy_to_host()
        cuda.synchronize()
        d2h_times.append((time.perf_counter() - start) * 1000)
    
    a_d2h_med, _, _, _ = robust_stats(d2h_times)
    print(f"  D2H: {a_d2h_med:.2f} ms")
    
    a_total = a_setup_med + a_h2d_med + a_kernel_med + a_d2h_med
    print(f"  Total: {a_total:.2f} ms")
    
    results['A'] = {
        'setup_ms': a_setup_med, 'h2d_ms': a_h2d_med,
        'memory_mb': a_memory_mb, 'kernel_ms': a_kernel_med,
        'd2h_ms': a_d2h_med, 'total_ms': a_total
    }
    
    del d_pixel_to_image
    
    # Baseline B
    print(f"\n[B: Binary Search]")
    
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        search_offsets = setup_baseline_b(offsets)
        setup_times.append((time.perf_counter() - start) * 1000)
    
    b_setup_med, _, _, _ = robust_stats(setup_times)
    b_memory_mb = search_offsets.nbytes / (1024 * 1024)
    print(f"  Setup: {b_setup_med:.4f} ms, Memory: {b_memory_mb:.4f} MB")
    
    b_h2d_med = 0.01
    print(f"  H2D: ~0 ms (offsets already on GPU)")
    
    kernel_b = kernel_baseline_b_heavy if heavy else kernel_baseline_b
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        kernel_b[grid_blocks, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, num_images, d_output)
    cuda.synchronize()
    
    kernel_times = []
    for _ in range(runs):
        if flush_cache:
            flush_l2_cache()
        start = time.perf_counter()
        kernel_b[grid_blocks, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, num_images, d_output)
        cuda.synchronize()
        kernel_times.append((time.perf_counter() - start) * 1000)
    
    b_kernel_med, _, _, _ = robust_stats(kernel_times)
    print(f"  Kernel: {b_kernel_med:.2f} ms")
    
    b_d2h_med = a_d2h_med
    print(f"  D2H: {b_d2h_med:.2f} ms")
    
    b_total = b_setup_med + b_h2d_med + b_kernel_med + b_d2h_med
    print(f"  Total: {b_total:.2f} ms")
    
    results['B'] = {
        'setup_ms': b_setup_med, 'h2d_ms': b_h2d_med,
        'memory_mb': b_memory_mb, 'kernel_ms': b_kernel_med,
        'd2h_ms': b_d2h_med, 'total_ms': b_total
    }
    
    # Baseline C
    print(f"\n[C: Block Metadata]")
    
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        block_to_image, block_start, block_end = setup_baseline_c(sizes, threshold)
        setup_times.append((time.perf_counter() - start) * 1000)
    
    c_setup_med, _, _, _ = robust_stats(setup_times)
    c_memory_mb = (block_to_image.nbytes + block_start.nbytes + block_end.nbytes) / (1024 * 1024)
    num_blocks = len(block_to_image)
    print(f"  Setup: {c_setup_med:.3f} ms, Memory: {c_memory_mb:.4f} MB ({num_blocks} blocks)")
    
    h2d_times = []
    for _ in range(runs):
        start = time.perf_counter()
        d_block_to_image = cuda.to_device(block_to_image)
        d_block_start = cuda.to_device(block_start)
        d_block_end = cuda.to_device(block_end)
        cuda.synchronize()
        h2d_times.append((time.perf_counter() - start) * 1000)
    
    c_h2d_med, _, _, _ = robust_stats(h2d_times)
    print(f"  H2D: {c_h2d_med:.3f} ms")
    
    kernel_c = kernel_baseline_c_heavy if heavy else kernel_baseline_c
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        kernel_c[num_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
    cuda.synchronize()
    
    kernel_times = []
    for _ in range(runs):
        if flush_cache:
            flush_l2_cache()
        start = time.perf_counter()
        kernel_c[num_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
        cuda.synchronize()
        kernel_times.append((time.perf_counter() - start) * 1000)
    
    c_kernel_med, _, _, _ = robust_stats(kernel_times)
    print(f"  Kernel: {c_kernel_med:.2f} ms")
    
    c_d2h_med = a_d2h_med
    print(f"  D2H: {c_d2h_med:.2f} ms")
    
    c_total = c_setup_med + c_h2d_med + c_kernel_med + c_d2h_med
    print(f"  Total: {c_total:.2f} ms")
    
    results['C'] = {
        'setup_ms': c_setup_med, 'h2d_ms': c_h2d_med,
        'memory_mb': c_memory_mb, 'kernel_ms': c_kernel_med,
        'd2h_ms': c_d2h_med, 'total_ms': c_total
    }
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Approach':<15} {'Memory':>10} {'Kernel':>10} {'Total':>10}")
    print(f"{'-'*45}")
    print(f"{'A (Table)':<15} {results['A']['memory_mb']:>9.2f}MB {results['A']['kernel_ms']:>9.2f}ms {results['A']['total_ms']:>9.2f}ms")
    print(f"{'B (Search)':<15} {results['B']['memory_mb']:>9.4f}MB {results['B']['kernel_ms']:>9.2f}ms {results['B']['total_ms']:>9.2f}ms")
    print(f"{'C (Block)':<15} {results['C']['memory_mb']:>9.4f}MB {results['C']['kernel_ms']:>9.2f}ms {results['C']['total_ms']:>9.2f}ms")
    
    c_vs_a = results['A']['total_ms'] / results['C']['total_ms']
    c_vs_b = results['B']['total_ms'] / results['C']['total_ms']
    
    print(f"\nC vs A: {c_vs_a:.2f}x faster, {results['A']['memory_mb']/results['C']['memory_mb']:.0f}x less memory")
    if 0.95 <= c_vs_b <= 1.05:
        print(f"C vs B: ~same speed ({abs(1-c_vs_b)*100:.1f}% diff)")
    elif c_vs_b > 1:
        print(f"C vs B: {c_vs_b:.2f}x faster")
    else:
        print(f"C vs B: B is {1/c_vs_b:.2f}x faster")
    
    return results


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Triple Baseline Benchmark')
    parser.add_argument('--flush-cache', action='store_true',
                        help='Flush L2 cache before each kernel')
    parser.add_argument('--images', type=int, default=10000,
                        help='Number of images (default: 10000)')
    parser.add_argument('--heavy', action='store_true',
                        help='Use heavy kernel (10x iterations)')
    parser.add_argument('--small', action='store_true',
                        help='Use small images (3K-8K pixels)')
    parser.add_argument('--tiny', action='store_true',
                        help='Use tiny images (300-800 pixels)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRIPLE BASELINE BENCHMARK")
    print("=" * 70)
    print()
    print("Baselines:")
    print("  A: O(1) lookup table (O(N) memory)")
    print("  B: Binary search (O(M) memory, O(log M) kernel)")
    print("  C: Block metadata (O(B) memory, O(1) kernel)")
    
    if args.heavy:
        print("\nHEAVY MODE: 10x iteration kernels")
    
    if args.flush_cache:
        print("\nCACHE FLUSH: L2 cleared before each kernel")
        init_cache_flush(100)
    
    np.random.seed(42)
    num_images = args.images
    
    if args.tiny:
        min_pixels, max_pixels = 300, 800
        add_8k = False
    elif args.small:
        min_pixels, max_pixels = 3000, 8000
        add_8k = False
    else:
        min_pixels, max_pixels = 30000, 80000
        add_8k = True
    
    sizes = np.random.randint(min_pixels, max_pixels, num_images).astype(np.int64)
    
    if add_8k:
        sizes = np.append(sizes, [7680 * 4320] * 3)
    
    offsets = np.zeros(len(sizes), dtype=np.int64)
    offsets[1:] = np.cumsum(sizes[:-1])
    total = int(np.sum(sizes))
    
    print(f"\nDataset: {len(sizes):,} images, {total:,} pixels")
    print(f"Binary search depth: log2({len(sizes)}) = {int(np.log2(len(sizes)))}")
    
    images_flat = np.random.rand(total).astype(np.float32)
    widths = np.sqrt(sizes).astype(np.int32)
    widths = np.maximum(widths, 1)
    heights = (sizes // widths).astype(np.int32)
    heights = np.maximum(heights, 1)
    
    run_triple_baseline(images_flat, offsets, sizes, widths, heights,
                        "Synthetic", flush_cache=args.flush_cache, heavy=args.heavy)
    
    if args.flush_cache:
        cleanup_cache_flush()
    
    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
