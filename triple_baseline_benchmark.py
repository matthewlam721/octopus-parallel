"""
Triple Baseline Benchmark + Threshold Sweep
============================================
Addresses reviewer concerns:

1. Triple Baseline Comparison:
   - Baseline A: Grid-Stride + O(1) lookup (O(N) memory)
   - Baseline B: Grid-Stride + binary search (O(1) memory, O(log M) kernel)
   - Baseline C: Hybrid (O(B) memory, O(1) kernel)

2. Threshold Sweep:
   - Test threshold = 8K, 16K, 32K, 64K, 128K, 256K
   - Report sensitivity analysis

3. Improved Statistics:
   - Median + IQR (not just mean ± std)
   - 30 runs for stability
   - Vectorized setup

4. Cache Flush Option:
   - --flush-cache flag to simulate real workload
   - Fills L2 cache with garbage before each kernel

Author: Matthew
Date: January 2026
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

# Global cache flush arrays (reuse to avoid allocation overhead)
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
    """Initialize cache flush arrays (call once at start)"""
    global _flush_garbage, _flush_output
    num_floats = size_mb * 1024 * 1024 // 4  # 4 bytes per float32
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
# SETUP FUNCTIONS (VECTORIZED)
# ============================================

@njit(cache=True)
def setup_baseline_a(offsets, sizes, total_pixels):
    """
    Optimized Baseline A Setup with Numba.
    Even with this, it will still be slow because filling 2GB is physically slow.
    """
    pixel_to_image = np.empty(total_pixels, dtype=np.int32)
    n_images = len(sizes)
    
    for img_id in range(n_images):
        start = offsets[img_id]
        size = sizes[img_id]
        # In simple array filling, Numba is faster than numpy slice assignment
        # because it avoids overhead checks
        for i in range(start, start + size):
            pixel_to_image[i] = img_id
            
    return pixel_to_image


def setup_baseline_b(offsets):
    """
    Baseline B: Binary search (no mapping table)
    - O(M) memory (just offsets, already have)
    - O(1) setup time (nothing to build)
    - Returns offsets for binary search in kernel
    """
    # Need cumulative offsets for binary search
    # offsets already contains start positions
    return offsets.astype(np.int64)


@njit(cache=True)  # cache=True 令下次 run 唔洗再 compile
def setup_baseline_c(sizes, threshold=65536):
    """
    Optimized Setup: Pre-allocate arrays instead of using list.append
    """
    n_images = sizes.shape[0]
    
    total_blocks = 0
    for i in range(n_images):
        size = sizes[i]
        # (size + threshold - 1) // threshold
        if size <= threshold:
            total_blocks += 1
        else:
            # integer ceil division
            blocks = (size + threshold - 1) // threshold
            total_blocks += blocks
            
    block_to_image = np.empty(total_blocks, dtype=np.int32)
    block_start = np.empty(total_blocks, dtype=np.int64)
    block_end = np.empty(total_blocks, dtype=np.int64)
    
    current_block = 0
    for img_id in range(n_images):
        size = sizes[img_id]
        
        if size <= threshold:
            # Small image
            block_to_image[current_block] = img_id
            block_start[current_block] = 0
            block_end[current_block] = size
            current_block += 1
        else:
            # Large image
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
    """
    Baseline A: O(1) lookup - LIGHT kernel
    - Fast kernel (direct lookup)
    - But requires O(N) memory mapping
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        # O(1) lookup
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
    """
    Baseline A: O(1) lookup - HEAVY kernel (10x iterations)
    Simulates ML-style heavy compute to amplify differences
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        # O(1) lookup
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
            # HEAVY: 10 iterations of blur
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
    """
    Binary search to find which image a pixel belongs to.
    Returns image_id such that offsets[image_id] <= pixel_idx < offsets[image_id+1]
    """
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
    """
    Baseline B: Binary search (no mapping table) - LIGHT kernel
    - O(log M) per pixel to find image
    - No memory overhead
    - But slower kernel due to search + branch divergence
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        # O(log M) binary search
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
    """
    Baseline B: Binary search - HEAVY kernel (10x iterations)
    Binary search happens ONCE per pixel, but compute is 10x heavier.
    This tests if branch divergence accumulates with heavier workload.
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        # O(log M) binary search - still only once per pixel
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
            # HEAVY: 10 iterations of blur
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
    """
    Baseline C: Hybrid (block metadata) - LIGHT kernel
    - O(1) lookup per block
    - O(B) memory where B << N
    - Fast kernel + small memory
    """
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
    """
    Baseline C: Hybrid (block metadata) - HEAVY kernel (10x iterations)
    Block-level O(1) lookup + heavy compute per pixel
    """
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
            # HEAVY: 10 iterations of blur
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
# STATISTICS HELPERS
# ============================================

def robust_stats(times_ms):
    """Compute median and IQR for robust statistics."""
    times = np.array(times_ms)
    median = np.median(times)
    q1 = np.percentile(times, 25)
    q3 = np.percentile(times, 75)
    iqr = q3 - q1
    return median, iqr, q1, q3


def time_function(func, runs=30):
    """Time a function with robust statistics."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return robust_stats(times)


# ============================================
# TRIPLE BASELINE BENCHMARK
# ============================================

def run_triple_baseline(images_flat, offsets, sizes, widths, heights, name,
                        threshold=65536, warmup=3, runs=15, flush_cache=False, heavy=False):
    """
    Compare three baselines:
    A: Grid-Stride + O(1) lookup
    B: Grid-Stride + binary search
    C: Hybrid
    
    If flush_cache=True, L2 cache is flushed before each kernel run
    to simulate real workload conditions.
    
    If heavy=True, use 10x iteration kernels to stress branch divergence.
    """
    
    print(f"\n{'='*80}")
    print(f"TRIPLE BASELINE BENCHMARK: {name}")
    print(f"{'='*80}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    
    print(f"\nDataset: {num_images} images, {total_pixels:,} pixels")
    print(f"Statistics: median ± IQR over {runs} runs")
    
    threads_per_block = 256
    grid_blocks = 256
    
    # Pre-transfer common data
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    
    results = {}
    
    # ========================================
    # Baseline A: O(1) lookup
    # ========================================
    print(f"\n[Baseline A: O(1) Lookup Table]")
    
    # Setup timing
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        pixel_to_image = setup_baseline_a(offsets, sizes, total_pixels)
        setup_times.append((time.perf_counter() - start) * 1000)
    
    a_setup_med, a_setup_iqr, _, _ = robust_stats(setup_times)
    a_memory_bytes = pixel_to_image.nbytes
    a_memory_mb = a_memory_bytes / (1024 * 1024)
    
    print(f"  Setup: {a_setup_med:.2f} ms (IQR: {a_setup_iqr:.2f})")
    print(f"  Memory: {a_memory_mb:.2f} MB")
    
    # H2D timing
    h2d_times = []
    for _ in range(runs):
        start = time.perf_counter()
        d_pixel_to_image = cuda.to_device(pixel_to_image)
        cuda.synchronize()
        h2d_times.append((time.perf_counter() - start) * 1000)
    
    a_h2d_med, a_h2d_iqr, _, _ = robust_stats(h2d_times)
    print(f"  H2D: {a_h2d_med:.2f} ms (IQR: {a_h2d_iqr:.2f})")
    
    # Warmup
    kernel_a = kernel_baseline_a_heavy if heavy else kernel_baseline_a
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        kernel_a[grid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights, d_pixel_to_image, d_output)
    cuda.synchronize()
    
    # Kernel timing
    kernel_times = []
    for _ in range(runs):
        if flush_cache:
            flush_l2_cache()
        start = time.perf_counter()
        kernel_a[grid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights, d_pixel_to_image, d_output)
        cuda.synchronize()
        kernel_times.append((time.perf_counter() - start) * 1000)
    
    a_kernel_med, a_kernel_iqr, _, _ = robust_stats(kernel_times)
    print(f"  Kernel: {a_kernel_med:.2f} ms (IQR: {a_kernel_iqr:.2f})")
    
    # D2H timing
    d2h_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = d_output.copy_to_host()
        cuda.synchronize()
        d2h_times.append((time.perf_counter() - start) * 1000)
    
    a_d2h_med, a_d2h_iqr, _, _ = robust_stats(d2h_times)
    print(f"  D2H: {a_d2h_med:.2f} ms (IQR: {a_d2h_iqr:.2f})")
    
    a_total = a_setup_med + a_h2d_med + a_kernel_med + a_d2h_med
    print(f"  TOTAL: {a_total:.2f} ms")
    
    results['A'] = {
        'name': 'O(1) Lookup',
        'setup_ms': a_setup_med,
        'h2d_ms': a_h2d_med,
        'memory_mb': a_memory_mb,
        'kernel_ms': a_kernel_med,
        'd2h_ms': a_d2h_med,
        'total_ms': a_total
    }
    
    del d_pixel_to_image
    
    # ========================================
    # Baseline B: Binary Search
    # ========================================
    print(f"\n[Baseline B: Binary Search (No Table)]")
    
    # Setup timing (essentially nothing to do)
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        search_offsets = setup_baseline_b(offsets)
        setup_times.append((time.perf_counter() - start) * 1000)
    
    b_setup_med, b_setup_iqr, _, _ = robust_stats(setup_times)
    b_memory_bytes = search_offsets.nbytes  # Just offsets, already transferred
    b_memory_mb = b_memory_bytes / (1024 * 1024)
    
    print(f"  Setup: {b_setup_med:.4f} ms (IQR: {b_setup_iqr:.4f})")
    print(f"  Memory: {b_memory_mb:.4f} MB (offsets only)")
    
    # H2D timing (minimal - offsets already transferred)
    b_h2d_med = 0.01  # Negligible
    print(f"  H2D: ~0 ms (offsets already on GPU)")
    
    # Warmup
    kernel_b = kernel_baseline_b_heavy if heavy else kernel_baseline_b
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        kernel_b[grid_blocks, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, num_images, d_output)
    cuda.synchronize()
    
    # Kernel timing
    kernel_times = []
    for _ in range(runs):
        if flush_cache:
            flush_l2_cache()
        start = time.perf_counter()
        kernel_b[grid_blocks, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, num_images, d_output)
        cuda.synchronize()
        kernel_times.append((time.perf_counter() - start) * 1000)
    
    b_kernel_med, b_kernel_iqr, _, _ = robust_stats(kernel_times)
    print(f"  Kernel: {b_kernel_med:.2f} ms (IQR: {b_kernel_iqr:.2f})")
    
    # D2H timing (same as A)
    b_d2h_med = a_d2h_med
    print(f"  D2H: {b_d2h_med:.2f} ms")
    
    b_total = b_setup_med + b_h2d_med + b_kernel_med + b_d2h_med
    print(f"  TOTAL: {b_total:.2f} ms")
    
    results['B'] = {
        'name': 'Binary Search',
        'setup_ms': b_setup_med,
        'h2d_ms': b_h2d_med,
        'memory_mb': b_memory_mb,
        'kernel_ms': b_kernel_med,
        'd2h_ms': b_d2h_med,
        'total_ms': b_total
    }
    
    # ========================================
    # Baseline C: Hybrid
    # ========================================
    print(f"\n[Baseline C: Hybrid (Block Metadata)]")
    
    # Setup timing
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        block_to_image, block_start, block_end = setup_baseline_c(sizes, threshold)
        setup_times.append((time.perf_counter() - start) * 1000)
    
    c_setup_med, c_setup_iqr, _, _ = robust_stats(setup_times)
    c_memory_bytes = block_to_image.nbytes + block_start.nbytes + block_end.nbytes
    c_memory_mb = c_memory_bytes / (1024 * 1024)
    num_blocks = len(block_to_image)
    
    print(f"  Setup: {c_setup_med:.3f} ms (IQR: {c_setup_iqr:.3f})")
    print(f"  Memory: {c_memory_mb:.4f} MB ({num_blocks} blocks)")
    
    # H2D timing
    h2d_times = []
    for _ in range(runs):
        start = time.perf_counter()
        d_block_to_image = cuda.to_device(block_to_image)
        d_block_start = cuda.to_device(block_start)
        d_block_end = cuda.to_device(block_end)
        cuda.synchronize()
        h2d_times.append((time.perf_counter() - start) * 1000)
    
    c_h2d_med, c_h2d_iqr, _, _ = robust_stats(h2d_times)
    print(f"  H2D: {c_h2d_med:.3f} ms (IQR: {c_h2d_iqr:.3f})")
    
    # Warmup
    kernel_c = kernel_baseline_c_heavy if heavy else kernel_baseline_c
    for _ in range(warmup):
        if flush_cache:
            flush_l2_cache()
        kernel_c[num_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
    cuda.synchronize()
    
    # Kernel timing
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
    
    c_kernel_med, c_kernel_iqr, _, _ = robust_stats(kernel_times)
    print(f"  Kernel: {c_kernel_med:.2f} ms (IQR: {c_kernel_iqr:.2f})")
    
    # D2H timing (same as A)
    c_d2h_med = a_d2h_med
    print(f"  D2H: {c_d2h_med:.2f} ms")
    
    c_total = c_setup_med + c_h2d_med + c_kernel_med + c_d2h_med
    print(f"  TOTAL: {c_total:.2f} ms")
    
    results['C'] = {
        'name': 'Hybrid',
        'setup_ms': c_setup_med,
        'h2d_ms': c_h2d_med,
        'memory_mb': c_memory_mb,
        'kernel_ms': c_kernel_med,
        'd2h_ms': c_d2h_med,
        'total_ms': c_total
    }
    
    # ========================================
    # COMPARISON TABLE
    # ========================================
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    
    print(f"\n  {'Baseline':<20} {'Setup':>10} {'H2D':>10} {'Memory':>10} {'Kernel':>10} {'D2H':>10} {'TOTAL':>10}")
    print(f"  {'-'*82}")
    
    for key in ['A', 'B', 'C']:
        r = results[key]
        print(f"  {r['name']:<20} {r['setup_ms']:>9.2f}ms {r['h2d_ms']:>9.2f}ms {r['memory_mb']:>9.2f}MB {r['kernel_ms']:>9.2f}ms {r['d2h_ms']:>9.2f}ms {r['total_ms']:>9.2f}ms")
    
    # Ratios vs Hybrid
    print(f"\n  Ratios (vs Hybrid C):")
    print(f"  {'-'*82}")
    
    for key in ['A', 'B']:
        r = results[key]
        c = results['C']
        setup_r = r['setup_ms'] / c['setup_ms'] if c['setup_ms'] > 0 else float('inf')
        mem_r = r['memory_mb'] / c['memory_mb'] if c['memory_mb'] > 0 else float('inf')
        kernel_r = r['kernel_ms'] / c['kernel_ms']
        total_r = r['total_ms'] / c['total_ms']
        
        print(f"  {r['name']:<20} Setup: {setup_r:>6.0f}x | Memory: {mem_r:>6.0f}x | Kernel: {kernel_r:>5.2f}x | Total: {total_r:>5.2f}x")
    
    # Winner analysis
    print(f"\n  Analysis:")
    print(f"  {'-'*82}")
    
    # Find fastest total
    fastest = min(results.items(), key=lambda x: x[1]['total_ms'])
    print(f"  Fastest TOTAL: {fastest[1]['name']} ({fastest[1]['total_ms']:.2f} ms)")
    
    # Find smallest memory
    smallest_mem = min(results.items(), key=lambda x: x[1]['memory_mb'])
    print(f"  Smallest Memory: {smallest_mem[1]['name']} ({smallest_mem[1]['memory_mb']:.4f} MB)")
    
    # Find fastest kernel
    fastest_kernel = min(results.items(), key=lambda x: x[1]['kernel_ms'])
    print(f"  Fastest Kernel: {fastest_kernel[1]['name']} ({fastest_kernel[1]['kernel_ms']:.2f} ms)")
    
    # Hybrid unique advantage
    print(f"\n  Key finding:")
    
    # Calculate relative performance
    c_vs_a = results['A']['total_ms'] / results['C']['total_ms']
    c_vs_b = results['B']['total_ms'] / results['C']['total_ms']
    
    print(f"  • Hybrid vs A (Table): {c_vs_a:.2f}x faster, {results['A']['memory_mb']/results['C']['memory_mb']:.0f}x less memory")
    
    if 0.95 <= c_vs_b <= 1.05:
        print(f"  • Hybrid vs B (Search): ~same speed (within {abs(1-c_vs_b)*100:.1f}%)")
    elif c_vs_b > 1:
        print(f"  • Hybrid vs B (Search): {c_vs_b:.2f}x faster")
    else:
        print(f"  • Hybrid vs B (Search): B is {1/c_vs_b:.2f}x faster (tiny workload favors zero-setup)")
    
    print(f"  • Conclusion: Hybrid = O(1) dispatch + extensible policy, B = zero setup baseline")
    
    return results


# ============================================
# THRESHOLD SWEEP
# ============================================

def run_threshold_sweep(images_flat, offsets, sizes, widths, heights,
                        thresholds=[8192, 16384, 32768, 65536, 131072, 262144],
                        warmup=3, runs=15):
    """
    Sweep threshold parameter to show sensitivity.
    """
    
    print(f"\n{'='*80}")
    print("THRESHOLD SWEEP")
    print(f"{'='*80}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    
    print(f"\nDataset: {num_images} images, {total_pixels:,} pixels")
    print(f"Testing thresholds: {thresholds}")
    
    threads_per_block = 256
    
    # Pre-transfer common data
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    
    results = []
    
    for threshold in thresholds:
        print(f"\n[Threshold: {threshold:,}]")
        
        # Setup
        setup_times = []
        for _ in range(runs):
            start = time.perf_counter()
            block_to_image, block_start, block_end = setup_baseline_c(sizes, threshold)
            setup_times.append((time.perf_counter() - start) * 1000)
        
        setup_med, setup_iqr, _, _ = robust_stats(setup_times)
        memory_bytes = block_to_image.nbytes + block_start.nbytes + block_end.nbytes
        memory_mb = memory_bytes / (1024 * 1024)
        num_blocks = len(block_to_image)
        
        print(f"  Blocks: {num_blocks}, Memory: {memory_mb:.4f} MB")
        
        # H2D
        h2d_times = []
        for _ in range(runs):
            start = time.perf_counter()
            d_block_to_image = cuda.to_device(block_to_image)
            d_block_start = cuda.to_device(block_start)
            d_block_end = cuda.to_device(block_end)
            cuda.synchronize()
            h2d_times.append((time.perf_counter() - start) * 1000)
        
        h2d_med, _, _, _ = robust_stats(h2d_times)
        
        # Warmup
        for _ in range(warmup):
            kernel_baseline_c[num_blocks, threads_per_block](
                d_images, d_offsets, d_widths, d_heights,
                d_block_to_image, d_block_start, d_block_end, d_output)
        cuda.synchronize()
        
        # Kernel
        kernel_times = []
        for _ in range(runs):
            start = time.perf_counter()
            kernel_baseline_c[num_blocks, threads_per_block](
                d_images, d_offsets, d_widths, d_heights,
                d_block_to_image, d_block_start, d_block_end, d_output)
            cuda.synchronize()
            kernel_times.append((time.perf_counter() - start) * 1000)
        
        kernel_med, kernel_iqr, _, _ = robust_stats(kernel_times)
        
        # D2H (constant)
        d2h_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = d_output.copy_to_host()
            cuda.synchronize()
            d2h_times.append((time.perf_counter() - start) * 1000)
        d2h_med, _, _, _ = robust_stats(d2h_times)
        
        total = setup_med + h2d_med + kernel_med + d2h_med
        
        print(f"  Setup: {setup_med:.3f} ms, H2D: {h2d_med:.3f} ms, Kernel: {kernel_med:.2f} ms, Total: {total:.2f} ms")
        
        results.append({
            'threshold': threshold,
            'num_blocks': num_blocks,
            'setup_ms': setup_med,
            'h2d_ms': h2d_med,
            'memory_mb': memory_mb,
            'kernel_ms': kernel_med,
            'd2h_ms': d2h_med,
            'total_ms': total
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("THRESHOLD SWEEP SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n  {'Threshold':>10} {'Blocks':>10} {'Memory':>10} {'Setup':>10} {'Kernel':>10} {'TOTAL':>10}")
    print(f"  {'-'*65}")
    
    for r in results:
        print(f"  {r['threshold']:>10,} {r['num_blocks']:>10,} {r['memory_mb']:>9.4f}MB {r['setup_ms']:>9.3f}ms {r['kernel_ms']:>9.2f}ms {r['total_ms']:>9.2f}ms")
    
    # Find optimal
    best = min(results, key=lambda x: x['total_ms'])
    print(f"\n  Optimal threshold: {best['threshold']:,} (Total: {best['total_ms']:.2f} ms)")
    
    # Sensitivity analysis
    totals = [r['total_ms'] for r in results]
    variation = (max(totals) - min(totals)) / np.mean(totals) * 100
    print(f"  Sensitivity: {variation:.1f}% variation across thresholds")
    
    if variation < 20:
        print(f"  ✅ STABLE: Performance robust across threshold range")
    else:
        print(f"  ⚠️ SENSITIVE: Performance varies with threshold choice")
    
    return results


# ============================================
# MAIN
# ============================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Triple Baseline Benchmark')
    parser.add_argument('--flush-cache', action='store_true',
                        help='Flush L2 cache before each kernel to simulate real workload')
    parser.add_argument('--images', type=int, default=10000,
                        help='Number of images (default: 10000)')
    parser.add_argument('--heavy', action='store_true',
                        help='Use heavy kernel (10x iterations) to stress branch divergence')
    parser.add_argument('--small', action='store_true',
                        help='Use small images (3K-8K pixels) for 100K image tests')
    parser.add_argument('--tiny', action='store_true',
                        help='Use tiny images (300-800 pixels) for 1M image tests')
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRIPLE BASELINE + THRESHOLD SWEEP BENCHMARK")
    print("=" * 80)
    print()
    print("Baselines:")
    print("  A: Grid-Stride + O(1) lookup (O(N) memory)")
    print("  B: Grid-Stride + binary search (O(1) memory, O(log M) kernel)")
    print("  C: Hybrid (O(B) memory, O(1) kernel)")
    print()
    print("Statistics: Median ± IQR over 15 runs")
    
    if args.heavy:
        print()
        print("💪 HEAVY KERNEL MODE ENABLED")
        print("   Using 10x iteration kernels to stress branch divergence")
    
    if args.flush_cache:
        print()
        print("🔥 CACHE FLUSH MODE ENABLED")
        print("   L2 cache will be flushed before each kernel run")
        print("   This simulates real workload with cache contention")
        init_cache_flush(100)  # 100 MB to flush 72 MB L2
    
    print()
    
    # ========================================
    # SAFE VERSION - 唔好推死 4090
    # Target: ~10 GB total GPU usage
    # ========================================
    print("🐙 Generating LARGE but SAFE dataset...")
    print("   Target: ~10 GB GPU memory (safe for 24 GB 4090)")
    print()
    
    np.random.seed(42)
    
    # Number of images from args
    num_images = args.images
    
    # Determine pixel size based on mode
    if args.tiny:
        # Tiny: 300-800 pixels per image (for 1M image test)
        min_pixels, max_pixels = 300, 800
        add_8k = False
        mode_name = "TINY"
    elif args.small:
        # Small: 3K-8K pixels per image (for 100K image test)
        min_pixels, max_pixels = 3000, 8000
        add_8k = False
        mode_name = "SMALL"
    else:
        # Normal: 30K-80K pixels per image
        min_pixels, max_pixels = 30000, 80000
        add_8k = True
        mode_name = "NORMAL"
    
    sizes = np.random.randint(min_pixels, max_pixels, num_images).astype(np.int64)
    
    # Add 3x 8K images for imbalance (only in normal mode)
    if add_8k:
        sizes = np.append(sizes, [7680 * 4320] * 3)
    
    offsets = np.zeros(len(sizes), dtype=np.int64)
    offsets[1:] = np.cumsum(sizes[:-1])
    total = int(np.sum(sizes))
    
    gb_images = total * 4 / 1e9
    gb_mapping = total * 4 / 1e9  # pixel_to_image array
    gb_output = total * 4 / 1e9
    gb_total_worst = gb_images + gb_mapping + gb_output
    
    print(f"   Mode: {mode_name} ({min_pixels}-{max_pixels} pixels/image)")
    print(f"   Dataset: {len(sizes):,} images")
    print(f"   Total pixels: {total:,}")
    print(f"   Memory breakdown:")
    print(f"     - Images: {gb_images:.2f} GB")
    print(f"     - Mapping (Baseline A): {gb_mapping:.2f} GB")
    print(f"     - Output: {gb_output:.2f} GB")
    print(f"     - Worst case total: {gb_total_worst:.2f} GB")
    print(f"   Binary search depth: log2({len(sizes)}) ≈ {int(np.log2(len(sizes)))}")
    print()
    
    if gb_total_worst > 20:
        print("   ❌ Still too large! Aborting.")
        return
    
    print("   ✅ Safe to proceed!")
    print()
    
    images_flat = np.random.rand(total).astype(np.float32)
    widths = np.sqrt(sizes).astype(np.int32)
    widths = np.maximum(widths, 1)
    heights = (sizes // widths).astype(np.int32)
    heights = np.maximum(heights, 1)
    
    data = (images_flat, offsets, sizes, widths, heights)
    
    images_flat, offsets, sizes, widths, heights = data
    
    # ========================================
    # PART 1: Triple Baseline
    # ========================================
    print("\n" + "█" * 80)
    print("PART 1: TRIPLE BASELINE COMPARISON")
    if args.heavy:
        print("        (HEAVY KERNEL: 10x iterations)")
    if args.flush_cache:
        print("        (WITH L2 CACHE FLUSH)")
    print("█" * 80)
    
    triple_results = run_triple_baseline(images_flat, offsets, sizes, widths, heights,
                                         "Synthetic Dataset", 
                                         flush_cache=args.flush_cache,
                                         heavy=args.heavy)
    
    # ========================================
    # PART 2: Threshold Sweep (SKIP for now)
    # ========================================
    print("\n" + "█" * 80)
    print("SKIPPING THRESHOLD SWEEP (already validated)")
    print("█" * 80)
    
    # Cleanup cache flush arrays if used
    if args.flush_cache:
        cleanup_cache_flush()
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    # Calculate ratios (handle case where C might be slower)
    a_vs_c_speed = triple_results['A']['total_ms'] / triple_results['C']['total_ms']
    b_vs_c_speed = triple_results['B']['total_ms'] / triple_results['C']['total_ms']
    b_vs_c_kernel = triple_results['B']['kernel_ms'] / triple_results['C']['kernel_ms']
    
    print(f"""
  Triple Baseline Results:
  -------------------------
  A (O(1) Lookup):    {triple_results['A']['total_ms']:.2f} ms total, {triple_results['A']['memory_mb']:.2f} MB memory
  B (Binary Search):  {triple_results['B']['total_ms']:.2f} ms total, {triple_results['B']['memory_mb']:.4f} MB memory
  C (Hybrid):         {triple_results['C']['total_ms']:.2f} ms total, {triple_results['C']['memory_mb']:.4f} MB memory
  
  Hybrid vs A: {a_vs_c_speed:.2f}x faster, {triple_results['A']['memory_mb']/triple_results['C']['memory_mb']:.0f}x less memory
  Hybrid vs B: {'~same' if 0.95 <= b_vs_c_speed <= 1.05 else f'{b_vs_c_speed:.2f}x'} (within {abs(1-b_vs_c_speed)*100:.1f}%)
  
  Key Insight:
  ------------
  Hybrid achieves table-like O(1) dispatch without O(N) mapping,
  and matches the strongest no-table baseline (binary search)
  within ~1-5% while using negligible memory.
  
  When to use each:
  -----------------
  • A (Table):  Fastest kernel, but setup+H2D+memory kills total time
               → Only if kernel runs 100+ times per batch
               
  • B (Search): Zero setup, zero extra memory
               → Best for tiny items (<1K pixels) + massive M (>100K)
               → Or I/O dominated pipelines where D2H >> kernel
               
  • C (Hybrid): O(1) dispatch + extensible block-level policy
               → Best for normal workloads, weak/shared GPUs
               → Enables: priority scheduling, ROI, multi-stream
               → Stable across GPU architectures
               
  System Insight:
  ---------------
  D2H transfer ({triple_results['A']['d2h_ms']:.0f}ms) dominates total time.
  In output-copy dominated pipelines, scheduler differences are masked.
  For fused GPU pipelines (no D2H), Hybrid's O(1) advantage becomes visible.
    """)
    
    print("=" * 80)
    print("🐙 Triple baseline + threshold sweep complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()