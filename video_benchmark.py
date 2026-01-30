"""
Video Frame GPU Benchmark
=========================
Benchmark three approaches on real video frames from ffmpeg extraction.

Run video_frame_extract.py first:
    python video_frame_extract.py --all --prepare

Then run this:
    python video_benchmark.py
"""

from numba import njit, cuda
import numpy as np
import time
from pathlib import Path

# ============================================
# SETUP FUNCTIONS (same as triple_baseline)
# ============================================

@njit(cache=True)
def setup_baseline_a(offsets, sizes, total_pixels):
    """Baseline A: Build pixel-to-image mapping."""
    pixel_to_image = np.empty(total_pixels, dtype=np.int32)
    n_images = len(sizes)
    
    for img_id in range(n_images):
        start = offsets[img_id]
        size = sizes[img_id]
        for i in range(start, start + size):
            pixel_to_image[i] = img_id
            
    return pixel_to_image


def setup_baseline_b(offsets):
    """Baseline B: Binary search needs only offsets."""
    return offsets.astype(np.int64)


@njit(cache=True)
def setup_baseline_c(sizes, threshold=65536):
    """Baseline C: Build block metadata."""
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
# KERNELS
# ============================================

@cuda.jit
def kernel_baseline_a(images_flat, offsets, widths, heights,
                      pixel_to_image, output):
    """Baseline A: O(1) lookup"""
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
    """Baseline B: Binary search"""
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
def kernel_baseline_c(images_flat, offsets, widths, heights,
                      block_to_image, block_start, block_end, output):
    """Baseline C: Block metadata"""
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

def run_benchmark(images_flat, offsets, sizes, widths, heights, 
                  warmup=3, runs=15):
    """Run triple baseline benchmark on video frames."""
    
    print("=" * 70)
    print("VIDEO FRAME BENCHMARK")
    print("=" * 70)
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    
    print(f"\nDataset: {num_images} frames, {total_pixels:,} pixels")
    print(f"Size range: {sizes.min():,} - {sizes.max():,} pixels")
    print(f"Imbalance: {sizes.max() / sizes.min():.2f}x")
    
    threads_per_block = 256
    grid_blocks = 256
    
    # Transfer to GPU
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    
    results = {}
    
    # ========== Baseline A ==========
    print(f"\n[A: Lookup Table]")
    
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        pixel_to_image = setup_baseline_a(offsets, sizes, total_pixels)
        setup_times.append((time.perf_counter() - start) * 1000)
    
    a_setup_med, _, _, _ = robust_stats(setup_times)
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
    
    for _ in range(warmup):
        kernel_baseline_a[grid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights, d_pixel_to_image, d_output)
    cuda.synchronize()
    
    kernel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        kernel_baseline_a[grid_blocks, threads_per_block](
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
    
    # ========== Baseline B ==========
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
    print(f"  H2D: ~0 ms")
    
    for _ in range(warmup):
        kernel_baseline_b[grid_blocks, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, num_images, d_output)
    cuda.synchronize()
    
    kernel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        kernel_baseline_b[grid_blocks, threads_per_block](
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
    
    # ========== Baseline C ==========
    print(f"\n[C: Block Metadata]")
    
    setup_times = []
    for _ in range(runs):
        start = time.perf_counter()
        block_to_image, block_start, block_end = setup_baseline_c(sizes)
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
    
    for _ in range(warmup):
        kernel_baseline_c[num_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
    cuda.synchronize()
    
    kernel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        kernel_baseline_c[num_blocks, threads_per_block](
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
    
    # ========== Summary ==========
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
        print(f"C vs B: ~same ({abs(1-c_vs_b)*100:.1f}% diff)")
    elif c_vs_b > 1:
        print(f"C vs B: {c_vs_b:.2f}x faster")
    else:
        print(f"C vs B: B is {1/c_vs_b:.2f}x faster")
    
    return results


def main():
    # Check if data file exists
    data_file = 'video_frames_data.npz'
    
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        print("Run this first:")
        print("  python video_frame_extract.py --all --prepare")
        return
    
    # Load data
    print("Loading video frame data...")
    data = np.load(data_file)
    
    images_flat = data['images_flat']
    offsets = data['offsets']
    sizes = data['sizes']
    widths = data['widths']
    heights = data['heights']
    
    print(f"Loaded {len(sizes)} frames, {len(images_flat):,} total pixels")
    
    # Run benchmark
    run_benchmark(images_flat, offsets, sizes, widths, heights)
    
    print("\nDone")


if __name__ == "__main__":
    main()