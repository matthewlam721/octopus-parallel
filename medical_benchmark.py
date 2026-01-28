"""
Medical Image Benchmark (Real Data)
====================================
Using real CT scan images from Kaggle Chest CT dataset.

Author: Matthew
Date: January 28, 2026
"""

from numba import cuda
import numpy as np
import time
from pathlib import Path
from PIL import Image
import os

# ============================================
# IMAGE LOADING
# ============================================

def load_medical_images(data_dir, max_images=None, grayscale=True):
    """
    Load real medical images from dataset.
    
    Args:
        data_dir: Path to Data folder
        max_images: Limit number of images (None = all)
        grayscale: Convert to grayscale (typical for medical)
    
    Returns:
        images_flat: Flattened pixel array
        offsets: Start offset for each image
        sizes: Number of pixels per image
        image_dims: List of (width, height) tuples
    """
    data_path = Path(data_dir)
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(data_path.rglob(ext))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Loading {len(image_files)} images...")
    
    # First pass: get sizes
    image_dims = []
    total_pixels = 0
    
    for img_path in image_files:
        with Image.open(img_path) as img:
            if grayscale:
                img = img.convert('L')  # Grayscale
            w, h = img.size
            image_dims.append((w, h))
            total_pixels += w * h
    
    print(f"Total pixels: {total_pixels:,}")
    
    # Second pass: load pixel data
    images_flat = np.zeros(total_pixels, dtype=np.float32)
    offsets = np.zeros(len(image_files), dtype=np.int64)
    sizes = np.zeros(len(image_files), dtype=np.int64)
    
    current_offset = 0
    for i, img_path in enumerate(image_files):
        with Image.open(img_path) as img:
            if grayscale:
                img = img.convert('L')
            
            # Normalize to 0-1 range
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0
            
            offsets[i] = current_offset
            sizes[i] = len(pixels)
            images_flat[current_offset:current_offset + len(pixels)] = pixels
            current_offset += len(pixels)
    
    print(f"Loaded successfully!")
    
    return images_flat, offsets, sizes, image_dims


# ============================================
# GPU KERNELS (same as before)
# ============================================

@cuda.jit
def naive_image_kernel(images_flat, image_offsets, image_sizes, output):
    """Naive: Each thread processes one entire image."""
    tid = cuda.grid(1)
    
    if tid < image_sizes.shape[0]:
        start = image_offsets[tid]
        num_pixels = image_sizes[tid]
        
        result = 0.0
        for i in range(num_pixels):
            pixel = images_flat[start + i]
            # Simulate medical image processing (e.g., contrast enhancement)
            result += pixel * 0.5 + 0.1
            result = result * 1.001
        
        output[tid] = result


@cuda.jit
def balanced_image_kernel(images_flat, work_start, work_end, output):
    """Balanced: Each thread processes equal number of pixels."""
    tid = cuda.grid(1)
    
    if tid < work_start.shape[0]:
        start = work_start[tid]
        end = work_end[tid]
        
        result = 0.0
        for i in range(start, end):
            pixel = images_flat[i]
            result += pixel * 0.5 + 0.1
            result = result * 1.001
        
        output[tid] = result


# ============================================
# BENCHMARK FUNCTIONS
# ============================================

def compute_balanced_work(total_pixels, num_threads):
    """Compute balanced work distribution."""
    pixels_per_thread = total_pixels // num_threads
    remainder = total_pixels % num_threads
    
    work_start = np.zeros(num_threads, dtype=np.int64)
    work_end = np.zeros(num_threads, dtype=np.int64)
    
    current = 0
    for tid in range(num_threads):
        work_start[tid] = current
        thread_work = pixels_per_thread + (1 if tid < remainder else 0)
        current += thread_work
        work_end[tid] = current
    
    return work_start, work_end


def run_medical_benchmark(images_flat, offsets, sizes, image_dims, 
                          num_threads, name, warmup=3, runs=30):
    """Run benchmark on real medical images."""
    
    print("\n" + "=" * 60)
    print(f"BENCHMARK: {name}")
    print("=" * 60)
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    max_image_pixels = max(sizes)
    min_image_pixels = min(sizes)
    
    print(f"\nDataset Statistics:")
    print(f"  Images: {num_images}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Min image: {min_image_pixels:,} pixels")
    print(f"  Max image: {max_image_pixels:,} pixels")
    print(f"  Avg image: {total_pixels // num_images:,} pixels")
    print(f"  Threads: {num_threads}")
    
    # Size distribution
    print(f"\nSize distribution (sample):")
    for i in range(min(5, len(image_dims))):
        print(f"  Image {i}: {image_dims[i][0]}x{image_dims[i][1]} = {sizes[i]:,} pixels")
    if len(image_dims) > 5:
        print(f"  ... and {len(image_dims) - 5} more")
    
    # Imbalance analysis
    avg_pixels = total_pixels / num_images
    imbalance = max_image_pixels / avg_pixels
    print(f"\nImbalance ratio: {imbalance:.2f}x")
    
    # Theoretical speedup
    theoretical = max_image_pixels / (total_pixels / num_threads)
    print(f"Theoretical max speedup: {theoretical:.2f}x")
    
    # Balanced work distribution
    work_start, work_end = compute_balanced_work(total_pixels, num_threads)
    
    # Transfer to GPU
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_work_start = cuda.to_device(work_start)
    d_work_end = cuda.to_device(work_end)
    
    d_output_naive = cuda.device_array(num_images, dtype=np.float32)
    d_output_balanced = cuda.device_array(num_threads, dtype=np.float32)
    
    # Kernel config
    threads_per_block = 256
    blocks_naive = (num_images + threads_per_block - 1) // threads_per_block
    blocks_balanced = (num_threads + threads_per_block - 1) // threads_per_block
    
    # Warmup
    print(f"\nWarmup ({warmup} runs)...")
    for _ in range(warmup):
        naive_image_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        balanced_image_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
    cuda.synchronize()
    
    # Benchmark naive
    print(f"Benchmarking NAIVE ({runs} runs)...")
    naive_times = []
    for _ in range(runs):
        start = time.perf_counter()
        naive_image_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        cuda.synchronize()
        naive_times.append(time.perf_counter() - start)
    
    # Benchmark balanced
    print(f"Benchmarking BALANCED ({runs} runs)...")
    balanced_times = []
    for _ in range(runs):
        start = time.perf_counter()
        balanced_image_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
        cuda.synchronize()
        balanced_times.append(time.perf_counter() - start)
    
    # Results
    from scipy import stats
    
    naive_avg = np.mean(naive_times) * 1000
    naive_std = np.std(naive_times) * 1000
    naive_ci = stats.t.interval(0.95, len(naive_times)-1, 
                                 loc=naive_avg, 
                                 scale=stats.sem(naive_times)*1000)
    
    balanced_avg = np.mean(balanced_times) * 1000
    balanced_std = np.std(balanced_times) * 1000
    balanced_ci = stats.t.interval(0.95, len(balanced_times)-1,
                                    loc=balanced_avg,
                                    scale=stats.sem(balanced_times)*1000)
    
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    print(f"\nTiming (n={len(naive_times)} runs):")
    print(f"  Naive:    {naive_avg:.3f} ms (±{naive_std:.3f})")
    print(f"            95% CI: [{naive_ci[0]:.3f}, {naive_ci[1]:.3f}]")
    print(f"  Balanced: {balanced_avg:.3f} ms (±{balanced_std:.3f})")
    print(f"            95% CI: [{balanced_ci[0]:.3f}, {balanced_ci[1]:.3f}]")
    
    # Statistical significance test
    t_stat, p_value = stats.ttest_ind(naive_times, balanced_times)
    print(f"\n  Statistical test:")
    print(f"    t-statistic: {t_stat:.2f}")
    print(f"    p-value: {p_value:.2e}")
    if p_value < 0.001:
        print(f"    >>> HIGHLY SIGNIFICANT (p < 0.001) <<<")
    elif p_value < 0.05:
        print(f"    >>> SIGNIFICANT (p < 0.05) <<<")
    
    speedup = naive_avg / balanced_avg
    
    if speedup > 1:
        print(f"\n  >>> SPEEDUP: {speedup:.2f}x <<<")
        print(f"  >>> Time saved: {(1-balanced_avg/naive_avg)*100:.1f}% <<<")
    else:
        print(f"\n  Balanced was {1/speedup:.2f}x slower")
    
    print("=" * 60)
    
    return {
        'name': name,
        'num_images': num_images,
        'total_pixels': total_pixels,
        'naive_ms': naive_avg,
        'naive_std': naive_std,
        'balanced_ms': balanced_avg,
        'balanced_std': balanced_std,
        'speedup': speedup,
        'theoretical': theoretical,
        'imbalance': imbalance
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("MEDICAL IMAGE BENCHMARK")
    print("Real CT Scan Data - Octopus Load Balancing")
    print("=" * 60)
    print(f"\nGPU: {cuda.gpus}")
    
    DATA_DIR = os.path.expanduser("~/cuda-test/Data")
    
    results = []
    
    # ----------------------------------------
    # TEST 1: Small batch (50 images)
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 1: Small batch (50 images)")
    print("█" * 60)
    
    images_flat, offsets, sizes, dims = load_medical_images(
        DATA_DIR, max_images=50)
    r = run_medical_benchmark(
        images_flat, offsets, sizes, dims,
        num_threads=50, name="Small Batch (50)")
    results.append(r)
    
    # ----------------------------------------
    # TEST 2: Medium batch (200 images)
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 2: Medium batch (200 images)")
    print("█" * 60)
    
    images_flat, offsets, sizes, dims = load_medical_images(
        DATA_DIR, max_images=200)
    r = run_medical_benchmark(
        images_flat, offsets, sizes, dims,
        num_threads=200, name="Medium Batch (200)")
    results.append(r)
    
    # ----------------------------------------
    # TEST 3: Full dataset (~1000 images)
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 3: Full dataset (~1000 images)")
    print("█" * 60)
    
    images_flat, offsets, sizes, dims = load_medical_images(
        DATA_DIR, max_images=None)
    r = run_medical_benchmark(
        images_flat, offsets, sizes, dims,
        num_threads=len(sizes), name="Full Dataset")
    results.append(r)
    
    # ----------------------------------------
    # TEST 4: Simulated mixed workload
    # Add one large synthetic image to create imbalance
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print("TEST 4: Mixed workload (real CT + 1 large synthetic)")
    print("█" * 60)
    
    # Load 100 real images
    images_flat, offsets, sizes, dims = load_medical_images(
        DATA_DIR, max_images=100)
    
    # Add one large synthetic "full resolution scan"
    large_size = 4096 * 4096
    large_image = np.random.rand(large_size).astype(np.float32)
    
    # Concatenate
    new_flat = np.concatenate([images_flat, large_image])
    new_offsets = np.concatenate([offsets, [len(images_flat)]])
    new_sizes = np.concatenate([sizes, [large_size]])
    new_dims = dims + [(4096, 4096)]
    
    r = run_medical_benchmark(
        new_flat, new_offsets, new_sizes, new_dims,
        num_threads=101, name="Mixed (100 CT + 1 Large)")
    results.append(r)
    
    # ----------------------------------------
    # SUMMARY
    # ----------------------------------------
    print("\n\n" + "=" * 60)
    print("SUMMARY - REAL MEDICAL IMAGE BENCHMARK")
    print("=" * 60)
    
    print(f"\n{'Test':<25} {'Images':>8} {'Pixels':>12} {'Imbal':>8} {'Speedup':>10} {'Status':>8}")
    print("-" * 75)
    
    for r in results:
        status = "✓ WIN" if r['speedup'] > 1.05 else ("~ TIE" if r['speedup'] > 0.95 else "✗ LOSE")
        print(f"{r['name']:<25} {r['num_images']:>8} {r['total_pixels']:>12,} {r['imbalance']:>7.2f}x {r['speedup']:>9.2f}x {status:>8}")
    
    print("\n" + "=" * 60)
    
    wins = sum(1 for r in results if r['speedup'] > 1.05)
    print(f"\nBalanced approach wins: {wins}/{len(results)} tests")
    
    if wins > 0:
        best = max(results, key=lambda x: x['speedup'])
        print(f"Best speedup: {best['speedup']:.2f}x on '{best['name']}'")
    
    print("\n🐙 Medical image benchmark complete!")
    
    return results


if __name__ == "__main__":
    results = main()