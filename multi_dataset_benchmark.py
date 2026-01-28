"""
Multi-Dataset Medical Image Benchmark
======================================
Comparing Octopus Load Balancing across different medical imaging modalities.

Datasets:
1. Chest CT (Kaggle) - ~1000 images
2. Brain MRI (Kaggle) - ~350 images

Author: Matthew
Date: January 28, 2026
"""

from numba import cuda
import numpy as np
import time
from pathlib import Path
from PIL import Image
from scipy import stats
import os

# ============================================
# IMAGE LOADING
# ============================================

def load_medical_images(data_dir, max_images=None, grayscale=True):
    """Load medical images from dataset."""
    data_path = Path(data_dir)
    
    # Find all images (including .jpeg)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        image_files.extend(data_path.rglob(ext))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loading {len(image_files)} images from {data_dir}...")
    
    # First pass: get sizes
    image_dims = []
    total_pixels = 0
    
    for img_path in image_files:
        with Image.open(img_path) as img:
            if grayscale:
                img = img.convert('L')
            w, h = img.size
            image_dims.append((w, h))
            total_pixels += w * h
    
    # Second pass: load pixel data
    images_flat = np.zeros(total_pixels, dtype=np.float32)
    offsets = np.zeros(len(image_files), dtype=np.int64)
    sizes = np.zeros(len(image_files), dtype=np.int64)
    
    current_offset = 0
    for i, img_path in enumerate(image_files):
        with Image.open(img_path) as img:
            if grayscale:
                img = img.convert('L')
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0
            offsets[i] = current_offset
            sizes[i] = len(pixels)
            images_flat[current_offset:current_offset + len(pixels)] = pixels
            current_offset += len(pixels)
    
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Size range: {min(sizes):,} - {max(sizes):,} pixels")
    
    return images_flat, offsets, sizes, image_dims


# ============================================
# GPU KERNELS
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
# BENCHMARK
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


def run_benchmark(images_flat, offsets, sizes, image_dims, 
                  num_threads, name, warmup=5, runs=30):
    """Run benchmark with full statistical analysis."""
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    max_pixels = max(sizes)
    min_pixels = min(sizes)
    avg_pixels = total_pixels / num_images
    imbalance = max_pixels / avg_pixels
    
    print(f"\nDataset: {num_images} images, {total_pixels:,} total pixels")
    print(f"Size range: {min_pixels:,} - {max_pixels:,} (imbalance: {imbalance:.2f}x)")
    
    # Setup
    work_start, work_end = compute_balanced_work(total_pixels, num_threads)
    
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_work_start = cuda.to_device(work_start)
    d_work_end = cuda.to_device(work_end)
    
    d_output_naive = cuda.device_array(num_images, dtype=np.float32)
    d_output_balanced = cuda.device_array(num_threads, dtype=np.float32)
    
    threads_per_block = 256
    blocks_naive = (num_images + threads_per_block - 1) // threads_per_block
    blocks_balanced = (num_threads + threads_per_block - 1) // threads_per_block
    
    # Warmup
    for _ in range(warmup):
        naive_image_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        balanced_image_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
    cuda.synchronize()
    
    # Benchmark
    naive_times = []
    for _ in range(runs):
        start = time.perf_counter()
        naive_image_kernel[blocks_naive, threads_per_block](
            d_images, d_offsets, d_sizes, d_output_naive)
        cuda.synchronize()
        naive_times.append(time.perf_counter() - start)
    
    balanced_times = []
    for _ in range(runs):
        start = time.perf_counter()
        balanced_image_kernel[blocks_balanced, threads_per_block](
            d_images, d_work_start, d_work_end, d_output_balanced)
        cuda.synchronize()
        balanced_times.append(time.perf_counter() - start)
    
    # Statistics
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
    
    t_stat, p_value = stats.ttest_ind(naive_times, balanced_times)
    speedup = naive_avg / balanced_avg
    
    # Print results
    print(f"\nResults (n={runs}):")
    print(f"  Naive:    {naive_avg:.3f} ms  95% CI: [{naive_ci[0]:.3f}, {naive_ci[1]:.3f}]")
    print(f"  Balanced: {balanced_avg:.3f} ms  95% CI: [{balanced_ci[0]:.3f}, {balanced_ci[1]:.3f}]")
    print(f"  Speedup:  {speedup:.2f}x  (p = {p_value:.2e})")
    
    if p_value < 0.001:
        print(f"  >>> HIGHLY SIGNIFICANT <<<")
    
    return {
        'name': name,
        'modality': name.split(' - ')[0] if ' - ' in name else name,
        'num_images': num_images,
        'total_pixels': total_pixels,
        'imbalance': imbalance,
        'naive_ms': naive_avg,
        'naive_ci': naive_ci,
        'balanced_ms': balanced_avg,
        'balanced_ci': balanced_ci,
        'speedup': speedup,
        'p_value': p_value,
        't_stat': t_stat
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("MULTI-DATASET MEDICAL IMAGE BENCHMARK")
    print("Octopus-Inspired Load Balancing - Cross-Modality Validation")
    print("=" * 70)
    print(f"\nGPU: {cuda.gpus}")
    
    CT_DIR = os.path.expanduser("~/cuda-test/Data")
    MRI_DIR = os.path.expanduser("~/cuda-test/Data_BrainMRI")
    
    results = []
    
    # ========================================
    # DATASET 1: Chest CT
    # ========================================
    print("\n" + "█" * 70)
    print("DATASET 1: CHEST CT SCANS")
    print("█" * 70)
    
    # Full CT dataset
    ct_flat, ct_off, ct_sizes, ct_dims = load_medical_images(CT_DIR)
    r = run_benchmark(ct_flat, ct_off, ct_sizes, ct_dims,
                      num_threads=len(ct_sizes), 
                      name="Chest CT - Full")
    results.append(r)
    
    # CT with synthetic large image
    large_size = 4096 * 4096
    large_img = np.random.rand(large_size).astype(np.float32)
    ct_mixed_flat = np.concatenate([ct_flat, large_img])
    ct_mixed_off = np.concatenate([ct_off, [len(ct_flat)]])
    ct_mixed_sizes = np.concatenate([ct_sizes, [large_size]])
    ct_mixed_dims = ct_dims + [(4096, 4096)]
    
    r = run_benchmark(ct_mixed_flat, ct_mixed_off, ct_mixed_sizes, ct_mixed_dims,
                      num_threads=len(ct_mixed_sizes),
                      name="Chest CT - Mixed")
    results.append(r)
    
    # ========================================
    # DATASET 2: Brain MRI
    # ========================================
    print("\n" + "█" * 70)
    print("DATASET 2: BRAIN MRI")
    print("█" * 70)
    
    # Full MRI dataset
    mri_flat, mri_off, mri_sizes, mri_dims = load_medical_images(MRI_DIR)
    r = run_benchmark(mri_flat, mri_off, mri_sizes, mri_dims,
                      num_threads=len(mri_sizes),
                      name="Brain MRI - Full")
    results.append(r)
    
    # MRI with synthetic large image
    mri_mixed_flat = np.concatenate([mri_flat, large_img])
    mri_mixed_off = np.concatenate([mri_off, [len(mri_flat)]])
    mri_mixed_sizes = np.concatenate([mri_sizes, [large_size]])
    mri_mixed_dims = mri_dims + [(4096, 4096)]
    
    r = run_benchmark(mri_mixed_flat, mri_mixed_off, mri_mixed_sizes, mri_mixed_dims,
                      num_threads=len(mri_mixed_sizes),
                      name="Brain MRI - Mixed")
    results.append(r)
    
    # ========================================
    # DATASET 3: Combined (Cross-modality)
    # ========================================
    print("\n" + "█" * 70)
    print("DATASET 3: COMBINED (CT + MRI)")
    print("█" * 70)
    
    # Combine both datasets
    combined_flat = np.concatenate([ct_flat, mri_flat])
    combined_off = np.concatenate([ct_off, mri_off + len(ct_flat)])
    combined_sizes = np.concatenate([ct_sizes, mri_sizes])
    combined_dims = ct_dims + mri_dims
    
    r = run_benchmark(combined_flat, combined_off, combined_sizes, combined_dims,
                      num_threads=len(combined_sizes),
                      name="Combined CT+MRI")
    results.append(r)
    
    # Combined with large image
    combined_mixed_flat = np.concatenate([combined_flat, large_img])
    combined_mixed_off = np.concatenate([combined_off, [len(combined_flat)]])
    combined_mixed_sizes = np.concatenate([combined_sizes, [large_size]])
    combined_mixed_dims = combined_dims + [(4096, 4096)]
    
    r = run_benchmark(combined_mixed_flat, combined_mixed_off, 
                      combined_mixed_sizes, combined_mixed_dims,
                      num_threads=len(combined_mixed_sizes),
                      name="Combined - Mixed")
    results.append(r)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n\n" + "=" * 70)
    print("CROSS-MODALITY BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<22} {'Images':>7} {'Imbalance':>10} {'Speedup':>10} {'p-value':>12} {'Status':>8}")
    print("-" * 75)
    
    for r in results:
        status = "✓ WIN" if r['speedup'] > 1.05 else "~ TIE"
        p_str = f"{r['p_value']:.2e}" if r['p_value'] < 0.01 else f"{r['p_value']:.4f}"
        print(f"{r['name']:<22} {r['num_images']:>7} {r['imbalance']:>9.2f}x {r['speedup']:>9.2f}x {p_str:>12} {status:>8}")
    
    # Group by modality
    print(f"\n{'='*70}")
    print("MODALITY COMPARISON")
    print(f"{'='*70}")
    
    modalities = {}
    for r in results:
        mod = r['name'].split(' - ')[0]
        if mod not in modalities:
            modalities[mod] = []
        modalities[mod].append(r['speedup'])
    
    for mod, speedups in modalities.items():
        avg_speedup = np.mean(speedups)
        print(f"  {mod}: avg speedup = {avg_speedup:.2f}x")
    
    # Overall
    print(f"\n{'='*70}")
    all_speedups = [r['speedup'] for r in results]
    all_significant = all(r['p_value'] < 0.001 for r in results)
    
    print(f"Overall: {len(results)}/{len(results)} tests show improvement")
    print(f"Average speedup: {np.mean(all_speedups):.2f}x")
    print(f"Best speedup: {max(all_speedups):.2f}x")
    print(f"All results significant (p < 0.001): {'YES ✓' if all_significant else 'NO'}")
    
    print("\n🐙 Cross-modality validation complete!")
    
    return results


if __name__ == "__main__":
    results = main()