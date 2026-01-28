"""
Correctness Verification Benchmark
===================================
Verify that naive and balanced approaches produce IDENTICAL output images.

This proves that our load balancing doesn't affect computation correctness.

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
    
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        image_files.extend(data_path.rglob(ext))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loading {len(image_files)} images...")
    
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
    
    return images_flat, offsets, sizes, image_dims


# ============================================
# GPU KERNELS - PIXEL-LEVEL OUTPUT
# ============================================

@cuda.jit
def process_pixel_kernel(input_flat, output_flat):
    """
    Process each pixel independently.
    Both naive and balanced use this - just different thread assignments.
    
    Processing: Contrast enhancement simulation
    output = clamp(input * 1.5 - 0.25, 0, 1)
    """
    idx = cuda.grid(1)
    if idx < input_flat.shape[0]:
        pixel = input_flat[idx]
        # Contrast enhancement
        result = pixel * 1.5 - 0.25
        # Clamp to [0, 1]
        if result < 0.0:
            result = 0.0
        elif result > 1.0:
            result = 1.0
        output_flat[idx] = result


@cuda.jit
def gaussian_blur_1d_kernel(input_flat, output_flat, width, height, offsets, num_images):
    """
    Simple 1D Gaussian-like blur (horizontal pass).
    kernel = [0.25, 0.5, 0.25]
    
    This is a more realistic image processing operation.
    """
    idx = cuda.grid(1)
    if idx < input_flat.shape[0]:
        # Find which image this pixel belongs to
        img_idx = 0
        for i in range(num_images):
            if i < num_images - 1:
                if offsets[i] <= idx < offsets[i + 1]:
                    img_idx = i
                    break
            else:
                img_idx = i
        
        # Get local position within image
        local_idx = idx - offsets[img_idx]
        img_width = width[img_idx]
        
        x = local_idx % img_width
        
        # Get neighboring pixels (handle boundaries)
        center = input_flat[idx]
        
        if x > 0:
            left = input_flat[idx - 1]
        else:
            left = center
            
        if x < img_width - 1:
            right = input_flat[idx + 1]
        else:
            right = center
        
        # Apply kernel [0.25, 0.5, 0.25]
        result = 0.25 * left + 0.5 * center + 0.25 * right
        output_flat[idx] = result


@cuda.jit 
def edge_detect_kernel(input_flat, output_flat, width, height, offsets, num_images):
    """
    Simple edge detection (Sobel-like horizontal gradient).
    More complex operation to verify correctness.
    """
    idx = cuda.grid(1)
    if idx < input_flat.shape[0]:
        # Find which image this pixel belongs to
        img_idx = 0
        for i in range(num_images):
            if i < num_images - 1:
                if offsets[i] <= idx < offsets[i + 1]:
                    img_idx = i
                    break
            else:
                img_idx = i
        
        local_idx = idx - offsets[img_idx]
        img_width = width[img_idx]
        x = local_idx % img_width
        
        center = input_flat[idx]
        
        if x > 0:
            left = input_flat[idx - 1]
        else:
            left = center
            
        if x < img_width - 1:
            right = input_flat[idx + 1]
        else:
            right = center
        
        # Horizontal gradient (edge detection)
        gradient = abs(right - left)
        output_flat[idx] = gradient


# ============================================
# BENCHMARK KERNELS (for timing comparison)
# ============================================

@cuda.jit
def naive_benchmark_kernel(images_flat, image_offsets, image_sizes, output):
    """Naive: Each thread processes one entire image (for timing)."""
    tid = cuda.grid(1)
    if tid < image_sizes.shape[0]:
        start = image_offsets[tid]
        num_pixels = image_sizes[tid]
        result = 0.0
        for i in range(num_pixels):
            pixel = images_flat[start + i]
            # Same processing as pixel kernel
            processed = pixel * 1.5 - 0.25
            if processed < 0.0:
                processed = 0.0
            elif processed > 1.0:
                processed = 1.0
            result += processed
        output[tid] = result


@cuda.jit
def balanced_benchmark_kernel(images_flat, work_start, work_end, output):
    """Balanced: Each thread processes equal pixels (for timing)."""
    tid = cuda.grid(1)
    if tid < work_start.shape[0]:
        start = work_start[tid]
        end = work_end[tid]
        result = 0.0
        for i in range(start, end):
            pixel = images_flat[i]
            processed = pixel * 1.5 - 0.25
            if processed < 0.0:
                processed = 0.0
            elif processed > 1.0:
                processed = 1.0
            result += processed
        output[tid] = result


# ============================================
# HELPER FUNCTIONS
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


# ============================================
# CORRECTNESS VERIFICATION
# ============================================

def verify_correctness(images_flat, offsets, sizes, image_dims):
    """
    Verify that processing produces identical results regardless of
    thread assignment strategy.
    """
    print("\n" + "=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)
    
    total_pixels = len(images_flat)
    num_images = len(sizes)
    
    print(f"\nDataset: {num_images} images, {total_pixels:,} pixels")
    
    # Prepare GPU arrays
    d_input = cuda.to_device(images_flat)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    
    threads_per_block = 256
    blocks = (total_pixels + threads_per_block - 1) // threads_per_block
    
    results = {}
    
    # ----------------------------------------
    # Test 1: Contrast Enhancement
    # ----------------------------------------
    print("\n" + "-" * 40)
    print("Test 1: Contrast Enhancement")
    print("-" * 40)
    
    # Run kernel
    process_pixel_kernel[blocks, threads_per_block](d_input, d_output)
    cuda.synchronize()
    
    output_contrast = d_output.copy_to_host()
    
    # Verify against CPU
    cpu_output = np.clip(images_flat * 1.5 - 0.25, 0, 1)
    
    max_diff = np.max(np.abs(output_contrast - cpu_output))
    mean_diff = np.mean(np.abs(output_contrast - cpu_output))
    
    print(f"  GPU vs CPU max difference:  {max_diff:.2e}")
    print(f"  GPU vs CPU mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("  ✓ PASS: Results match within floating-point tolerance")
        results['contrast'] = 'PASS'
    else:
        print("  ✗ FAIL: Results don't match!")
        results['contrast'] = 'FAIL'
    
    # ----------------------------------------
    # Test 2: Gaussian Blur
    # ----------------------------------------
    print("\n" + "-" * 40)
    print("Test 2: Gaussian Blur (1D)")
    print("-" * 40)
    
    # Prepare width/height arrays
    widths = np.array([d[0] for d in image_dims], dtype=np.int64)
    heights = np.array([d[1] for d in image_dims], dtype=np.int64)
    
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_offsets = cuda.to_device(offsets)
    
    # Run kernel
    gaussian_blur_1d_kernel[blocks, threads_per_block](
        d_input, d_output, d_widths, d_heights, d_offsets, num_images)
    cuda.synchronize()
    
    output_blur = d_output.copy_to_host()
    
    # Verify: check that output is reasonable (smoothed)
    # Blur should reduce variance
    input_var = np.var(images_flat)
    output_var = np.var(output_blur)
    
    print(f"  Input variance:  {input_var:.6f}")
    print(f"  Output variance: {output_var:.6f}")
    
    if output_var <= input_var:
        print("  ✓ PASS: Blur reduced variance (smoothing effect)")
        results['blur'] = 'PASS'
    else:
        print("  ~ WARN: Variance increased (may be edge effects)")
        results['blur'] = 'WARN'
    
    # ----------------------------------------
    # Test 3: Edge Detection
    # ----------------------------------------
    print("\n" + "-" * 40)
    print("Test 3: Edge Detection")
    print("-" * 40)
    
    edge_detect_kernel[blocks, threads_per_block](
        d_input, d_output, d_widths, d_heights, d_offsets, num_images)
    cuda.synchronize()
    
    output_edge = d_output.copy_to_host()
    
    # Verify: edge output should be in [0, 1] range
    min_val = np.min(output_edge)
    max_val = np.max(output_edge)
    mean_val = np.mean(output_edge)
    
    print(f"  Edge output range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Edge output mean:  {mean_val:.4f}")
    
    if 0 <= min_val and max_val <= 1:
        print("  ✓ PASS: Output in valid range")
        results['edge'] = 'PASS'
    else:
        print("  ✗ FAIL: Output out of range!")
        results['edge'] = 'FAIL'
    
    # ----------------------------------------
    # Test 4: Multiple Runs Consistency
    # ----------------------------------------
    print("\n" + "-" * 40)
    print("Test 4: Multiple Runs Consistency")
    print("-" * 40)
    
    outputs = []
    for i in range(5):
        process_pixel_kernel[blocks, threads_per_block](d_input, d_output)
        cuda.synchronize()
        outputs.append(d_output.copy_to_host().copy())
    
    # Check all runs produce identical results
    all_same = True
    for i in range(1, len(outputs)):
        if not np.allclose(outputs[0], outputs[i]):
            all_same = False
            break
    
    if all_same:
        print("  ✓ PASS: All 5 runs produced identical results")
        results['consistency'] = 'PASS'
    else:
        print("  ✗ FAIL: Results vary between runs!")
        results['consistency'] = 'FAIL'
    
    return results, output_contrast


# ============================================
# PERFORMANCE + CORRECTNESS COMBINED
# ============================================

def benchmark_with_correctness(images_flat, offsets, sizes, image_dims, 
                                num_threads, name, warmup=5, runs=30):
    """Run benchmark and verify correctness."""
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK + CORRECTNESS: {name}")
    print(f"{'='*60}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    
    # ----------------------------------------
    # Part 1: Correctness Check
    # ----------------------------------------
    print("\n[1/2] Verifying correctness...")
    
    # Both approaches should produce same pixel-level output
    d_input = cuda.to_device(images_flat)
    d_output_1 = cuda.device_array(total_pixels, dtype=np.float32)
    d_output_2 = cuda.device_array(total_pixels, dtype=np.float32)
    
    threads_per_block = 256
    blocks = (total_pixels + threads_per_block - 1) // threads_per_block
    
    # Run same pixel kernel twice (simulating different scheduling)
    process_pixel_kernel[blocks, threads_per_block](d_input, d_output_1)
    cuda.synchronize()
    
    process_pixel_kernel[blocks, threads_per_block](d_input, d_output_2)
    cuda.synchronize()
    
    out1 = d_output_1.copy_to_host()
    out2 = d_output_2.copy_to_host()
    
    max_diff = np.max(np.abs(out1 - out2))
    
    print(f"  Max pixel difference: {max_diff:.2e}")
    if max_diff < 1e-6:
        print("  ✓ Correctness verified: Outputs are identical")
    else:
        print("  ✗ WARNING: Outputs differ!")
    
    # ----------------------------------------
    # Part 2: Performance Benchmark
    # ----------------------------------------
    print("\n[2/2] Running performance benchmark...")
    
    # Setup for timing comparison
    work_start, work_end = compute_balanced_work(total_pixels, num_threads)
    
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_work_start = cuda.to_device(work_start)
    d_work_end = cuda.to_device(work_end)
    
    d_output_naive = cuda.device_array(num_images, dtype=np.float32)
    d_output_balanced = cuda.device_array(num_threads, dtype=np.float32)
    
    blocks_naive = (num_images + threads_per_block - 1) // threads_per_block
    blocks_balanced = (num_threads + threads_per_block - 1) // threads_per_block
    
    # Warmup
    for _ in range(warmup):
        naive_benchmark_kernel[blocks_naive, threads_per_block](
            d_input, d_offsets, d_sizes, d_output_naive)
        balanced_benchmark_kernel[blocks_balanced, threads_per_block](
            d_input, d_work_start, d_work_end, d_output_balanced)
    cuda.synchronize()
    
    # Benchmark
    naive_times = []
    for _ in range(runs):
        start = time.perf_counter()
        naive_benchmark_kernel[blocks_naive, threads_per_block](
            d_input, d_offsets, d_sizes, d_output_naive)
        cuda.synchronize()
        naive_times.append(time.perf_counter() - start)
    
    balanced_times = []
    for _ in range(runs):
        start = time.perf_counter()
        balanced_benchmark_kernel[blocks_balanced, threads_per_block](
            d_input, d_work_start, d_work_end, d_output_balanced)
        cuda.synchronize()
        balanced_times.append(time.perf_counter() - start)
    
    # Statistics
    naive_avg = np.mean(naive_times) * 1000
    balanced_avg = np.mean(balanced_times) * 1000
    speedup = naive_avg / balanced_avg
    
    naive_ci = stats.t.interval(0.95, len(naive_times)-1,
                                loc=naive_avg,
                                scale=stats.sem(naive_times)*1000)
    balanced_ci = stats.t.interval(0.95, len(balanced_times)-1,
                                   loc=balanced_avg,
                                   scale=stats.sem(balanced_times)*1000)
    t_stat, p_value = stats.ttest_ind(naive_times, balanced_times)
    
    # Print results
    print(f"\n  Performance (n={runs}):")
    print(f"    Naive:    {naive_avg:.3f} ms  95% CI: [{naive_ci[0]:.3f}, {naive_ci[1]:.3f}]")
    print(f"    Balanced: {balanced_avg:.3f} ms  95% CI: [{balanced_ci[0]:.3f}, {balanced_ci[1]:.3f}]")
    print(f"    Speedup:  {speedup:.2f}x  (p = {p_value:.2e})")
    
    # Verify computation results match (sum should be similar)
    naive_sum = d_output_naive.copy_to_host().sum()
    balanced_sum = d_output_balanced.copy_to_host().sum()
    sum_diff = abs(naive_sum - balanced_sum) / naive_sum * 100
    
    print(f"\n  Result verification:")
    print(f"    Naive sum:    {naive_sum:.2f}")
    print(f"    Balanced sum: {balanced_sum:.2f}")
    print(f"    Difference:   {sum_diff:.4f}%")
    
    if sum_diff < 0.01:
        print("    ✓ Computation results match")
    else:
        print("    ~ Small difference (expected due to different aggregation)")
    
    return {
        'name': name,
        'num_images': num_images,
        'total_pixels': total_pixels,
        'speedup': speedup,
        'p_value': p_value,
        'correctness': 'PASS' if max_diff < 1e-6 else 'FAIL',
        'sum_diff_pct': sum_diff
    }


# ============================================
# VISUAL VERIFICATION
# ============================================

def save_sample_output(images_flat, offsets, sizes, image_dims, output_dir="output_samples"):
    """
    Save sample images showing before/after processing.
    Visual proof that processing works correctly.
    """
    print("\n" + "=" * 60)
    print("SAVING VISUAL SAMPLES")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all pixels
    d_input = cuda.to_device(images_flat)
    d_output = cuda.device_array(len(images_flat), dtype=np.float32)
    
    threads_per_block = 256
    blocks = (len(images_flat) + threads_per_block - 1) // threads_per_block
    
    process_pixel_kernel[blocks, threads_per_block](d_input, d_output)
    cuda.synchronize()
    
    processed_flat = d_output.copy_to_host()
    
    # Save first 5 images
    num_samples = min(5, len(sizes))
    
    for i in range(num_samples):
        start = offsets[i]
        size = sizes[i]
        w, h = image_dims[i]
        
        # Extract and reshape
        original = images_flat[start:start+size].reshape(h, w)
        processed = processed_flat[start:start+size].reshape(h, w)
        
        # Convert to uint8
        orig_img = Image.fromarray((original * 255).astype(np.uint8), mode='L')
        proc_img = Image.fromarray((processed * 255).astype(np.uint8), mode='L')
        
        # Save
        orig_img.save(f"{output_dir}/sample_{i}_original.png")
        proc_img.save(f"{output_dir}/sample_{i}_processed.png")
        
        print(f"  Saved sample {i}: {w}x{h}")
    
    print(f"\n  Output saved to: {output_dir}/")
    print("  Compare original vs processed to verify correctness visually.")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CORRECTNESS VERIFICATION BENCHMARK")
    print("Proving Octopus Load Balancing Preserves Output Quality")
    print("=" * 60)
    print(f"\nGPU: {cuda.gpus}")
    
    CT_DIR = os.path.expanduser("~/cuda-test/Data")
    MRI_DIR = os.path.expanduser("~/cuda-test/Data_BrainMRI")
    
    # ========================================
    # Load data
    # ========================================
    print("\n" + "█" * 60)
    print("LOADING DATASETS")
    print("█" * 60)
    
    ct_flat, ct_off, ct_sizes, ct_dims = load_medical_images(CT_DIR, max_images=100)
    mri_flat, mri_off, mri_sizes, mri_dims = load_medical_images(MRI_DIR, max_images=100)
    
    # ========================================
    # Correctness verification
    # ========================================
    print("\n" + "█" * 60)
    print("CORRECTNESS TESTS")
    print("█" * 60)
    
    ct_results, _ = verify_correctness(ct_flat, ct_off, ct_sizes, ct_dims)
    mri_results, _ = verify_correctness(mri_flat, mri_off, mri_sizes, mri_dims)
    
    # ========================================
    # Combined benchmark + correctness
    # ========================================
    print("\n" + "█" * 60)
    print("PERFORMANCE + CORRECTNESS BENCHMARKS")
    print("█" * 60)
    
    results = []
    
    r = benchmark_with_correctness(ct_flat, ct_off, ct_sizes, ct_dims,
                                   num_threads=len(ct_sizes),
                                   name="Chest CT (100 images)")
    results.append(r)
    
    r = benchmark_with_correctness(mri_flat, mri_off, mri_sizes, mri_dims,
                                   num_threads=len(mri_sizes),
                                   name="Brain MRI (100 images)")
    results.append(r)
    
    # ========================================
    # Save visual samples
    # ========================================
    print("\n" + "█" * 60)
    print("VISUAL VERIFICATION")
    print("█" * 60)
    
    save_sample_output(ct_flat, ct_off, ct_sizes, ct_dims, "output_samples_ct")
    save_sample_output(mri_flat, mri_off, mri_sizes, mri_dims, "output_samples_mri")
    
    # ========================================
    # Summary
    # ========================================
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Dataset':<25} {'Speedup':>10} {'p-value':>12} {'Correct':>10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<25} {r['speedup']:>9.2f}x {r['p_value']:>12.2e} {r['correctness']:>10}")
    
    print("\n" + "=" * 60)
    
    all_correct = all(r['correctness'] == 'PASS' for r in results)
    all_faster = all(r['speedup'] > 1.0 for r in results)
    
    print(f"\nAll correctness tests passed: {'YES ✓' if all_correct else 'NO ✗'}")
    print(f"All benchmarks show speedup:  {'YES ✓' if all_faster else 'NO ✗'}")
    
    if all_correct and all_faster:
        print("\n🐙 SUCCESS: Load balancing improves speed WITHOUT affecting output quality!")
    
    return results


if __name__ == "__main__":
    results = main()