"""
Edge Device Benchmark: CPU vs Block Metadata GPU
=================================================
Direct comparison for edge deployment scenario:
- PIL CPU resize (baseline - what you'd use without GPU)
- Block metadata GPU resize (your approach)

This simulates edge device where PyTorch isn't available.

Usage:
    python edge_benchmark.py
"""

import numpy as np
import time
from pathlib import Path
from PIL import Image
import torch
from numba import cuda, njit

# ============================================
# LOAD FRAMES
# ============================================

def load_frames_from_folders(base_dir='frames', max_per_res=50):
    """Load frames from different resolution folders."""
    base_path = Path(base_dir)
    resolutions = ['180p', '480p', '720p', '1080p']
    
    frames = []
    metadata = []
    
    for res in resolutions:
        res_dir = base_path / res
        if not res_dir.exists():
            continue
        
        frame_files = sorted(res_dir.glob('frame_*.jpg'))[:max_per_res]
        
        for f in frame_files:
            img = Image.open(f).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            frames.append(arr)
            metadata.append({
                'resolution': res,
                'width': arr.shape[1],
                'height': arr.shape[0],
                'pixels': arr.shape[0] * arr.shape[1],
            })
        
        print(f"{res}: loaded {len(frame_files)} frames")
    
    return frames, metadata


# ============================================
# METHOD 1: PIL CPU RESIZE (EDGE BASELINE)
# ============================================

def preprocess_pil_cpu(frames, target_size=640):
    """
    CPU resize using PIL - edge device baseline.
    No GPU, no PyTorch, just pure CPU.
    """
    processed = []
    
    for frame in frames:
        # Convert to PIL
        img = Image.fromarray((frame * 255).astype(np.uint8))
        
        # Resize on CPU
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        
        # Back to numpy, normalize, HWC -> CHW
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        processed.append(arr)
    
    return np.stack(processed)


# ============================================
# METHOD 2: BLOCK METADATA GPU RESIZE
# ============================================

@njit(cache=True)
def build_block_metadata(widths, heights, target_size, threshold=4096):
    """Build block metadata for resize."""
    num_images = len(widths)
    output_pixels_per_image = target_size * target_size * 3
    
    total_blocks = 0
    for i in range(num_images):
        pixels = output_pixels_per_image
        if pixels <= threshold:
            total_blocks += 1
        else:
            total_blocks += (pixels + threshold - 1) // threshold
    
    block_to_image = np.empty(total_blocks, dtype=np.int32)
    block_start = np.empty(total_blocks, dtype=np.int64)
    block_end = np.empty(total_blocks, dtype=np.int64)
    
    current_block = 0
    for img_id in range(num_images):
        pixels = output_pixels_per_image
        
        if pixels <= threshold:
            block_to_image[current_block] = img_id
            block_start[current_block] = 0
            block_end[current_block] = pixels
            current_block += 1
        else:
            num_blocks = (pixels + threshold - 1) // threshold
            pixels_per_block = (pixels + num_blocks - 1) // num_blocks
            
            for b in range(num_blocks):
                start = b * pixels_per_block
                end = min(start + pixels_per_block, pixels)
                
                block_to_image[current_block] = img_id
                block_start[current_block] = start
                block_end[current_block] = end
                current_block += 1
    
    return block_to_image, block_start, block_end


@cuda.jit
def resize_kernel_block_metadata(
    flat_input, offsets, widths, heights,
    block_to_image, block_start, block_end,
    output, target_size
):
    """Resize kernel with O(1) block metadata lookup."""
    block_id = cuda.blockIdx.x
    
    if block_id >= block_to_image.shape[0]:
        return
    
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    src_w = widths[img_id]
    src_h = heights[img_id]
    src_offset = offsets[img_id]
    
    scale_x = src_w / target_size
    scale_y = src_h / target_size
    
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for out_idx in range(local_start + tid, local_end, stride):
        c = out_idx // (target_size * target_size)
        spatial_idx = out_idx % (target_size * target_size)
        out_y = spatial_idx // target_size
        out_x = spatial_idx % target_size
        
        src_x = (out_x + 0.5) * scale_x - 0.5
        src_y = (out_y + 0.5) * scale_y - 0.5
        
        x0 = int(src_x)
        y0 = int(src_y)
        x1 = x0 + 1
        y1 = y0 + 1
        
        x0 = max(0, min(x0, src_w - 1))
        x1 = max(0, min(x1, src_w - 1))
        y0 = max(0, min(y0, src_h - 1))
        y1 = max(0, min(y1, src_h - 1))
        
        wx = src_x - int(src_x)
        wy = src_y - int(src_y)
        wx = max(0.0, min(1.0, wx))
        wy = max(0.0, min(1.0, wy))
        
        idx00 = src_offset + (y0 * src_w + x0) * 3 + c
        idx01 = src_offset + (y0 * src_w + x1) * 3 + c
        idx10 = src_offset + (y1 * src_w + x0) * 3 + c
        idx11 = src_offset + (y1 * src_w + x1) * 3 + c
        
        v00 = flat_input[idx00]
        v01 = flat_input[idx01]
        v10 = flat_input[idx10]
        v11 = flat_input[idx11]
        
        value = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)
        
        output[img_id, c, out_y, out_x] = value


def preprocess_block_metadata_gpu(frames, metadata, target_size=640):
    """GPU resize using block metadata - your approach."""
    num_images = len(frames)
    
    # Flatten all frames
    flat_list = []
    offsets = [0]
    widths = []
    heights = []
    
    for frame in frames:
        h, w, c = frame.shape
        flat_list.append(frame.flatten())
        offsets.append(offsets[-1] + h * w * c)
        widths.append(w)
        heights.append(h)
    
    flat_input = np.concatenate(flat_list).astype(np.float32)
    offsets = np.array(offsets[:-1], dtype=np.int64)
    widths = np.array(widths, dtype=np.int32)
    heights = np.array(heights, dtype=np.int32)
    
    # Build block metadata
    block_to_image, block_start, block_end = build_block_metadata(
        widths, heights, target_size
    )
    num_blocks = len(block_to_image)
    
    # Transfer to GPU
    d_flat_input = cuda.to_device(flat_input)
    d_offsets = cuda.to_device(offsets)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_block_to_image = cuda.to_device(block_to_image)
    d_block_start = cuda.to_device(block_start)
    d_block_end = cuda.to_device(block_end)
    d_output = cuda.device_array((num_images, 3, target_size, target_size), dtype=np.float32)
    
    # Launch kernel
    threads = 256
    resize_kernel_block_metadata[num_blocks, threads](
        d_flat_input, d_offsets, d_widths, d_heights,
        d_block_to_image, d_block_start, d_block_end,
        d_output, target_size
    )
    cuda.synchronize()
    
    return d_output.copy_to_host()


# ============================================
# YOLO INFERENCE
# ============================================

def run_yolo(model, batch):
    """Run YOLO inference."""
    # Convert to torch tensor if numpy
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch).cuda()
    
    with torch.no_grad():
        results = model(batch, verbose=False)
    torch.cuda.synchronize()
    return results


def count_detections(results):
    """Count detections."""
    total = 0
    by_class = {}
    
    for r in results:
        boxes = r.boxes
        total += len(boxes)
        for cls_id in boxes.cls.cpu().numpy():
            cls_name = r.names[int(cls_id)]
            by_class[cls_name] = by_class.get(cls_name, 0) + 1
    
    return total, by_class


# ============================================
# BENCHMARK
# ============================================

def run_benchmark(frames, metadata, runs=5):
    """Direct comparison: CPU vs Block Metadata GPU."""
    
    print("\n" + "=" * 70)
    print("EDGE DEVICE BENCHMARK")
    print("CPU (PIL) vs GPU (Block Metadata)")
    print("=" * 70)
    
    print(f"\nDataset: {len(frames)} images")
    sizes = [m['pixels'] for m in metadata]
    print(f"Size range: {min(sizes):,} - {max(sizes):,} pixels")
    print(f"Imbalance: {max(sizes) / min(sizes):.2f}x")
    
    print("\nBy resolution:")
    for res in ['180p', '480p', '720p', '1080p']:
        count = sum(1 for m in metadata if m['resolution'] == res)
        if count > 0:
            print(f"  {res}: {count} images")
    
    # Load YOLO
    print("\nLoading YOLOv8...")
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.to('cuda')
    
    # Warmup
    print("Warming up...")
    dummy = torch.rand(1, 3, 640, 640).cuda()
    for _ in range(5):
        _ = model(dummy, verbose=False)
    torch.cuda.synchronize()
    
    # ========================================
    # CPU Baseline (PIL)
    # ========================================
    print("\n[CPU: PIL Resize]")
    print("  (Simulates edge device without GPU preprocessing)")
    
    cpu_preprocess_times = []
    cpu_inference_times = []
    
    for run in range(runs):
        start = time.perf_counter()
        batch_cpu = preprocess_pil_cpu(frames)
        cpu_preprocess_times.append((time.perf_counter() - start) * 1000)
        
        batch_tensor = torch.from_numpy(batch_cpu).cuda()
        
        start = time.perf_counter()
        results_cpu = run_yolo(model, batch_tensor)
        cpu_inference_times.append((time.perf_counter() - start) * 1000)
    
    cpu_preprocess = np.median(cpu_preprocess_times)
    cpu_inference = np.median(cpu_inference_times)
    cpu_total = cpu_preprocess + cpu_inference
    
    det_cpu, by_class = count_detections(results_cpu)
    
    print(f"  Preprocess: {cpu_preprocess:.2f} ms")
    print(f"  Inference: {cpu_inference:.2f} ms")
    print(f"  Total: {cpu_total:.2f} ms")
    print(f"  Detections: {det_cpu}")
    
    # ========================================
    # GPU Block Metadata (Your Approach)
    # ========================================
    print("\n[GPU: Block Metadata Resize]")
    print("  (Your approach - O(1) lookup)")
    
    gpu_preprocess_times = []
    gpu_inference_times = []
    
    for run in range(runs):
        cuda.synchronize()
        start = time.perf_counter()
        batch_gpu = preprocess_block_metadata_gpu(frames, metadata)
        cuda.synchronize()
        gpu_preprocess_times.append((time.perf_counter() - start) * 1000)
        
        batch_tensor = torch.from_numpy(batch_gpu).cuda()
        
        start = time.perf_counter()
        results_gpu = run_yolo(model, batch_tensor)
        gpu_inference_times.append((time.perf_counter() - start) * 1000)
    
    gpu_preprocess = np.median(gpu_preprocess_times)
    gpu_inference = np.median(gpu_inference_times)
    gpu_total = gpu_preprocess + gpu_inference
    
    det_gpu, _ = count_detections(results_gpu)
    
    print(f"  Preprocess: {gpu_preprocess:.2f} ms")
    print(f"  Inference: {gpu_inference:.2f} ms")
    print(f"  Total: {gpu_total:.2f} ms")
    print(f"  Detections: {det_gpu}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Method':<30} {'Preprocess':>12} {'Inference':>12} {'Total':>12}")
    print("-" * 70)
    print(f"{'CPU (PIL)':<30} {cpu_preprocess:>11.2f}ms {cpu_inference:>11.2f}ms {cpu_total:>11.2f}ms")
    print(f"{'GPU (Block Metadata)':<30} {gpu_preprocess:>11.2f}ms {gpu_inference:>11.2f}ms {gpu_total:>11.2f}ms")
    
    preprocess_speedup = cpu_preprocess / gpu_preprocess
    total_speedup = cpu_total / gpu_total
    
    print(f"\nSpeedup (Block Metadata vs CPU):")
    print(f"  Preprocess: {preprocess_speedup:.2f}x faster")
    print(f"  Total pipeline: {total_speedup:.2f}x faster")
    
    print(f"\nCorrectness:")
    print(f"  CPU detections: {det_cpu}")
    print(f"  GPU detections: {det_gpu}")
    if det_cpu == det_gpu:
        print("  PASS")
    else:
        print(f"  MISMATCH (diff: {abs(det_cpu - det_gpu)})")
    
    print(f"\nTop detections:")
    for cls_name, count in sorted(by_class.items(), key=lambda x: -x[1])[:5]:
        print(f"  {cls_name}: {count}")
    
    return {
        'cpu': {'preprocess': cpu_preprocess, 'inference': cpu_inference, 'total': cpu_total},
        'gpu': {'preprocess': gpu_preprocess, 'inference': gpu_inference, 'total': gpu_total},
        'speedup': {'preprocess': preprocess_speedup, 'total': total_speedup}
    }


# ============================================
# MAIN
# ============================================

def main():
    print("CUDA check...")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ERROR: CUDA not available")
        return
    
    print("\nLoading frames...")
    frames, metadata = load_frames_from_folders(max_per_res=50)
    
    if len(frames) == 0:
        print("No frames found.")
        return
    
    run_benchmark(frames, metadata)
    
    print("\nDone")


if __name__ == "__main__":
    main()