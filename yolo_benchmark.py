"""
YOLO Object Detection Benchmark
===============================
Benchmark object detection on variable-size images.

Measures:
- Preprocessing time (resize/normalize)
- Inference time
- Detection accuracy (mAP)
- Different image sizes

Usage:
    pip install ultralytics
    python yolo_benchmark.py
"""

import numpy as np
import time
from pathlib import Path
from PIL import Image
import torch

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
            print(f"{res}: not found, skipping")
            continue
        
        frame_files = sorted(res_dir.glob('frame_*.jpg'))[:max_per_res]
        
        for f in frame_files:
            img = Image.open(f).convert('RGB')
            arr = np.array(img)
            frames.append(arr)
            metadata.append({
                'resolution': res,
                'width': arr.shape[1],
                'height': arr.shape[0],
                'pixels': arr.shape[0] * arr.shape[1],
                'path': str(f)
            })
        
        print(f"{res}: loaded {len(frame_files)} frames")
    
    return frames, metadata


# ============================================
# PREPROCESSING APPROACHES
# ============================================

def preprocess_naive(frames, target_size=640):
    """
    Naive preprocessing: resize each image individually with PIL.
    This is what most people do.
    """
    processed = []
    for frame in frames:
        img = Image.fromarray(frame)
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.array(img_resized).astype(np.float32) / 255.0
        # HWC -> CHW for PyTorch
        arr = arr.transpose(2, 0, 1)
        processed.append(arr)
    
    return np.stack(processed)


def preprocess_batched_torch(frames, target_size=640):
    """
    Batched preprocessing using PyTorch GPU resize.
    More efficient for GPU pipelines.
    """
    import torch.nn.functional as F
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed = []
    
    for frame in frames:
        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)
        
        # Resize on GPU
        resized = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
        processed.append(resized)
    
    # Stack all
    batch = torch.cat(processed, dim=0)
    return batch


# ============================================
# YOLO INFERENCE
# ============================================

def run_yolo_inference(model, images, batch_size=16):
    """
    Run YOLO inference on preprocessed images.
    Returns detections and timing.
    """
    results = []
    total_time = 0
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        start = time.perf_counter()
        batch_results = model(batch, verbose=False)
        torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        
        results.extend(batch_results)
    
    return results, total_time


def count_detections(results):
    """Count total detections across all images."""
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
    """Run full benchmark."""
    
    print("\n" + "=" * 70)
    print("YOLO OBJECT DETECTION BENCHMARK")
    print("=" * 70)
    
    print(f"\nDataset: {len(frames)} images")
    sizes = [m['pixels'] for m in metadata]
    print(f"Size range: {min(sizes):,} - {max(sizes):,} pixels")
    print(f"Imbalance: {max(sizes) / min(sizes):.2f}x")
    
    # Count by resolution
    print("\nBy resolution:")
    for res in ['180p', '480p', '720p', '1080p']:
        count = sum(1 for m in metadata if m['resolution'] == res)
        if count > 0:
            print(f"  {res}: {count} images")
    
    # Load YOLO model
    print("\nLoading YOLOv8 model...")
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # nano model, fastest
    model.to('cuda')
    
    # Warmup
    print("Warming up...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        _ = model(dummy, verbose=False)
    torch.cuda.synchronize()
    
    results_summary = {}
    
    # ========================================
    # Test 1: Naive preprocessing (PIL)
    # ========================================
    print("\n[Method 1: Naive PIL Preprocessing]")
    
    preprocess_times = []
    inference_times = []
    
    for run in range(runs):
        # Preprocess
        start = time.perf_counter()
        processed = preprocess_naive(frames)
        preprocess_time = (time.perf_counter() - start) * 1000
        preprocess_times.append(preprocess_time)
        
        # Inference (use raw frames for YOLO, it handles preprocessing)
        start = time.perf_counter()
        yolo_results, _ = run_yolo_inference(model, [Image.fromarray(f) for f in frames])
        torch.cuda.synchronize()
        inference_time = (time.perf_counter() - start) * 1000
        inference_times.append(inference_time)
    
    naive_preprocess = np.median(preprocess_times)
    naive_inference = np.median(inference_times)
    naive_total = naive_preprocess + naive_inference
    
    total_detections, by_class = count_detections(yolo_results)
    
    print(f"  Preprocess: {naive_preprocess:.2f} ms")
    print(f"  Inference: {naive_inference:.2f} ms")
    print(f"  Total: {naive_total:.2f} ms")
    print(f"  Detections: {total_detections}")
    
    results_summary['naive'] = {
        'preprocess_ms': naive_preprocess,
        'inference_ms': naive_inference,
        'total_ms': naive_total,
        'detections': total_detections
    }
    
    # ========================================
    # Test 2: PyTorch GPU preprocessing
    # ========================================
    print("\n[Method 2: PyTorch GPU Preprocessing]")
    
    preprocess_times = []
    inference_times = []
    
    for run in range(runs):
        # Preprocess on GPU
        start = time.perf_counter()
        torch.cuda.synchronize()
        processed = preprocess_batched_torch(frames)
        torch.cuda.synchronize()
        preprocess_time = (time.perf_counter() - start) * 1000
        preprocess_times.append(preprocess_time)
        
        # Inference
        start = time.perf_counter()
        yolo_results, _ = run_yolo_inference(model, [Image.fromarray(f) for f in frames])
        torch.cuda.synchronize()
        inference_time = (time.perf_counter() - start) * 1000
        inference_times.append(inference_time)
    
    torch_preprocess = np.median(preprocess_times)
    torch_inference = np.median(inference_times)
    torch_total = torch_preprocess + torch_inference
    
    print(f"  Preprocess: {torch_preprocess:.2f} ms")
    print(f"  Inference: {torch_inference:.2f} ms")
    print(f"  Total: {torch_total:.2f} ms")
    print(f"  Detections: {total_detections}")
    
    results_summary['torch'] = {
        'preprocess_ms': torch_preprocess,
        'inference_ms': torch_inference,
        'total_ms': torch_total,
        'detections': total_detections
    }
    
    # ========================================
    # Test 3: By image size
    # ========================================
    print("\n[Analysis by Image Size]")
    
    for res in ['180p', '480p', '720p', '1080p']:
        res_frames = [f for f, m in zip(frames, metadata) if m['resolution'] == res]
        if not res_frames:
            continue
        
        start = time.perf_counter()
        res_results = model([Image.fromarray(f) for f in res_frames], verbose=False)
        torch.cuda.synchronize()
        res_time = (time.perf_counter() - start) * 1000
        
        res_detections, _ = count_detections(res_results)
        avg_time = res_time / len(res_frames)
        
        print(f"  {res}: {len(res_frames)} images, {res_time:.2f} ms total, {avg_time:.2f} ms/img, {res_detections} detections")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Preprocess':>12} {'Inference':>12} {'Total':>12}")
    print("-" * 65)
    print(f"{'Naive (PIL)':<25} {results_summary['naive']['preprocess_ms']:>11.2f}ms {results_summary['naive']['inference_ms']:>11.2f}ms {results_summary['naive']['total_ms']:>11.2f}ms")
    print(f"{'PyTorch GPU':<25} {results_summary['torch']['preprocess_ms']:>11.2f}ms {results_summary['torch']['inference_ms']:>11.2f}ms {results_summary['torch']['total_ms']:>11.2f}ms")
    
    speedup = results_summary['naive']['preprocess_ms'] / results_summary['torch']['preprocess_ms']
    print(f"\nPreprocess speedup (GPU vs CPU): {speedup:.2f}x")
    
    print(f"\nTotal detections: {total_detections}")
    print(f"Top classes detected:")
    for cls_name, count in sorted(by_class.items(), key=lambda x: -x[1])[:5]:
        print(f"  {cls_name}: {count}")
    
    return results_summary


# ============================================
# MAIN
# ============================================

def main():
    # Check CUDA
    print("Checking CUDA...")
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: CUDA not available, will be slow")
    
    # Load frames
    print("\nLoading frames...")
    frames, metadata = load_frames_from_folders(max_per_res=50)
    
    if len(frames) == 0:
        print("No frames found. Run video_frame_extract.py first.")
        return
    
    # Run benchmark
    run_benchmark(frames, metadata)
    
    print("\nDone")


if __name__ == "__main__":
    main()