import time
import numpy as np
import cv2
from numba import cuda, float32, int32, uint8

# ==========================================
#  CONFIGURATION
# ==========================================
NUM_CROPS = 1000       # Total number of crops to process
TARGET_W = 224
TARGET_H = 224
CHANNELS = 3
ITERATIONS = 50        # How many times to loop for the median benchmark

# ==========================================
#  THE OCTOPUS KERNEL
# ==========================================
@cuda.jit(fastmath=True)
def octopus_bilinear_kernel(src_flat, metadata, out_tensor):
    # 1. Grab the Task ID
    # One CUDA block handles one complete crop.
    task_id = cuda.blockIdx.x
    
    # Safety check: Prevent out-of-bounds access if blocks > tasks
    if task_id >= metadata.shape[0]: return

    # 2. Load Metadata (Global -> Registers)
    # We pull all necessary crop info into local registers for speed.
    src_offset = metadata[task_id, 0]
    src_w      = metadata[task_id, 1]
    src_h      = metadata[task_id, 2] # Needed for Y-axis clamping
    crop_x     = metadata[task_id, 3]
    crop_y     = metadata[task_id, 4]
    crop_w     = metadata[task_id, 5]
    crop_h     = metadata[task_id, 6]
    dst_idx    = metadata[task_id, 7]

    # Pre-calculate scaling factors (Dst -> Src mapping)
    scale_x = crop_w / float32(TARGET_W)
    scale_y = crop_h / float32(TARGET_H)

    # 3. Grid Stride Loop
    # Standard pattern to cover all pixels in the target 224x224 patch
    start = cuda.threadIdx.x
    stride = cuda.blockDim.x
    total_pixels = TARGET_W * TARGET_H

    # Calculate max valid indices for "Clamp-to-Edge" simulation
    max_x_idx = src_w - 1
    max_y_idx = src_h - 1

    for i in range(start, total_pixels, stride):
        # Map 1D index back to 2D target coordinates
        ty = i // TARGET_W
        tx = i % TARGET_W

        # Map Target(tx, ty) -> Source(gx, gy) in floating point
        gx = tx * scale_x + crop_x
        gy = ty * scale_y + crop_y

        # --- Coordinate Clamping (Texture Sampler Simulation) ---
        # We need integer coordinates (ix, iy). 
        # Crucial: We must ensure ix+1 and iy+1 don't read outside memory.
        # So we clamp (ix, iy) to (width-2, height-2) at max.
        
        ix = int32(gx)
        iy = int32(gy)
        
        # Manual clamping logic (faster than branching logic in some cases)
        if ix < 0: ix = 0
        elif ix >= max_x_idx: ix = max_x_idx - 1
        
        if iy < 0: iy = 0
        elif iy >= max_y_idx: iy = max_y_idx - 1

        # Calculate interpolation weights (the fractional part)
        fx = gx - ix
        fy = gy - iy

        # --- Memory Access ---
        # Calculate flat indices for the 2x2 grid
        # Row 0
        base_idx = src_offset + (iy * src_w + ix) * CHANNELS
        # Row 1 (Just one stride down)
        down_idx = base_idx + src_w * CHANNELS

        for c in range(CHANNELS):
            # Fetch the 4 neighbors (P00, P10, P01, P11)
            # Memory access is coalesced since 'tx' is consecutive
            p00 = float32(src_flat[base_idx + c])
            p10 = float32(src_flat[base_idx + CHANNELS + c])
            p01 = float32(src_flat[down_idx + c])
            p11 = float32(src_flat[down_idx + CHANNELS + c])
            
            # --- Bilinear Interpolation Math ---
            # Linear interp on X-axis first, then Y-axis
            top = p00 + (p10 - p00) * fx
            bot = p01 + (p11 - p01) * fx
            val = top + (bot - top) * fy
            
            # --- Correctness: Rounding & Clamping ---
            # 1. Round to nearest integer (+0.5 trick)
            val = val + 0.5
            # 2. Clamp to valid uint8 range [0, 255] to prevent overflow artifacts
            if val < 0: val = 0.0
            if val > 255: val = 255.0
            
            # Write back to output
            out_tensor[dst_idx, ty, tx, c] = uint8(val)

def run_benchmark():
    print(f"Octopus V4 Benchmark (Correctness + Events)")
    print(f" Config: {NUM_CROPS} crops | {TARGET_W}x{TARGET_H} | Bilinear Mode")
    
    # --- 1. Data Prep ---
    src_w, src_h = 4096, 4096
    # Metadata structure: [offset, src_w, src_h, x, y, w, h, dst_idx]
    meta_host = np.zeros((NUM_CROPS, 8), dtype=np.int32)
    cpu_crops = []
    
    print("Generating random test data...")
    for i in range(NUM_CROPS):
        w = np.random.randint(100, 500)
        h = np.random.randint(100, 500)
        # Ensure crop stays within source bounds
        x = np.random.randint(0, src_w - w - 1)
        y = np.random.randint(0, src_h - h - 1)
        
        meta_host[i, 0] = 0        
        meta_host[i, 1] = src_w
        meta_host[i, 2] = src_h
        meta_host[i, 3] = x
        meta_host[i, 4] = y
        meta_host[i, 5] = w
        meta_host[i, 6] = h
        meta_host[i, 7] = i
        
        # Keep a copy for CPU verification
        cpu_crops.append((x, y, w, h))

    # Create giant flat source image
    src_host = np.random.randint(0, 255, (src_h * src_w * 3), dtype=np.uint8)
    
    # Move data to GPU
    src_dev = cuda.to_device(src_host)
    meta_dev = cuda.to_device(meta_host)
    out_dev = cuda.device_array((NUM_CROPS, TARGET_H, TARGET_W, CHANNELS), dtype=np.uint8)

    # Launch params
    threads_per_block = 256
    blocks = NUM_CROPS

    # --- 2. GPU Benchmark (CUDA Events) ---
    print("\n Waking up the GPU (Warmup)...")
    # Run a few times to trigger JIT compilation
    for _ in range(10):
        octopus_bilinear_kernel[blocks, threads_per_block](src_dev, meta_dev, out_dev)
    cuda.synchronize()

    print(f"⏱ Running {ITERATIONS} timed iterations...")
    start_event = cuda.event()
    end_event = cuda.event()
    gpu_times = []

    for _ in range(ITERATIONS):
        start_event.record()
        octopus_bilinear_kernel[blocks, threads_per_block](src_dev, meta_dev, out_dev)
        end_event.record()
        end_event.synchronize() # Wait for the GPU to finish this run
        gpu_times.append(cuda.event_elapsed_time(start_event, end_event))

    median_gpu = np.median(gpu_times)
    p95_gpu = np.percentile(gpu_times, 95)
    
    print(f"✅ GPU Time (Median): {median_gpu:.3f} ms")
    print(f"✅ GPU Time (P95):    {p95_gpu:.3f} ms")
    print(f"   Per Crop: {median_gpu/NUM_CROPS:.4f} ms")

    # --- 3. CPU Benchmark (Fair Comparison) ---
    print("\n Running CPU Benchmark (OpenCV INTER_LINEAR)...")
    print("   (Only running 5 loops because CPU is painfully slow)")
    
    # Reshape for OpenCV
    cpu_src_img = src_host.reshape((src_h, src_w, 3))
    
    cpu_start = time.time()
    for _ in range(5): 
        for (x, y, w, h) in cpu_crops:
            roi = cpu_src_img[y:y+h, x:x+w]
            # FORCE LINEAR to make it an Apple-to-Apple comparison
            _ = cv2.resize(roi, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    cpu_end = time.time()
    
    avg_cpu = ((cpu_end - cpu_start) / 5.0) * 1000
    print(f"✅ CPU Time (Avg): {avg_cpu:.2f} ms")

    # --- 4. The Verdict ---
    speedup = avg_cpu / median_gpu
    print("-" * 30)
    print(f"🏆 FAIR Speedup (Bilinear vs Bilinear): {speedup:.2f}x")
    print("-" * 30)

if __name__ == "__main__":
    if cuda.is_available():
        run_benchmark()
    else:
        print("Error: No CUDA device found.")