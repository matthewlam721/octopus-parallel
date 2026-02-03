import numpy as np
from numba import cuda
import time

num_images = 100000
sizes = np.random.randint(100, 500, num_images).astype(np.int64)
offsets = np.zeros(num_images + 1, dtype=np.int64)
offsets[1:] = np.cumsum(sizes)
total_pixels = int(offsets[-1])
images_flat = np.random.rand(total_pixels).astype(np.float32)

print(f"Images: {num_images:,}, Pixels: {total_pixels:,}")

# Different operations
@cuda.jit
def kernel_multiply(images, block_to_image, block_start, block_end, output):
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
def kernel_normalize(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = (images[i] - 0.5) * 2.0

@cuda.jit
def kernel_threshold(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = 1.0 if images[i] > 0.5 else 0.0

@cuda.jit
def kernel_gamma(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = images[i] ** 0.45  # Gamma correction

THREADS = 256
block_to_image = []
block_start = []
block_end = []
for img_id in range(num_images):
    s, e = int(offsets[img_id]), int(offsets[img_id + 1])
    for b in range((e - s + THREADS - 1) // THREADS):
        block_to_image.append(img_id)
        block_start.append(s + b * THREADS)
        block_end.append(min(s + (b + 1) * THREADS, e))

block_to_image = np.array(block_to_image, dtype=np.int32)
block_start = np.array(block_start, dtype=np.int64)
block_end = np.array(block_end, dtype=np.int64)

d_images = cuda.to_device(images_flat)
d_output = cuda.device_array(total_pixels, dtype=np.float32)
d_b2i = cuda.to_device(block_to_image)
d_bs = cuda.to_device(block_start)
d_be = cuda.to_device(block_end)

num_blocks = len(block_to_image)

kernels = [
    ("Multiply (x2)", kernel_multiply),
    ("Normalize", kernel_normalize),
    ("Threshold", kernel_threshold),
    ("Gamma (0.45)", kernel_gamma),
]

print(f"\nBenchmarking different operations:")
print("-" * 40)

for name, kernel in kernels:
    # Warmup
    kernel[num_blocks, THREADS](d_images, d_b2i, d_bs, d_be, d_output)
    cuda.synchronize()
    
    times = []
    for _ in range(5):
        start = time.perf_counter()
        kernel[num_blocks, THREADS](d_images, d_b2i, d_bs, d_be, d_output)
        cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    print(f"{name}: {np.median(times)*1000:.2f} ms")
