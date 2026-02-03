import numpy as np
from numba import cuda
import time

num_images = 200000
sizes = np.random.randint(100, 500, num_images).astype(np.int64)
offsets = np.zeros(num_images + 1, dtype=np.int64)
offsets[1:] = np.cumsum(sizes)
total_pixels = int(offsets[-1])
images_flat = np.random.rand(total_pixels).astype(np.float32)

print(f"Images: {num_images:,}, Pixels: {total_pixels:,}")
print(f"Offset array: {offsets.nbytes / 1e6:.2f} MB")

@cuda.jit
def kernel_b(images, offsets, num_images, output):
    idx = cuda.grid(1)
    if idx >= len(images):
        return
    lo, hi = 0, num_images
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid + 1] <= idx:
            lo = mid + 1
        else:
            hi = mid
    output[idx] = images[idx] * 2.0

@cuda.jit
def kernel_c(images, block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    if block_id >= len(block_to_image):
        return
    start = block_start[block_id]
    end = block_end[block_id]
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    for i in range(start + tid, end, stride):
        output[i] = images[i] * 2.0

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
print(f"Blocks: {len(block_to_image):,}")

d_images = cuda.to_device(images_flat)
d_output = cuda.device_array(total_pixels, dtype=np.float32)
d_offsets = cuda.to_device(offsets)
d_b2i = cuda.to_device(block_to_image)
d_bs = cuda.to_device(block_start)
d_be = cuda.to_device(block_end)

blocks_b = (total_pixels + 255) // 256
kernel_b[blocks_b, 256](d_images, d_offsets, num_images, d_output)
kernel_c[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
cuda.synchronize()

times_b = []
for _ in range(3):
    start = time.perf_counter()
    kernel_b[blocks_b, 256](d_images, d_offsets, num_images, d_output)
    cuda.synchronize()
    times_b.append(time.perf_counter() - start)

times_c = []
for _ in range(3):
    start = time.perf_counter()
    kernel_c[len(block_to_image), 256](d_images, d_b2i, d_bs, d_be, d_output)
    cuda.synchronize()
    times_c.append(time.perf_counter() - start)

print(f"B (Search): {np.median(times_b)*1000:.2f} ms")
print(f"C (Block):  {np.median(times_c)*1000:.2f} ms")
print(f"C vs B: {np.median(times_b)/np.median(times_c):.2f}x")
