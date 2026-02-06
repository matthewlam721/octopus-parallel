# 🐙 Octopus: Block-Level GPU Scheduling for Variable-Length Batches

I had a batch of 10,000 images, all different sizes. Wanted to process them on GPU without padding everything to the max size (wasteful). The obvious solution is to flatten them into one big array, but then... how does each GPU thread know which image it's working on?

Tried three approaches. Benchmarked on RTX 4090, T4, and Jetson Orin Nano. Turns out cache size matters way more than I expected.

## The Three Approaches
```
Flattened pixels:  [████ img0 ████|██ img1 ██|██████ img2 ██████|...]
                    ↑ pixel 12345 belongs to which image?
```

**A: Lookup Table** — Store `pixel_to_image[i]` for every pixel. Simple, but 500M pixels × 4 bytes = 2GB. Nope.

**B: Binary Search** — Just store where each image starts. Each thread does binary search to find its image. Tiny memory, but O(log n) per pixel and cache-dependent.

**C: Block Metadata** — Each CUDA block knows which image it handles. O(1) lookup per block, not per pixel. Small memory, deterministic access pattern.

## The Results

On my 4090 (72MB L2 cache), B and C were basically the same. Everything fits in cache, binary search is free.

On T4 (4MB L2), C started winning by 22-28%. Cache pressure is real.

On Jetson (4MB L2, but 3x less memory bandwidth)... C crushed B by 2.5-3.5x. Those 17 binary search lookups per pixel really add up when memory is slow.

| GPU | L2 Cache | Memory BW | C vs B |
|-----|----------|-----------|--------|
| RTX 4090 | 72 MB | 1 TB/s | ~same |
| T4 | 4 MB | 320 GB/s | 1.2-1.3x |
| Jetson Orin Nano | 4 MB | 102 GB/s | **2.5-3.5x** |

## Jetson Numbers

This is where it gets interesting. Ran a bunch of different tests:

**Scaling with image count:**
- 50K images: 3.45x faster
- 100K images: 2.59x faster  
- 250K images: 2.55x faster

**Real-world scenarios:**
- 100 actual photos (14K-2M pixels each): 1.48x
- 500 video frames: 1.91x
- 5K drone/satellite tiles: 2.75x
- 50K tiny patches: 3.45x

**Different operations:**
- Simple stuff (multiply, normalize, threshold): 2.5-4x speedup
- Compute-heavy stuff (blur, gamma): ~1x, the math dominates

**Crop & resize for ML preprocessing (bilinear, 1000 crops from 4K image):**
- CPU (OpenCV): 643 ms
- Individual CUDA kernels (1000 launches): 246 ms — kernel launch overhead adds up
- Octopus (single launch): 172 ms kernel, 243 ms end-to-end
- **3.7x faster than CPU, 1.43x faster than individual kernels**

The pattern: more images = deeper binary search = bigger win for block metadata. And memory-bound ops benefit way more than compute-bound ones.

## vs TensorRT on Jetson

Had to test this — TensorRT is NVIDIA's own inference optimizer. Built a head-to-head for the crop+resize workload (1000 crops from 4K → 224×224 bilinear).

**Uniform crops (256×256, TensorRT's best case):**

| Method | Kernel | End-to-End | Notes |
|--------|--------|------------|-------|
| TensorRT | **25 ms** | 216 ms | 750MB padded input, 0% waste |
| Octopus | 173 ms | 241 ms | ~4KB metadata |

TensorRT kernel is 6.8x faster. No surprise — NVIDIA's hand-tuned resize kernel vs my numba JIT. But end-to-end is only 1.1x faster because transferring 750MB of padded data eats most of the gain.

**Variable crops (32-512px, the real world):**

TensorRT with all 1000 crops in one batch? Engine build fails. 1000 × 3 × 512 × 512 × 4 bytes ≈ 3GB of padded float32. On an 8GB Jetson, nope.

So I split into batches of 100, each batch padded to its own max size:

| Method | Kernel | End-to-End | Padding Waste |
|--------|--------|------------|---------------|
| TensorRT (10 batches) | **55 ms** | 536 ms | 70%, 2.9GB total transfer |
| Octopus (single launch) | 172 ms | 243 ms | 0%, ~4KB metadata |

TensorRT kernel is still 3x faster. But **Octopus end-to-end is 2.2x faster** because:
- 70% of TensorRT's computation is on padding (a 32px crop padded to 512px = 256x wasted pixels)
- 2.9GB of data needs to move host→GPU across 10 batches
- 10 separate engine builds, 10 separate inference calls

This isn't a knock on TensorRT — it's built for neural network inference where inputs are uniform. Variable-size image processing is just a different problem.

## Honest Caveat: 3x3 Blur

For compute-heavy operations like 3x3 blur, B and C perform about the same on Jetson:

| Images | B (Search) | C (Block) | Speedup |
|--------|------------|-----------|---------|
| 50K | 28.6 ms | 28.5 ms | 1.00x |
| 100K | 57.2 ms | 56.5 ms | 1.01x |
| 150K | 85.4 ms | 84.1 ms | 1.02x |

The binary search overhead gets buried under the actual computation. Each pixel reads 9 neighbors, does math — the O(log n) lookup becomes negligible.

**Bottom line:** Block metadata wins big for memory-bound ops (multiply, normalize, threshold). For compute-bound ops (blur, convolution, gamma), it doesn't really matter which approach you use.

## T4 Numbers

Similar story, just smaller gains since it has 3x the memory bandwidth of Jetson.

- 100K images: 22% faster
- 500K images: 28% faster
- Real video frames: 22% faster

Also ran YOLO object detection on 200 frames. GPU preprocessing gave 2.9x end-to-end speedup (4384ms → 1490ms) with identical detection results.

## Auto-Tuner

Built a simple thing that runs two micro-benchmarks (multiply and blur) to decide whether block metadata is worth it on whatever hardware you're running. Takes <50ms.
```
[Probe 1: Multiply (memory-bound)]
  B: 17.29 ms, C: 5.45 ms → 3.17x

[Probe 2: Blur (compute-bound)]  
  B: 33.11 ms, C: 28.83 ms → 1.15x

→ Use BLOCK METADATA (C)
  Reason: Memory-bound ops benefit significantly
```

If multiply shows big speedup but blur doesn't, your workload is memory-bound → use C. Otherwise, doesn't matter.

## When to Use What

- **Beefy GPU (4090, A100)**: Doesn't matter, pick whichever
- **Edge device + memory-bound ops**: Block metadata, definitely
- **Edge device + compute-bound ops**: Doesn't matter
- **Variable-size batches on edge**: Octopus over TensorRT — no padding, no multi-engine juggling
- **Need scheduling flexibility**: Block metadata is the only option anyway
- **Memory super tight**: Binary search uses less memory (but slower)

## The Why

Binary search does O(log M) random memory accesses per pixel. With 100K images that's 17 lookups. On Jetson with 102 GB/s bandwidth, random access hurts.

Block metadata does O(1) lookup per block. Sequential access within each block. Cache-friendly, predictable.

But if you're doing heavy computation per pixel anyway (blur = 9 reads + math), that lookup overhead becomes noise.

## Running It
```bash
pip install numba numpy pillow

# RTX 4090 / high-end GPU
python triple_baseline_benchmark.py --images 10000

# T4 (Google Colab)
python triple_baseline_benchmark.py --images 100000 --tiny

# Jetson (8GB shared memory, can't run approach A)
python test_multi_v2.py           # B vs C comparison
python test_real_scale.py         # Real-world scenarios
python test_bc_blur.py            # Blur comparison (spoiler: ~same)
python auto_tuner.py              # Hardware probe
python crop_resize_bilinear_benchmark.py  # ML preprocessing
python arena_benchmark.py         # TensorRT vs Octopus head-to-head
python trt_batched_variable.py    # TensorRT batched variable-size
```

## Files

- `triple_baseline_benchmark.py` — Full A vs B vs C (needs >8GB GPU memory)
- `test_multi_v2.py` — B vs C only, works on Jetson
- `test_real_scale.py` — Different real-world scenarios
- `test_functions.py` — Different operations (multiply, blur, etc)
- `test_sizes.py` — Different image sizes
- `test_bc_blur.py` — Blur kernel comparison
- `auto_tuner.py` — Runtime hardware probe
- `crop_resize_bilinear_benchmark.py` — ML preprocessing benchmark
- `arena_benchmark.py` — TensorRT vs Octopus (uniform + variable crops)
- `trt_batched_variable.py` — TensorRT batched approach for variable sizes

---

Tested on RTX 4090, T4 (Colab), and Jetson Orin Nano. The Jetson results surprised me — 3x speedup for simple ops, but basically nothing for blur. The TensorRT comparison was the other surprise: faster kernels don't mean faster pipelines when you're drowning in padding.
