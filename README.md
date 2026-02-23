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

## Edge Simulation: Satellite & Drone

Okay so the benchmarks above are synthetic workloads. What happens in actual edge scenarios where there's literally no cloud option?

### Satellite Onboard Filtering

Setup: 8192×8192 satellite image (typical Earth observation), cut into 727 variable-size tiles (128-512px). Synthetic image with ocean (dark), vegetation (medium), and urban areas (bright). GPU runs normalize + threshold on all tiles in a single kernel launch, decides which tiles are worth downlinking.

Downlink: 2 Mbps (typical LEO satellite).

```
Without filtering:
  192 MB → 805 seconds to downlink

With Octopus filtering (147ms processing):
  727 tiles → keep 274 (38%) → 74 MB → 312 seconds

Bandwidth saved: 61%
Pipeline speedup: 2.6x
Processing overhead: 147ms (0.05% of total pipeline time)
```

The 147ms is basically free. The bottleneck is always the downlink, never the processing. Even if the GPU kernel was 10x slower it wouldn't matter — you're trading milliseconds of compute for minutes of bandwidth.

For a satellite doing continuous imaging (dozens of captures per orbit), this compounds to hours of saved downlink per day.

### Drone Real-Time Classification

Setup: 1920×1080 @ 30fps surveillance drone. Object detector finds 25-28 bounding boxes per frame (variable size, 20-300px). Each detection gets cropped and resized to 224×224 for a classifier. Frame budget: 33.3ms.

```
              Total (10s video)    Per-frame    Budget used
CPU OpenCV        3942ms            13.1ms         39%
Individual CUDA   2080ms             6.9ms         21%
Octopus           1454ms             4.8ms         14%
```

All three make real-time, but the point isn't just "can it keep up" — it's how much headroom you leave for the rest of the pipeline (YOLO + classification + tracking + decision logic). Octopus uses 14% of the frame budget on preprocessing, leaving 86% for everything else.

Scales linearly too: 738 detections (1s), 3985 (5s), 8471 (10s) — per-frame time stays at 4.3-4.8ms.

## Power Efficiency on Edge

If you're running on a drone or satellite, speed is only half the story. The other half is battery. So I measured actual power draw on Jetson using the onboard INA3221 sensors.

Setup: 4K frame (3840×2160), variable-size crops (16-320px, drone/satellite detection distribution), sampled power every 2ms during sustained load.

| Objects | Octopus (ms) | Individual (ms) | Speedup | Octopus (mJ) | Individual (mJ) | Energy Saved |
|---------|-------------|----------------|---------|-------------|----------------|-------------|
| 50 | 11.8 | 12.6 | 1.1x | 58.2 | 62.2 | +6.5% |
| 100 | 21.2 | 24.7 | 1.2x | 104.0 | 121.8 | +14.6% |
| 200 | 40.2 | 49.3 | 1.2x | 197.3 | 242.5 | +18.6% |
| 500 | 100.5 | 122.5 | 1.2x | 493.6 | 602.8 | +18.1% |
| 1000 | 198.5 | 244.8 | 1.2x | 975.2 | 1204.7 | +19.0% |
| 2000 | 396.4 | 489.3 | 1.2x | 1947.1 | 2405.6 | +19.1% |
| 3000 | 594.3 | 733.9 | 1.2x | 2919.6 | 3605.4 | +19.0% |
| 5000 | 993.0 | 1221.9 | 1.2x | 4877.3 | 6002.8 | +18.7% |

Idle power: ~4994 mW. Both methods draw about the same wattage (~4950 mW) during execution. The energy saving comes from Octopus finishing faster — GPU goes back to idle sooner.

At 18 objects (like the surgical toolkit VR test I ran for a collaboration), there's zero measurable difference. The GPU barely wakes up. But at 100+ objects, the pattern is clear and consistent: **~19% less energy per frame**, plateauing from 200 objects onwards.

19% less energy per frame means a drone flies 19% longer on the same battery, or a satellite squeezes 19% more inference per orbit. For something that's literally unreachable once it's launched, that matters.

## Real-World Data Distributions

All my benchmarks above use uniform random crop sizes. A reviewer would rightfully ask: "what about real workloads?" Real detections aren't uniform — drone footage is 80% tiny distant objects with a few big ones up close. Satellite tiles cluster around two sizes. Video frames barely change between consecutive frames.

So I tested three realistic distributions against uniform as baseline. 4K frame, 1000 objects.

**Padding waste — this is the killer:**

| Distribution | Actual Data | Padded to Max | Waste | Octopus Metadata |
|---|---|---|---|---|
| Uniform | 204.4 MB | 750.0 MB | 73% | 31.2 KB |
| Long-tail (drone) | 17.8 MB | 456.6 MB | **96%** | 31.2 KB |
| Bimodal (satellite) | 262.2 MB | 750.0 MB | 65% | 31.2 KB |

Long-tail is brutal. 80% of your crops are 16-60px, but a handful of 400px detections force the entire batch to pad to 400. So you're transferring 456MB to process 17.8MB of actual data. 96% of your GPU memory bandwidth is literally processing zeros.

Octopus doesn't care. 31.2KB of metadata regardless of distribution.

**Speed across distributions (Jetson, 1000 objects):**

Speedup is 1.19-1.20x across all distributions. Doesn't matter if crops are uniform, long-tail, or bimodal — the advantage comes from kernel launch overhead (1 launch vs 1000), not crop sizes. This isn't a best-case benchmark artifact.

**Temporal coherence (video, 500 detections, 10 consecutive frames):**

Frame-to-frame bbox jitter of ±5px. Both methods show near-zero variance (±0.06ms for Octopus, ±0.02ms for individual). No meaningful difference here — but it confirms neither method degrades with temporal correlation.

The takeaway: uniform benchmarks actually *understate* the memory advantage. Real workloads with long-tail distributions make the padding problem worse, not better.

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
- **Disconnected edge (satellite, drone, rover)**: Only option that works under memory + bandwidth constraints
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
python edge_simulation.py         # Satellite & drone scenarios
python power_benchmark_paper.py   #Power efficiency (needs root for sensors) 
python3 distribution_benchmark.py    # Distribution impact (takes a while at 5000 objects)
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
- `edge_simulation.py` — Satellite tile filtering & drone real-time classification
- `power_benchmark_paper.py` — Power/energy measurement across object counts (paper data)
- `distribution_benchmark.py` — Uniform vs long-tail vs bimodal distributions

---

Tested on RTX 4090, T4 (Colab), and Jetson Orin Nano. The Jetson results surprised me — 3x speedup for simple ops, but basically nothing for blur. The TensorRT comparison was the other surprise: faster kernels don't mean faster pipelines when you're drowning in padding. The satellite sim drove it home — 147ms of GPU time saving 8 minutes of downlink. That's the kind of trade-off that matters on edge.

The power data was the cherry on top — 19% energy saving at scale, just from eliminating kernel launch overhead. On a battery that can't be recharged in orbit, that's not a nice-to-have.