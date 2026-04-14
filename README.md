# 🐙 Octopus: Block-Level GPU Scheduling for Variable-Length Batches

I had a batch of 10,000 images, all different sizes. Wanted to process them on GPU without padding everything to the max size (wasteful). The obvious solution is to flatten them into one big array, but then... how does each GPU thread know which image it's working on?

Tried three approaches. Benchmarked on RTX 4090, T4, and Jetson Orin Nano — both with synthetic workloads and **real production data** (VisDrone-DET aerial footage, Sentinel-2 satellite imagery). Cache size matters way more than I expected, and the memory advantage is the real story on edge devices.

## 📋 Changelog

**2026-04-13 — Real data validation + sensor methodology corrected**

Two big updates:

1. **Power measurement was wrong.** The original benchmark sampled `/sys/class/hwmon/hwmon1/in1_input` believing it returned milliwatts. On Jetson Orin Nano this path returns voltage in millivolts — rail voltage rounded to ~4920 mV with negligible fluctuation. Energy savings computed against this signal collapsed to a function of execution time alone (which inflated to ~19% because it equaled the speedup ratio). After switching to `power = voltage × current` via INA3221 (`in*_input × curr*_input / 1000`), the corrected energy savings on real VisDrone data range from **+3% (50 objects) to +13.5% (5000 objects)**, plateauing around +12-13% for production-scale workloads. Time-domain results (speedup, ms/frame) are unaffected — those come from `time.perf_counter()` with `cuda.synchronize()`, not the power sensor. Old (buggy) data archived in `power_results_v1_buggy_sensor.csv`.

2. **Validated all major claims on real data**, not just synthetic distributions:
   - **VisDrone-DET val** (real drone footage, real bbox annotations) for the drone scenario — 2045 real bboxes pooled, median 39×56, P95 162×151
   - **Sentinel-2 L2A** (real Greenland satellite tile, 8192² UInt16 → 192MB) for satellite scenario
   - All numbers in tables below are from real data unless explicitly marked synthetic

## The Three Approaches

```
Flattened pixels:  [████ img0 ████|██ img1 ██|██████ img2 ██████|...]
                    ↑ pixel 12345 belongs to which image?
```

**A: Lookup Table** — Store `pixel_to_image[i]` for every pixel. Simple, but 500M pixels × 4 bytes = 2GB. Nope.

**B: Binary Search** — Just store where each image starts. Each thread does binary search to find its image. Tiny memory, but O(log n) per pixel and cache-dependent.

**C: Block Metadata** — Each CUDA block knows which image it handles. O(1) lookup per block, not per thread. Small memory, deterministic access pattern.

## The Results

On my 4090 (72MB L2 cache), B and C were basically the same. Everything fits in cache, binary search is free.

On T4 (4MB L2), C started winning by 22-28%. Cache pressure is real.

On Jetson (4MB L2, but 3x less memory bandwidth)... C crushed B by 2.5-3.5x. Those 17 binary search lookups per pixel really add up when memory is slow.

| GPU | L2 Cache | Memory BW | C vs B |
|-----|----------|-----------|--------|
| RTX 4090 | 72 MB | 1 TB/s | ~same |
| T4 | 4 MB | 320 GB/s | 1.2-1.3x |
| Jetson Orin Nano | 4 MB | 102 GB/s | **2.5-3.5x** |

## Per-Stage Timing Breakdown (Real VisDrone)

To attribute the speedup to specific pipeline stages, I instrumented both methods with `cuda.synchronize()`-bracketed `time.perf_counter()` measurements (30 reps per stage after warmup) on real VisDrone bboxes:

| Stage                              | n=100      | n=500      | n=1000     | n=2000     | Scaling          |
|------------------------------------|-----------:|-----------:|-----------:|-----------:|------------------|
| **Octopus**                        |            |            |            |            |                  |
| 1. Build metadata (host)           | 0.098 ms   | 0.495 ms   | 1.000 ms   | 1.985 ms   | O(N), 1 µs/obj   |
| 2. H2D metadata transfer           | 0.761 ms   | 0.468 ms   | 0.452 ms   | 0.460 ms   | O(1), ~0.5 ms    |
| 3. Crop+resize kernel              | 17.997 ms  | 86.612 ms  | 172.165 ms | 344.158 ms | O(N pixels)      |
| 4. Normalize kernel                | 2.988 ms   | 13.449 ms  | 26.519 ms  | 54.220 ms  | O(N pixels)      |
| **Octopus overhead (1+2)**         | **0.86 ms**| **0.96 ms**| **1.45 ms**| **2.45 ms**| < 1% of e2e      |
| Octopus total e2e                  | 21.4 ms    | 100.9 ms   | 200.0 ms   | 399.5 ms   |                  |
| **Individual** (N kernel launches) | 24.5 ms    | 122.4 ms   | 244.6 ms   | 488.5 ms   |                  |
| Per-launch overhead (isolated)     | 0.373 ms   | 0.371 ms   | 0.370 ms   | 0.371 ms   | constant ~370 µs |
| **Net kernel saving**              | **+3.5 ms**| **+22.3 ms**| **+45.9 ms**| **+90.1 ms**|                  |
| **E2E speedup**                    | 1.14x      | 1.21x      | 1.22x      | 1.22x      | Structural       |

**Octopus overhead is bounded.** Host metadata build is ~1 µs/object (Python for-loop), H2D is latency-bound (constant ~0.5 ms since metadata is only 16 KB at 2000 objects). Combined: ≤1% of e2e at production scale.

**Speedup source.** The amortized per-launch cost in a back-to-back sequence is ~46 µs (244 ms / 1000 launches − 199 ms / 1000 effective). At 1000 objects, 46 µs × 1000 = 46 ms gap accounts for the entire observed 45.9 ms net saving.

## Power Efficiency on Edge (Real VisDrone, INA3221 V×I)

Setup: real VisDrone drone footage as canonical 1080p frame, real bbox annotations sampled from the dataset (median 39×56, much smaller than uniform random would assume). Power computed as voltage × current on **VDD_IN rail** (total system) via Jetson INA3221 sensors, sampled every 2 ms during sustained load.

| Objects | Octopus (ms) | Individual (ms) | Speedup | Octopus (mW) | Individual (mW) | Octopus (mJ) | Individual (mJ) | Energy Saved |
|--------:|-------------:|----------------:|--------:|-------------:|----------------:|-------------:|----------------:|-------------:|
|     50  |      11.4    |      12.4       |  1.08x  |     8144     |     7759        |       93     |       96        |    +3.3%     |
|    100  |      20.9    |      24.5       |  1.18x  |     8500     |     7911        |      177     |      194        |    +8.6%     |
|    200  |      40.0    |      48.9       |  1.22x  |     8556     |     7932        |      342     |      388        |   +11.9%     |
|    500  |     100.2    |     122.4       |  1.22x  |     8663     |     7960        |      868     |      974        |   +10.8%     |
|   1000  |     198.6    |     244.0       |  1.23x  |     8707     |     8019        |     1729     |     1956        |   +11.6%     |
|   2000  |     397.4    |     487.5       |  1.23x  |     8666     |     8099        |     3443     |     3947        |   +12.8%     |
|   3000  |     595.6    |     733.4       |  1.23x  |     8685     |     8090        |     5172     |     5933        |   +12.8%     |
|   5000  |     992.0    |    1216.7       |  1.23x  |     8686     |     8192        |     8616     |     9965        |   +13.5%     |

Idle: 5730 mW (±29). All numbers from real GPU rail VDD_IN measurements.

### The interesting part: it's a Pareto trade-off, not a free lunch

Look at the power columns carefully. Octopus actually draws **5-9% more power** than the individual kernel approach during execution. Why? Because Octopus saturates the GPU continuously — one tight kernel doing real work the whole time. The individual approach has gaps between launches where the GPU partially idles, so its average wattage is lower.

But Octopus finishes 23% faster, and **energy = power × time**. The shorter execution window more than compensates for the higher instantaneous draw, netting **~12% lower energy per frame** at production scale, plateauing around 13%.

For battery-bound deployments this is the trade-off you want: total joules-per-frame determines flight time and orbit duration, not peak watts.

## TensorRT vs Octopus on Jetson (Real VisDrone)

Built a head-to-head for the crop+resize workload (1000 crops from VisDrone 1080p frame → 224×224 bilinear) and added VRAM measurement to capture the structural memory advantage.

**Uniform crops (256×256, TensorRT's best case):**

| Method | Kernel | E2E | VRAM | Notes |
|--------|--------|-----|------|-------|
| TensorRT | **25.4 ms** | 200.6 ms | **3369 MB** | 750 MB padded input, 0% waste |
| Octopus | 172.1 ms | 215.2 ms | 230 MB | ~31 KB metadata |

TensorRT kernel is 6.78x faster (NVIDIA hand-tuned vs my numba JIT). End-to-end only 1.07x faster because transferring 750 MB of padded data eats most of the gain. **Octopus uses 14.6x less VRAM.**

**Variable crops (real VisDrone bboxes, median 41×58, P95 162×151):**

| Method | Kernel | E2E | VRAM | Padding Waste |
|--------|--------|-----|------|---------------|
| TensorRT | **25.0 ms** | 244.1 ms | **2426 MB** | 95% (1118 MB padded float32) |
| Octopus | 172.2 ms | 215.6 ms | 296 MB | 0% (~31 KB metadata) |

TensorRT kernel still 6.89x faster in isolation. But **Octopus end-to-end is 1.13x faster** AND uses **8.2x less VRAM** because:
- 95% of TensorRT's compute is on padding (a 41×58 real bbox padded to 312×312 = 40× wasted pixels)
- 1.12 GB of float32 padding moves host→GPU per batch
- Peak VRAM (2.4 GB) is prohibitive on shared 8 GB devices where the GPU is also running model inference

This isn't a knock on TensorRT — it's built for neural network inference where inputs are uniform. Variable-size image processing is a different problem, especially under memory constraints.

## Edge Deployment: Satellite & Drone (Real Data)

### Satellite Onboard Filtering — Real Sentinel-2

Setup: real Sentinel-2 L2A B04 (red band) Greenland tile, 10980×10980 native, cropped to 8192×8192 for benchmark. Cut into 727 variable-size tiles (128-512px) via random grid. GPU runs normalize + threshold on all tiles in a single kernel launch, decides which tiles are worth downlinking.

Threshold tuned to 0.535 for Greenland scene (high albedo from ice/snow shifts the brightness distribution; synthetic threshold 0.35 was tuned for mixed ocean/vegetation/urban — adaptive thresholding is future work).

Downlink: 2 Mbps (typical LEO).

```
Without filtering:
  192 MB → 805 seconds (13.4 min) to downlink

With Octopus filtering (147 ms processing):
  727 tiles → keep 325 (45%) → 86 MB → 361 seconds

Bandwidth saved: 55%
Pipeline speedup: 2.2x
Processing overhead: 147ms (0.04% of total pipeline time)
```

The 147 ms is basically free. The bottleneck is always the downlink, never the processing. For a satellite doing continuous imaging, this compounds to hours of saved downlink per day.

### Drone Real-Time Classification — Real VisDrone

Setup: real VisDrone-DET val frames as canonical 1920×1080 input, 28 detections/frame avg sampled from real bbox pool (matches typical surveillance drone output). Each detection cropped and resized to 224×224 for classifier. Frame budget: 33.3 ms.

```
              Total (10s video)    Per-frame    Budget used
CPU OpenCV        2697 ms            9.0 ms        27%
Individual CUDA   2035 ms            6.8 ms        20%
Octopus           1437 ms            4.8 ms        14%
```

All three make real-time, but the point isn't just "can it keep up" — it's how much headroom you leave for the rest of the pipeline (YOLO + classification + tracking + decision logic). **Octopus uses 14% of the frame budget on preprocessing, leaving 86% for everything else.**

Per-frame Octopus time stays at 4.8 ms across 1s, 5s, and 10s batches (851 → 8347 detections), confirming linear scaling and no memory-bound degradation.

## Real-World Data Distributions

**Padding waste — this is the killer:**

| Distribution | Actual Data | Padded to Max | Waste | Octopus Metadata |
|---|---|---|---|---|
| Uniform | 204 MB | 750 MB | 73% | 31 KB |
| Long-tail (drone, real) | 18 MB | 457 MB | **96%** | 31 KB |
| Bimodal (satellite, real) | 262 MB | 750 MB | 65% | 31 KB |

Long-tail is brutal. 80% of real drone bboxes are 16-60px, but a handful of 312px detections force the entire batch to pad to 312². So you transfer 457 MB to process 18 MB of actual data. **96% of GPU memory bandwidth is processing zeros.**

Octopus doesn't care. 31 KB of metadata regardless of distribution.

## Honest Caveat: 3×3 Blur

For compute-heavy operations like 3×3 blur, B and C perform about the same on Jetson:

| Images | B (Search) | C (Block) | Speedup |
|--------|------------|-----------|---------|
| 50K | 28.6 ms | 28.5 ms | 1.00x |
| 100K | 57.2 ms | 56.5 ms | 1.01x |
| 150K | 85.4 ms | 84.1 ms | 1.02x |

The binary search overhead gets buried under the actual computation. Each pixel reads 9 neighbors, does math — the O(log n) lookup becomes negligible.

**Bottom line:** Block metadata wins big for memory-bound ops (multiply, normalize, threshold). For compute-bound ops (blur, convolution, gamma), it doesn't really matter which approach you use.

## T4 Numbers

Similar story, just smaller gains since it has 3× the memory bandwidth of Jetson.

- 100K images: 22% faster
- 500K images: 28% faster
- Real video frames: 22% faster

Also ran YOLO object detection on 200 frames. GPU preprocessing gave 2.9× end-to-end speedup (4384 ms → 1490 ms) with identical detection results.

## Auto-Tuner

Built a simple thing that runs two micro-benchmarks (multiply and blur) to decide whether block metadata is worth it on whatever hardware you're running. Takes <50 ms.

```
[Probe 1: Multiply (memory-bound)]
  B: 17.29 ms, C: 5.45 ms → 3.17x

[Probe 2: Blur (compute-bound)]
  B: 33.11 ms, C: 28.83 ms → 1.15x

→ Use BLOCK METADATA (C)
  Reason: Memory-bound ops benefit significantly
```

## When to Use What

- **Beefy GPU (4090, A100)**: Doesn't matter, pick whichever
- **Edge device + memory-bound ops**: Block metadata, definitely
- **Edge device + compute-bound ops**: Doesn't matter
- **Variable-size batches on edge**: Octopus over TensorRT — not because the kernel is faster (it isn't), but because of 8-15× less VRAM, 0% padding waste, and no per-batch engine rebuild
- **Disconnected edge (satellite, drone, rover)**: Only option that works under memory + bandwidth constraints
- **Need scheduling flexibility**: Block metadata is the only option anyway
- **Memory super tight**: Block metadata uses 8-15× less VRAM than TRT for variable-size workloads

## The Why

Binary search does O(log M) random memory accesses per pixel. With 100K images that's 17 lookups. On Jetson with 102 GB/s bandwidth, random access hurts.

Block metadata does O(1) lookup per block. Sequential access within each block. Cache-friendly, predictable.

But if you're doing heavy computation per pixel anyway (blur = 9 reads + math), that lookup overhead becomes noise.

## Running It

```bash
pip install numba numpy pillow opencv-python rasterio

# RTX 4090 / high-end GPU
python triple_baseline_benchmark.py --images 10000

# T4 (Google Colab)
python triple_baseline_benchmark.py --images 100000 --tiny

# Jetson — synthetic baselines (all work on 8GB shared memory, can't run approach A)
python test_multi_v2.py           # B vs C comparison
python test_real_scale.py         # Various scenarios
python test_bc_blur.py            # Blur comparison (spoiler: ~same)
python auto_tuner.py              # Hardware probe
python crop_resize_bilinear_benchmark.py  # ML preprocessing
python arena_benchmark.py         # TensorRT vs Octopus head-to-head
python edge_simulation.py         # Satellite & drone scenarios
python distribution_benchmark.py  # Distribution impact

# Jetson — REAL DATA (requires VisDrone + Sentinel-2 download)
# See satellite_loader.py / visdrone_loader.py for paths
python power_benchmark_real.py    # Power on real VisDrone
python edge_simulation_real.py    # Real drone + real satellite
python arena_benchmark_real.py    # TRT vs Octopus on real VisDrone + VRAM
python stage_timing_real.py       # Per-stage breakdown profiler
python make_stage_figure.py       # Generate paper figure (PNG + PDF)
```

## Files

### Benchmarks (synthetic)
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
- `power_benchmark.py` — Power/energy measurement (corrected sensor methodology)
- `distribution_benchmark.py` — Uniform vs long-tail vs bimodal distributions

### Benchmarks (real data) — added 2026-04-13
- `visdrone_loader.py` — Load VisDrone-DET val frames + bbox pool
- `satellite_loader.py` — Load Sentinel-2 L2A region + ROI generator
- `power_benchmark_real.py` — Power benchmark on real VisDrone
- `edge_simulation_real.py` — Real VisDrone drone + real Sentinel-2 satellite
- `arena_benchmark_real.py` — TRT vs Octopus on real VisDrone + VRAM tracking
- `stage_timing_real.py` — Per-stage breakdown profiler

### Figures
- `make_stage_figure.py` — Paper-ready matplotlib figure generator
- `figs/stage_breakdown.{png,pdf}` — Per-stage timing visualization

### Data
- `power_results.csv` — Synthetic baseline (corrected sensor)
- `power_results_real.csv` — Real VisDrone (paper-grade)
- `power_results_v1_buggy_sensor.csv` — Deprecated, archived for reference
- `stage_timing_real.csv` — Per-stage timings
- `distribution_results.csv` — Distribution analysis

---

Tested on RTX 4090, T4 (Colab), and Jetson Orin Nano. Real data validated on VisDrone-DET val (real drone bboxes) and Sentinel-2 L2A Greenland (real satellite imagery). The Jetson results surprised me — 3× speedup for simple ops, but basically nothing for blur. The TensorRT comparison was the other surprise: faster kernels don't mean faster pipelines when you're drowning in 1.1 GB of padding zeros and your GPU only has 8 GB shared with the rest of the system. The satellite sim drove it home — 147 ms of GPU time saving 7 minutes of downlink. That's the kind of trade-off that matters on edge.

The honest power data was the cherry on top: not the original 19% (which turned out to be a sensor bug), but a real **12% energy saving + 8× memory reduction** on real production data. On a battery that can't be recharged in orbit, every joule and every megabyte counts.
