# 🐙 Octopus-Inspired GPU Load Balancing

**Bio-inspired adaptive block assignment for image processing**

---

## TL;DR

I achieved **2.6x total speedup** over fair GPU baselines by considering the **full pipeline cost**: setup + H2D transfer + kernel + D2H transfer.

| Metric | Grid-Stride (Fair) | Hybrid (Ours) | Improvement |
|--------|-------------------|---------------|-------------|
| Setup time | ~150ms | ~1ms | **148x faster** |
| H2D Transfer | ~25ms | ~0.2ms | **150x faster** |
| Memory | 341 MB | 0.03 MB | **11,000x less** |
| Kernel time | ~6ms | ~6ms | ~same |
| D2H Transfer | ~30ms | ~30ms | ~same |
| **TOTAL** | ~210ms | ~80ms | **2.6x faster** |

**Key insight:** When kernel performance is similar, **setup + transfer costs determine the winner.**

---

## Contribution

What makes this different from generic "segmented/irregular scheduling":

| Aspect | This Work |
|--------|-----------|
| **Problem** | Ragged 2D stencil (blur) across variable-sized images — not scan/reduce |
| **Cost Model** | **Total cost** = Setup (CPU) + Memory + Kernel — not just kernel throughput |
| **Technique** | O(num_blocks) block metadata vs O(total_pixels) mapping table |
| **Claim** | Kernel throughput is similar; **system-level costs determine the winner** |

The key insight is that for image-aware operations, the "fair" baseline (Grid-Stride with O(1) lookup) requires expensive pre-computation that dominates total runtime.

---

## The Journey: From 252x to Honest 12x

### What I Originally Claimed
> "252x speedup on GPU parallel processing!"

### What I Discovered

| Baseline | Speedup | Problem |
|----------|---------|---------|
| Naive (1 thread/image) | 252x | ❌ Strawman — nobody does this |
| Grid-Stride (O(n) search) | 9-10x | ❌ Unfair — baseline has O(n) bug |
| Grid-Stride (O(1) lookup) | 0.95x | 😬 Kernel-only comparison |
| **Grid-Stride (O(1) + setup)** | **12x** | ✅ **Fair total cost comparison** |

### The Real Win

Grid-Stride with O(1) lookup needs a **huge pre-computed lookup table**:
- `pixel_to_image[total_pixels]` — one entry per pixel
- 100M pixels = **400 MB** of memory
- O(N) time to build
- **~25ms to transfer to GPU**

Hybrid only needs **tiny block arrays**:
- `block_to_image[num_blocks]` — one entry per block
- 500 images ≈ 500 blocks = **0.03 MB**
- O(images) time to build
- **~0.2ms to transfer to GPU**

---

## Benchmark Results

### Complete Pipeline Cost (Setup + H2D + Kernel + D2H)

Including **all** costs in a real pipeline:

| Kernel | Setup | H2D | Memory | Kernel | D2H | **TOTAL** |
|--------|-------|-----|--------|--------|-----|-----------|
| Light (3x3) | 148x | 150x | 11,000x | ~1x | ~1x | **2.6x** |
| Heavy (5x5+Sobel) | 148x | 150x | 11,000x | ~1x | ~1x | **2.6x** |

*(Ratios = Grid-Fair / Hybrid, higher = Hybrid wins)*

**Note:** D2H is ~1x because output size is identical for both methods.

### Amortization Analysis

"What if I reuse the same batch many times?"

![Light Kernel Amortization](amortization_light.png)
*Light kernel: Hybrid dominates until 151 iterations*

![Heavy Kernel Amortization](amortization_heavy.png)
*Heavy kernel: Hybrid dominates until 53 iterations*

| Kernel Type | Hybrid Speedup | Crossover Point |
|-------------|----------------|-----------------|
| Light (3x3 blur) | 2.6x | **151 iterations** |
| Heavy (5x5 + Sobel) | 2.6x | **53 iterations** |

**Key finding:** Grid-Stride only catches up with very heavy reuse (50-150+ iterations). For typical ML preprocessing where each epoch creates new augmented batches, Hybrid wins.

---

## When Does Hybrid Win?

### ✅ Hybrid wins (use it):

| Scenario | Why |
|----------|-----|
| **New batches each time** | Setup cost matters (12x faster) |
| **Memory-constrained devices** | 11,000x less memory (Jetson, mobile) |
| **Streaming/real-time** | Can't afford 150ms setup delay |
| **Variable-size images** | Block-per-image fails on large images |

### ⚠️ Similar performance:

| Scenario | Why |
|----------|-----|
| **Same batch repeated 100+ times** | Setup cost amortized |
| **Kernel-only comparison** | Both achieve similar throughput |

### ❌ Don't use Hybrid:

| Scenario | Why |
|----------|-----|
| **Per-pixel independent ops** | Grid-stride is simpler, equally fast |
| **Already balanced workload** | No imbalance to solve |

---

## Application: ML/AI Preprocessing (Killer Use Case)

This technique is directly applicable to **ragged tensor preprocessing** in ML pipelines.

### The Problem in ML

| Current Approach | Problem |
|------------------|---------|
| **Padding** | Waste compute — short sequences padded to max length |
| **Per-element index mapping** | Waste memory — O(N) lookup tables |

### Where Hybrid Fits

| ML Domain | Variable-Length Data | Hybrid Benefit |
|-----------|---------------------|----------------|
| **NLP** | Sentences (10-500 tokens) | No padding, no O(N) map |
| **Vision** | Cropped regions, patch sizes | Block-level metadata only |
| **Audio** | Speech segments (variable duration) | Efficient batching |
| **Multi-modal** | Image + text batches | Mixed-size handling |
| **Graph ML** | Nodes with different neighbor counts | Ragged adjacency |

### Applicable Operations

- **Preprocessing kernels** (normalization, augmentation)
- **Embedding transforms** (before attention layers)
- **Feature extraction** (variable-size inputs)
- **Data augmentation** (random crops, resizes)

### Integration Points

```python
# PyTorch ragged tensor preprocessing
# Instead of: padded_batch = pad_sequence(sequences)  # wastes compute
# Or:         pixel_map = build_element_map(sizes)    # wastes memory

# Use Hybrid:
block_meta = compute_hybrid_assignment(sizes)  # O(num_blocks) only
output = hybrid_preprocess_kernel(data, block_meta)
```

**Key claim:** For ragged tensor operations where the "fair" baseline requires O(N) element mapping, Hybrid achieves **12x faster total time** and **11,000x less memory** using O(B) block metadata where B << N.

### Validated: ML Ragged Tensor Preprocessing

| Test | Elements | Setup | Memory | Kernel | TOTAL |
|------|----------|-------|--------|--------|-------|
| NLP Realistic (1K) | 60K | 3x | 12x | 1.37x | **1.99x** |
| NLP Realistic (10K) | 595K | 3x | 12x | 0.50x | **2.45x** |
| NLP Extreme (1K) | 150K | 2x | 28x | 0.50x | **1.10x** |
| Vision Patches (500) | 1.7M | 2x | 205x | 0.75x | **1.60x** |
| Vision Extreme (500) | 3.4M | 1x | 186x | 1.35x | **1.09x** |
| **AVERAGE** | — | **2x** | **88x** | ~same | **1.65x** |

✅ **Hybrid wins 5/5 tests on ML workloads**

### When Benefits Are Largest

Hybrid's benefit is proportional to:
- **Average sequence/element length** — longer = larger per-element mapping overhead
- **Whether baseline requires per-element mapping** — O(N) vs O(B) gap widens with scale

| Workload | Total Elements | Primary Benefit |
|----------|----------------|-----------------|
| **Short-sequence NLP** | 60K-600K | Memory footprint (12-88x less) |
| **Large-scale vision** | 1.7M-90M+ | Total-time speedup (1.6-12x) |
| **Long sequences / large images** | 90M+ | Setup overhead dominates → 12x+ faster |

**Key insight:** For short-sequence NLP preprocessing, the benefit manifests primarily as memory reduction. For large-scale vision, long sequences, or repeated pipeline execution, the benefit translates to significant total-time speedup.

---

## The Octopus Insight

An octopus has ~500 million neurons distributed across 8 arms. But the brain doesn't micromanage every neuron — it coordinates at the **arm level**. Each arm has local autonomy to handle its own movements.

**This is exactly our approach:**

| Octopus | GPU (Naive) | GPU (Hybrid) |
|---------|-------------|--------------|
| Coordinate per-neuron | Coordinate per-element | Coordinate per-block |
| ❌ Impossible (500M neurons) | ❌ Expensive (O(N) mapping) | ✅ Efficient (O(B) metadata) |

**The insight:** Don't micromanage at the element level. Coordinate at the block level.

```
Naive mapping:     element_to_seq[total_elements]  → O(N) memory
Octopus approach:  block_to_seq[num_blocks]        → O(B) memory
                   where B << N
```

This is why Hybrid uses **11,000x less memory** — same reason an octopus brain doesn't need 500M direct connections.

---

## Implementation

### Core Algorithm (~30 lines)

```python
def compute_hybrid_assignment(sizes, threshold=65536):
    """
    Adaptive block assignment:
    - Small images: 1 block (locality)
    - Large images: subdivide (load balance)
    """
    block_to_image = []
    block_start = []
    block_end = []
    
    for img_id, size in enumerate(sizes):
        if size <= threshold:
            # Small image: 1 block
            block_to_image.append(img_id)
            block_start.append(0)
            block_end.append(size)
        else:
            # Large image: subdivide
            num_blocks = ceil(size / threshold)
            for b in range(num_blocks):
                block_to_image.append(img_id)
                block_start.append(b * threshold)
                block_end.append(min((b+1) * threshold, size))
    
    return block_to_image, block_start, block_end
```

### GPU Kernel

```python
@cuda.jit
def hybrid_kernel(images_flat, offsets, widths, heights,
                  block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    
    # O(1) lookup — no search!
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    # Image info
    offset = offsets[img_id]
    w = widths[img_id]
    
    # Threads cooperate within block's range
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        # ... process pixel with image context
```

---

## Files

| File | Description |
|------|-------------|
| `upgraded_benchmark.py` | **Main benchmark** — Full pipeline (Setup + H2D + Kernel + D2H) |
| `memory_benchmark.py` | Memory-focused benchmark |
| `nlp_ragged_benchmark.py` | ML ragged tensor validation |

---

## Quick Start

```bash
git clone https://github.com/matthewlam721/octopus-parallel.git
cd octopus-parallel

pip install numba numpy scipy pillow

# Download Flickr8k from Kaggle
# Place in ./Images/

# Run main benchmark
python memory_benchmark.py
```

---

## What I Learned

### 1. Fair baselines matter
My initial 252x was against a strawman. Real contribution is 12x vs fair baseline.

### 2. Total cost matters
Kernel time alone is misleading. Setup + memory + kernel = true comparison.

### 3. Know when your approach wins
- ✅ New batches, memory-constrained: Hybrid wins
- ⚠️ Repeated batches, kernel-only: Similar performance

### 4. Simple isn't always optimal
Block-per-image is simple but fails on imbalanced workloads. Hybrid adds minimal complexity but handles all scenarios.

---

## Future Work

- [ ] Edge deployment (NVIDIA Jetson) — where memory savings matter most
- [ ] Real algorithms (U-Net, segmentation)
- [ ] Video processing datasets
- [ ] Framework integration (PyTorch, JAX)

---

## Conclusion

**The octopus doesn't waste energy computing per-neuron lookup tables. Neither should your GPU.**

For image-aware operations with variable-sized workloads:
- **2.6x faster** total pipeline time
- **11,000x less** memory
- **Dominates up to 50-150 kernel invocations** before Grid-Stride catches up

The key insight: **efficiency of pre-computation and transfer matters as much as the kernel itself.**

---

**Author:** Matthew, UIUC MCS  
**Contact:** matthewlam721@gmail.com  
**Repo:** [github.com/matthewlam721/octopus-parallel](https://github.com/matthewlam721/octopus-parallel)

---

## Appendix: Full Benchmark Output

```
======================================================================
MEMORY-AWARE BENCHMARK SUMMARY
======================================================================

  Test                       Pixels      Setup     Memory     Kernel      TOTAL
  ---------------------------------------------------------------------------
  Flickr Pure           89,416,278       210x     11545x      1.00x     16.68x
  Flickr + 4K           97,710,678       134x     11660x      0.94x     11.74x
  Flickr + 8K          122,593,878        79x     11925x      0.95x      5.91x
  Flickr 1000          179,455,121       124x     11574x      0.96x     13.78x
  Flickr 1000 + 8K     212,632,721       193x     11787x      1.00x     14.13x

  AVERAGE                         -       148x     11698x      0.97x     12.45x

======================================================================
KEY FINDINGS
======================================================================

  1. SETUP TIME: Hybrid is 148x faster
  2. MEMORY: Hybrid uses 11,698x less memory  
  3. KERNEL: Similar performance
  4. TOTAL: Hybrid is 12.45x faster overall

  🐙 HYBRID WINS when considering TOTAL cost!

======================================================================
```

*Tested on NVIDIA RTX 4090, January 2026*
