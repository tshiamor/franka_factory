# Cosmos-Transfer2.5 GPU Timing Guide

This document summarizes GPU configurations and timing estimates for running NVIDIA Cosmos-Transfer2.5 augmentation on the MCX Card Demos dataset.

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total videos | 430 (215 demos × 2 cameras) |
| Frame range | 289 - 448 frames |
| Average frames | 338 frames |
| FPS | 30 |
| Duration | 9-15 seconds per video |

### Frame Distribution

| Frame Range | Videos | Duration |
|-------------|--------|----------|
| 289-300 | 126 | ~9-10 sec |
| 300-400 | 228 | ~10-13 sec |
| 400-448 | 76 | ~13-15 sec |

---

## Understanding MAX_FRAMES

`MAX_FRAMES` controls how much of each video Cosmos processes:

```
Full demo (448 frames, ~15 sec):
[===GRASP===|===LIFT===|===MOVE===|===PLACE===|===RELEASE===]
     ↑                                                    ↑
   start                                                 end

MAX_FRAMES=100 (3.3 sec):
[===GRASP===|==LI...]
     ↑          ↑
   start       CUT OFF HERE
```

### Trade-offs

| Fewer Frames | More Frames |
|--------------|-------------|
| Faster inference | Slower inference |
| Less video context | Full motion captured |
| May miss end of action | Complete augmentation |

### Recommended MAX_FRAMES Values

| MAX_FRAMES | Captures | Use Case |
|------------|----------|----------|
| 100 | First 3.3 sec (grasp only) | Quick runs, grasp-focused learning |
| 200 | First 6.7 sec (grasp + lift) | Balanced speed/coverage |
| 338 | Average video length | Most videos complete |
| 448 | All videos complete | Full trajectory learning |

---

## Per-Inference Time by GPU

Estimated time per video inference at full frames (448):

| GPU | Time per Inference |
|-----|-------------------|
| A100 | ~18-20 min |
| H100 | ~12-15 min |
| H200 | ~8-10 min |
| B200 | ~4-6 min |

---

## Timing Tables by GPU Type

### 8× A100 GPUs

| Videos | Augmentations | MAX_FRAMES | Time | Output |
|--------|---------------|------------|------|--------|
| 430 | 1 | 448 (full) | ~15-18 hrs | 860 |
| 430 | 1 | 100 | ~4-5 hrs | 860 |
| 200 | 1 | 448 (full) | ~7-8 hrs | 400 |
| 200 | 5 | 448 (full) | ~35-40 hrs | 1,200 |

### 8× H100 GPUs

| Videos | Augmentations | MAX_FRAMES | Time | Output |
|--------|---------------|------------|------|--------|
| 430 | 1 | 448 (full) | ~11-14 hrs | 860 |
| 430 | 1 | 100 | ~3-4 hrs | 860 |
| 200 | 1 | 448 (full) | ~5-6 hrs | 400 |
| 200 | 3 | 448 (full) | ~15-18 hrs | 800 |
| 200 | 5 | 448 (full) | ~25-30 hrs | 1,200 |

### 8× H200 GPUs

| Videos | Augmentations | MAX_FRAMES | Time | Output |
|--------|---------------|------------|------|--------|
| 430 | 1 | 448 (full) | ~7-9 hrs | 860 |
| 430 | 3 | 448 (full) | ~22-27 hrs | 1,720 |
| 430 | 5 | 448 (full) | ~36-45 hrs | 2,580 |
| 200 | 1 | 448 (full) | ~3.5-4 hrs | 400 |
| 200 | 3 | 448 (full) | ~10-12 hrs | 800 |
| 200 | 5 | 448 (full) | ~17-20 hrs | 1,200 |

### 4× H200 GPUs

| Videos | Augmentations | MAX_FRAMES | Time | Output |
|--------|---------------|------------|------|--------|
| 430 | 1 | 448 (full) | ~15-18 hrs | 860 |
| 430 | 3 | 448 (full) | ~45-54 hrs | 1,720 |
| 430 | 5 | 448 (full) | ~75-90 hrs | 2,580 |
| 200 | 1 | 448 (full) | ~7-8 hrs | 400 |
| 200 | 3 | 448 (full) | ~20-24 hrs | 800 |
| 200 | 5 | 448 (full) | ~34-40 hrs | 1,200 |
| 100 | 1 | 448 (full) | ~3.5-4 hrs | 200 |

### 2× B200 GPUs

| Videos | Augmentations | MAX_FRAMES | Time | Output |
|--------|---------------|------------|------|--------|
| 430 | 1 | 448 (full) | ~15-18 hrs | 860 |
| 200 | 1 | 448 (full) | ~7-8 hrs | 400 |
| 90 | 5 | 448 (full) | ~18 hrs | 540 |
| 60 | 5 | 448 (full) | ~12 hrs | 360 |
| 40 | 5 | 448 (full) | ~8 hrs | 240 |
| 20 | 5 | 448 (full) | ~4 hrs | 120 |

### 1× B200 GPU

| Videos | Augmentations | MAX_FRAMES | Time | Output |
|--------|---------------|------------|------|--------|
| 430 | 1 | 448 (full) | ~30-36 hrs | 860 |

---

## Quick Reference: Target 4 Hours

What can you process in approximately 4 hours?

| GPU Setup | Full Frames (448) | Reduced Frames (100) |
|-----------|-------------------|----------------------|
| 4× A100 | 50 videos × 1 aug | 200 videos × 1 aug |
| 8× A100 | 100 videos × 1 aug | 430 videos × 1 aug |
| 4× H100 | 75 videos × 1 aug | 300 videos × 1 aug |
| 8× H100 | 150 videos × 1 aug | 430 videos × 1 aug |
| 4× H200 | 100 videos × 1 aug | 430 videos × 1 aug |
| 8× H200 | 200 videos × 1 aug | 430 videos × 2 aug |
| 4× B200 | 200 videos × 1 aug | 430 videos × 2 aug |
| 8× B200 | 400 videos × 1 aug | 430 videos × 5 aug |

---

## GPU Comparison Summary

Relative speed (A100 = 1.0×):

| GPU | Relative Speed | Notes |
|-----|----------------|-------|
| A100 | 1.0× | Baseline |
| H100 | 1.3-1.5× | ~30-50% faster than A100 |
| H200 | 1.8-2.0× | ~80-100% faster than A100 |
| B200 | 3.0-4.0× | ~3-4× faster than A100 |

---

## Configuration Examples

### Fast Run (4 hours, 4× H200)
```bash
NUM_AUGMENTATIONS=1
NUM_GPUS=4
MAX_FRAMES=448
MAX_VIDEOS=100
# Output: 200 videos
```

### Overnight Run (8-12 hours, 4× H200)
```bash
NUM_AUGMENTATIONS=1
NUM_GPUS=4
MAX_FRAMES=448
MAX_VIDEOS=200
# Output: 400 videos
```

### Full Dataset, Single Aug (15-18 hours, 4× H200)
```bash
NUM_AUGMENTATIONS=1
NUM_GPUS=4
MAX_FRAMES=448
MAX_VIDEOS=0  # All 430 videos
# Output: 860 videos
```

### Maximum Diversity (3-4 days, 4× H200)
```bash
NUM_AUGMENTATIONS=5
NUM_GPUS=4
MAX_FRAMES=448
MAX_VIDEOS=0  # All 430 videos
# Output: 2,580 videos
```

---

## Recommendations for VLA Training

1. **Minimum viable dataset**: 200-400 augmented videos
   - 100 original + 100 augmented with full frames
   - ~4 hours on 4× H200

2. **Balanced dataset**: 800-1,000 videos
   - 200 original + 600 augmented (3× aug)
   - ~20-24 hours on 4× H200

3. **Large scale training**: 2,000+ videos
   - 430 original + 1,720 augmented (4× aug)
   - ~3 days on 4× H200, or ~1.5 days on 8× H200

### Quality vs Quantity Trade-offs

| Priority | Recommendation |
|----------|----------------|
| Full trajectories | Use MAX_FRAMES=448, fewer videos |
| More samples | Use MAX_FRAMES=100, more videos |
| Visual diversity | Increase NUM_AUGMENTATIONS |
| Training speed | Fewer total videos |

---

## Script Usage

```bash
# Set your HuggingFace token
export HF_TOKEN="your_token"

# Run with default settings
bash brev_cosmos_augment.sh

# Or edit configuration at top of script:
# NUM_AUGMENTATIONS=1    # Augmentations per video
# NUM_GPUS=4             # Number of GPUs
# MAX_FRAMES=448         # Frames per video (448 = full)
# MAX_VIDEOS=100         # Videos to process (0 = all)
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce MAX_FRAMES
- Use fewer parallel containers (reduce NUM_GPUS)

### Job taking too long
- Reduce MAX_VIDEOS
- Reduce NUM_AUGMENTATIONS
- Reduce MAX_FRAMES

### Need more output videos
- Increase NUM_AUGMENTATIONS (visual diversity)
- Process all videos (MAX_VIDEOS=0)
- Run multiple times with different prompts
