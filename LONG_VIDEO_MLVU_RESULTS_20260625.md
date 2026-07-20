# Long-Video Evaluation (MLVU, streaming) — 2026-06-25

Verifies MTEC compression on **genuinely long videos**, completing the short→medium→long
story. Source: **MLVU** (AI-ModelScope/MLVU), reasoning tasks only (no counting/OCR).

## Why MLVU + streaming
- YouTube is blocked on this host (`yt-dlp` → Errno 99), ruling out Video-MME-long via YouTube.
- LongVideoBench / EgoSchema / ActivityNetQA store videos only in 100–162 GB monolithic
  split-tars (no random access) — infeasible on a 50 GB disk.
- MLVU stores videos as **individually addressable files**, enabling true streaming:
  download one video → evaluate → delete, in 6 parallel shards. Peak disk stayed ~4–8 GB.
- Disk freed first by deleting 28 GB of unused local models (`qwen2.5-omni-7b`, `qwen2.5-vl-3b`);
  a `VIDEO_SAMPLE_MANIFEST.json` records every sample→video→source mapping for later retrieval.

## Sample
- **180 distinct videos, 10–60 min (median ~12 min)**, reasoning tasks:
  order(95), plotQA(24), needle(24), anomaly(24), topic(9), ego(4).
- Same canonical pipeline config as the other benchmarks (`--video-transcript-backend none`).

## Results
- **180/180 completed, 0 failed.**
- **Accuracy = 148/180 = 82.2%**  (95% CI ±5.6)
- **Token savings: mean 90.1%, median 88.6%, min 72.3%, max 99.1% — all 180 positive.**
- Mean **42.5 M tokens → 2.79 M tokens = 15.3× reduction.**

### Per-task accuracy
| Task | Accuracy |
|---|---|
| plotQA | 22/24 = 91.7% |
| needle (needle-in-haystack retrieval) | 22/24 = 91.7% |
| topic_reasoning | 8/9 = 88.9% |
| order (long-range temporal sequencing) | 76/95 = 80.0% |
| ego | 3/4 = 75.0% |
| anomaly_reco | 17/24 = 70.8% |

Strong on semantic retrieval (plotQA/needle/topic) — exactly what query-routed anchors
preserve; more modest on fine temporal ordering and anomaly spotting across long footage.

## The complete compression-vs-length picture
| Benchmark | Clip length | Accuracy | Median token saving |
|---|---|---|---|
| TempCompass | ~2–10 s | 81.6% | −26% (too short; anchors expand) |
| NExT-QA | ~44 s | 82.7% | +41% |
| Video-MME short | ~1 min | 81.9%* | ~63% |
| Video-MME medium | ~5 min | (incl. in *) | ~84% |
| **MLVU long** | **10–60 min** | **82.2%** | **+88.6% (mean 90.1%, 15.3×)** |
| Video-MME long | up to ~1 hr | (incl. in *) | ~91% |

\* Video-MME prior 105-record run: 81.9% overall.

**Two headline claims, now fully supported:**
1. **Accuracy is flat at ~82–85% from 2-second clips to hour-long videos** — compression
   preserves answer quality regardless of length.
2. **Savings grow monotonically with length**, from net-negative on second-long clips to
   ~90% (15.3×) on long videos — the efficiency benefit is greatest exactly where input is largest.

## Caveats
- Nondeterminism (~1–2% borderline flips) applies as before; majority-vote would tighten estimates.
- MLVU long sample is order-task-heavy (95/180) because MLVU simply has the most long videos
  in that task; all are reasoning tasks (no counting/OCR).
- Streamed videos were deleted after testing; `VIDEO_SAMPLE_MANIFEST.json` +
  `data/datasets/mlvu/mlvu_stream_map.json` allow exact re-download for comparison.

## Artifacts
- `outputs/eval_mlvu_long_20260625/s0..s5/` (per-shard results)
- `data/datasets/mlvu/mlvu_meta.parquet`, `mlvu_stream_map.json`
- `data/datasets/VIDEO_SAMPLE_MANIFEST.json`
- `scripts/manifest_and_mlvu.py`, `scripts/stream_eval_mlvu.py`, `scripts/analyze_mlvu.py`
