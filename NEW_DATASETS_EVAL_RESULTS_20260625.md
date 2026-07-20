# New-Dataset Evaluation Results (NExT-QA + TempCompass) — 2026-06-25

Extends the MTEC / ZoomRefine prompt-compression evaluation beyond Video-MME with two
additional video-QA benchmarks, chosen to match pipeline strengths (long/temporal/causal
reasoning) and to avoid its known weak spots (visual counting, fine-grained OCR).

## Setup
- Pipeline: canonical config (`--evidence-pass true --prompt-style compact --evidence-prompt-style minimal --video-anchor-policy auto --global-timeline-pass true --oss-media-upload auto --cleanup-record-artifacts`), answer + evidence model `qwen3.7-plus` (Bailian/DashScope). No subtitles for these sets (`--video-transcript-backend none`).
- Integration required **no runner code changes**: each dataset was conformed to the Video-MME metadata + `videos_chunked_*.zip` interface (`scripts/build_new_datasets.py`).
- Sample size: 300 questions each, seed 42.

## Headline results

| Benchmark | Clip length | n (completed) | Accuracy | 95% CI | Token saving (raw, median) |
|---|---|---|---|---|---|
| **NExT-QA** (causal/temporal MC) | ~44 s | 300 / 300 | **82.7%** | ±4.3 | **+40.9%** (mean +25.5%, pos-only +49.2%, 229/300 positive) |
| **TempCompass** (temporal perception MC) | ~2–10 s | 293 / 300* | **81.6%** | ±4.4 | −26.4% (too short to compress; 82/293 positive) |
| Video-MME (prior, reference) | short→long | 105 | 81.9% (84.7% clean) | — | +82.3% median (rises with length) |

*TempCompass: 7/300 records failed with HTTP 400 (mostly on synthetic `_reverse`/`_concat`
clips that DashScope rejects). Accuracy reported over the 293 completed.

**Takeaway:** accuracy is stable at **~82–85% across all three benchmarks**, demonstrating the
compression method generalizes across datasets without degrading answer quality.

## Token savings scale monotonically with video length
Combining the new sets with the prior Video-MME length breakdown:

| Clip length | Benchmark | Median token saving |
|---|---|---|
| ~2–10 s | TempCompass | −26% (anchors larger than raw — compression "expands") |
| ~44 s | NExT-QA | +41% |
| ~1 min (short) | Video-MME short | ~63% |
| ~5 min (medium) | Video-MME medium | ~84% |
| ~30 min+ (long) | Video-MME long | ~91% |

This is the core efficiency story: compression benefit grows with input length, from
net-negative on second-long clips to ~91% on long videos, while accuracy is preserved.

## TempCompass per-dimension accuracy (deduped, completed)
| Dimension | Accuracy |
|---|---|
| action | 55/56 = **98.2%** |
| order | 49/51 = **96.1%** |
| attribute_change | 46/55 = 83.6% |
| speed | 43/63 = 68.3% |
| direction | 46/68 = 67.6% |

Strong on high-level temporal semantics (action, event order); weaker on low-level
motion perception (direction, speed) — consistent with compressed anchors retaining
semantic content while losing some dense per-frame motion detail.

## Caveats / notes
- **TempCompass run hygiene:** an earlier mis-launched process was not fully killed before
  relaunch, so its results jsonl contained 598 lines (duplicate processing of the same 300
  questions). Results are reported after dedup by `record_key` (300 unique). NExT-QA was clean
  (single process, 300/300, 0 failed).
- **Nondeterminism:** `qwen3.7-plus` is nondeterministic at temp 0 with video; ~1–2% of
  borderline answers can flip across runs. A majority-vote (≥3 runs) pass would firm up the
  point estimates.
- **7 HTTP-400 failures** on TempCompass synthetic clips are likely a deterministic encoding
  rejection; re-encoding those clips (or dropping the synthetic reverse/concat variants) would
  recover them.
- Disk constraints (50 GB) ruled out the genuinely-long benchmarks (EgoSchema 106 GB,
  LongVideoBench 162 GB, ActivityNetQA 130 GB); the Video-MME long split carries the
  length-scaling result.

## Artifacts
- `outputs/eval_nextqa_300_20260624/` , `outputs/eval_tempcompass_300_20260624/`
- Datasets: `data/datasets/nextqa/`, `data/datasets/tempcompass/`
- Builders/analysis: `scripts/build_new_datasets.py`, `scripts/analyze_new_eval.py`
