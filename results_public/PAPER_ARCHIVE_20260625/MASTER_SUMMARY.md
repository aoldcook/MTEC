# MTEC Experiment Archive — 20260625

Consolidated per-record results + metrics for all runs. Each `<dataset>/` holds the deduped
`results.jsonl` (per-record logs) and a `summary.json` (config + metrics).

## Master metrics table

| Dataset | Platform | Model | n | Completed | Acc (compl.) | Acc (incl. fail) | Token save (med raw) | Top failures |
|---|---|---|---|---|---|---|---|---|
| Video-MME 300 (compressed) | bailian | qwen3.7-plus | 300 | 300 | 76.3% | 76.3% | 85.9% | — |
| NExT-QA 300 (compressed) | bailian | qwen3.7-plus | 300 | 300 | 82.7% | 82.7% | 40.9% | — |
| TempCompass 300 (compressed) | bailian | qwen3.7-plus | 300 | 293 | 81.6% | 79.7% | -26.4% | — |
| MLVU-long 180 (compressed) | bailian | qwen3.7-plus | 180 | 180 | 82.2% | 82.2% | 88.6% | — |
| Video-MME short/med 105 (raw direct) | bailian | qwen3.7-plus | 27 | 27 | 81.5% | 81.5% | — | — |
| Video-MME long 100 (raw, Bailian pool) | bailian | qwen3.6-plus + pool | 100 | 29 | 72.4% | 21.0% | — | all_models_exhausted:59, quota_or_capability:3, content_insp |
| NExT-QA 300 (raw, SiliconFlow) | siliconflow | Qwen3-VL-32B-Instruct | 300 | 291 | 79.7% | 77.3% | — | api_error:9 |
| Video-MME medium 102 (raw, SiliconFlow) | siliconflow | Qwen3-VL-32B-Instruct | 102 | 63 | 54.0% | 33.3% | — | video_too_large:37, api_error:2 |
| MLVU-long 180 (raw, SiliconFlow) | siliconflow | Qwen3-VL-32B-Instruct | 180 | 39 | 61.5% | 13.3% | — | video_too_large:141 |
| MLVU-long 141-rerun (raw, Bailian OSS) | bailian | qwen3.6-plus | 76 | 62 | 90.3% | 73.7% | — | content_inspection:10, quota:2, api_error:1, video_too_large |

### Video-MME 300 (compressed) — duration breakdown

| Split | n | Accuracy | Token save (median raw) |
|---|---|---|---|
| short | 100 | 88.0% | 76.5% |
| medium | 100 | 74.0% | 86.0% |
| long | 100 | 67.0% | 89.3% |

## Notes / caveats per run

- **Video-MME 300 (compressed)** (`videomme_300_compressed/`): balanced 100/100/100. Failures: {}
- **NExT-QA 300 (compressed)** (`nextqa_300_compressed/`): video embedded in parquet. Failures: {}
- **TempCompass 300 (compressed)** (`tempcompass_300_compressed/`): short clips; dedup of double-process artifact. Failures: {'(none)': 7}
- **MLVU-long 180 (compressed)** (`mlvu_long_180_compressed/`): 10-60min videos. Failures: {}
- **Video-MME short/med 105 (raw direct)** (`videomme_105_raw_bailian/`): early raw probe. Failures: {}
- **Video-MME long 100 (raw, Bailian pool)** (`videomme_long_raw_bailian_pool/`): mixed-model pool; quota-limited. Failures: {'(none)': 8, 'quota_or_capability': 3, 'all_models_exhausted': 59, 'content_inspection': 1}
- **NExT-QA 300 (raw, SiliconFlow)** (`nextqa_300_raw_siliconflow/`): small videos. Failures: {'api_error': 9}
- **Video-MME medium 102 (raw, SiliconFlow)** (`videomme_medium_raw_siliconflow/`): many exceed ~50-80MB base64 ceiling. Failures: {'video_too_large': 37, 'api_error': 2}
- **MLVU-long 180 (raw, SiliconFlow)** (`mlvu_long_raw_siliconflow/`): most exceed base64 ceiling -> video_too_large. Failures: {'video_too_large': 141}
- **MLVU-long 141-rerun (raw, Bailian OSS)** (`mlvu_long_141rerun_raw_bailian/`): PARTIAL/in-progress; platform comparison. Failures: {'content_inspection': 10, 'api_error': 1, 'quota': 2, 'video_too_large_for_disk': 1}
