# results_public

Curated, version-controlled copy of the MTEC experiment outputs. Generated from
the full run directories under `outputs/` by `scripts/slim_results.py`.

## What is here

| Directory | Contents |
|---|---|
| `ablation_20260701/` | Full ablation study: 15 `RESULT_*.txt` comparisons, `FINAL_CHECKPOINT.txt`, per-config run records under `runs/`, and the probe sweep. |
| `PAPER_ARCHIVE_20260625/` | Comparison-phase archive used as the config-A reference baseline: Video-MME, NExT-QA, TempCompass, and MLVU-long, compressed and raw. |

## What was removed, and why

The full run directories are ~23 GB, which is not appropriate for git. Two
classes of data were dropped:

1. **Media artifacts** — 4,199 `.jpg` anchor frames, 335 `.mp4` clips, and the
   `.wav`/`.mp3` audio extracts. These are reproducible from the source datasets.
2. **Compressed prompt payloads** — the `low_resolution_anchor`,
   `structured_evidence_prompt`, `computed_evidence_prompt`, and the
   `pre_api_input_audit` / `final_input_audit` fields on each record. These are
   ~87% of every record's bytes.

This takes 4.60 GB of candidate files down to ~122 MB.

Everything needed to recompute the reported numbers is retained: per-record
`correct`, `ground_truth`, `raw_response`, `status`, `original_tokens`,
`compressed_tokens`, `token_saving_ratio`, `compression_ratio`, timings, and the
full per-stage API metadata.

The unabridged outputs, including media and prompt payloads, remain on the A800
box at `/root/autodl-tmp/MTEC/outputs/`.

## Credentials

The archived `queue.txt` files recorded the API key on each generated command
line. Those keys have been replaced with `${REDACTED_API_KEY}` and the
underlying credentials rotated. Keys now live in an untracked `.env` — see
`.env.example`.

## Regenerating

```bash
python scripts/slim_results.py
```

Rebuilds `results_public/` from `outputs/`. Destructive: it clears the target
directory first.
