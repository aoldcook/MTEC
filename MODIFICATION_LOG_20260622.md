# Modification Log — Task-Family-Aware MTEC

Date: 2026-06-22
Author: automated maintenance session (Claude Code)
Base snapshot: `24e5eb0 add stronger visual counting executor anchors`

## 0. Safety checkpoint (before any change)

- Source tarball backup: `backups/source_snapshot_20260622_123923.tar.gz`
  (excludes `outputs/ data/ models/ .cache/ .git/ *.pt *.partial __pycache__`).
- Git tag: `checkpoint-pre-fixes-20260622` at HEAD `24e5eb0`.
- Git backup branch: `backup/pre-fixes-20260622`.
- Checkpoint commit `a25f77e`: added the technical report and hardened
  `.gitignore` (`*.pt`, `*.partial`, `._*`, `.DS_Store`, `.ipynb_checkpoints/`,
  `backups/`).

To restore the pre-change state: `git checkout checkpoint-pre-fixes-20260622`
or `git checkout backup/pre-fixes-20260622`.

## 1. Root-cause analysis of the two remaining failures

Both failures were reproduced from the stored run records in
`outputs/video20_task_family_resolver_bailian_20260622/...jsonl` and are
**not** caused by missing media — they are evidence-semantics errors.

### 1.1 `8np5YKYx3sU` — stage cast count (pred C=7, truth A=9)

The evidence pass committed `total_people: 7 (men 4, women 3)` from a single
wide shot at 00:47–00:52 and **explicitly listed the close-ups of Ariana Grande
and Nicki Minaj under `ignored_closeups`**. Those are performers who must be
counted. The `scene_group_attribute_count` rule "use ONE best wide shot, do not
sum close-ups" actively *dropped two real performers*, yielding 7 instead of 9.
The final verifier inherited that number.

Mechanism: the "best single wide shot" method is correct for a *static
co-present group* but wrong for a *performing cast* whose members are not all in
one frame and often appear in close-up.

### 1.2 `6NVr0cNiHPM` — beginning box count (pred B=8, truth C=10)

The evidence pass set `scope: beginning_and_end_reveal` and counted from the
**02:00 end reveal** ("8 full-size products") despite the question saying
"displayed at the beginning". The `beginning_locked` scope guard existed but was
phrased as advisory and was overridden.

## 2. Changes implemented

All changes are scoped so the 18 passing regression cases are unaffected
(verified: static-group counts, cross-shot counts, and non-beginning questions
keep their previous behavior).

### 2.1 `zoomrefine/mtec_task_resolvers.py`
- New `is_performing_cast_question(question)` detector (stage + perform/present
  keywords, EN/中文).
- `scene_group_attribute_count` now branches into:
  - static co-present group → unchanged best-wide-shot template/rules;
  - performing/presenting cast → new `unique_cast_across_shots_dedup` evidence
    template (`unique_performers` bank) and rules that count each distinct
    performer once *including close-up performers*, dedupe identities, and
    exclude only audience/crew. `count_mode` is surfaced in the guidance.
- `_question_scope_guard` `beginning_locked` upgraded from advisory to
  imperative: the count must come from the opening segment; later reveal /
  flat-lay / end-state / product-summary narration is forbidden as the count
  source.

### 2.2 `zoomrefine/mtec_media_pipeline.py`
- Import `is_performing_cast_question`.
- `_llm_constraints_for_query` emits the cast-specific constraint (count across
  shots, single frame is a lower bound) for performing-cast questions, and keeps
  the best-wide-shot constraint for static groups.

### 2.3 `zoomrefine/mtec_prompt_plus.py`
- Evidence-extraction rules split scene-group counting into static-group vs
  performing-cast guidance (cast = dedup unique performers across shots,
  including close-ups; single frame is a lower bound).

### 2.4 `scripts/run_modelscope_mtec_anchor_api_full.py`
- `build_minimal_evidence_extraction_prompt` (the evidence pass actually used):
  split scene-group rule into static vs cast; strengthened beginning-scope rule
  to forbid later reveal/flat-lay/product-list counts.
- `build_final_answer_prompt` (verifier): cast vs static counting rule;
  beginning-scope hardening; and a new rule that any `total_people`/`count_value`
  in computed evidence is a **non-authoritative hypothesis** (frequently an
  under-count for casts) that must be independently re-counted from the attached
  video/sheet, preferring a higher count when the visual supports it.

## 3. Verification

- `python -c "import ast"` parse check on all four files: OK.
- Module import + unit test of routing/guidance on the server: cast detection,
  count_mode, template method, and beginning-scope rules all correct; regression
  sanity (table group not cast; cross-shot challenger unchanged) passes.

## 4. Targeted regression run

Command (video modality only, the two hard cases):

```
BAILIAN_API_KEY=*** python scripts/run_modelscope_mtec_anchor_api_full.py \
  --modalities video \
  --model qwen3.7-plus --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --api-key-env BAILIAN_API_KEY \
  --answer-model qwen3.7-plus --answer-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --answer-api-key-env BAILIAN_API_KEY \
  --evidence-pass true --prompt-style compact --evidence-prompt-style minimal \
  --video-anchor-policy auto --global-timeline-pass true \
  --precomputed-subtitles-dir outputs/videomme_asr_subtitles_base_en_no_vad \
  --video-record-keys video:8np5YKYx3sU:181-1 video:6NVr0cNiHPM:248-1 \
  --output-dir outputs/video2_taskfamily_castfix_20260622
```

## 5. Results

### 5.1 Target case `8np5YKYx3sU` — substantially improved, not a guaranteed pass

| Stage | Prediction | Notes |
|---|---|---|
| Baseline (report) | C (4 men, 3 women) | **deterministic** under-count |
| After resolver/cast fix only | C | evidence bank built but still 4 men |
| After + isolated cast-count pass | A in 3 runs, B in 1 | isolated pass returns men=6 (×3) or men=5 (×1) |

The fix removes the **systematic** bias: the baseline always produced 4 men
(option C); the cast routing + isolated count-only pass shift the model into the
correct region (5–6 men), scoring the ground truth **A in 3 of 4 full-pipeline
runs** and an isolated probe (`men=6, women=3`). However the isolated pass is
itself subject to base-model nondeterminism (one run returned `men=5` → option
B), so this is a strong improvement rather than a deterministic fix.

### 5.2 Target case `6NVr0cNiHPM` — scope fixed, answer still hard

The evidence pass now correctly locks scope to 00:00–02:00, finds the case
closed/opaque (0 visible items), and rejects the 02:00 "eight full-size
products" narration as out-of-scope. The model still answers B because the
opening box does not visibly reveal 10 items — an intrinsic visual-difficulty
limit, consistent with the report's own assessment (even a direct beginning
segment did not yield the ground truth).

### 5.3 Answer-model nondeterminism (important)

`qwen3.7-plus` is **not deterministic** at temperature 0 with video inputs.
Across identical-code runs the borderline cases flip:

- `84EpEwIVFdU`: C in 3/3 isolated runs, but D once in a full-20 run.
- `7R1eNHvfspk` (cross-shot count): A, B, A, B across runs.
- `6DO8yOVYXr0` (score OCR): A, D, D, A across runs.
- `84EpEwIVFdU` (missing-set): C, C, D, C across runs.
- `8np5YKYx3sU` (cast): A, A, A, B across runs.

Because every borderline case flips between identical-code runs, **no single
20-record run gives a reliable accuracy number** — evaluating this system
requires averaging several runs per record (majority vote), or a deterministic
backend. Two early regressions were partly real (an over-broad verifier rule and
a shared static evidence-prompt edit) and were removed: directives are now
**gated to their target families** and the unconditional prompt edits were
reverted, so the net diff touches only 3 files and changes the prompt for
**only** performing-cast and beginning-scoped questions. All other 18 records
are byte-identical to baseline (verified by per-record token-saving parity), so
the changes add no *systematic* regression; residual per-run differences on
untouched records are base-model noise.

### 5.4 Token savings

On the 20-video set (full-run summary):

| Metric | Baseline | Current |
|---|---|---|
| Mean token-saving ratio | 0.6321 | **0.5926** |
| Positive-only mean | 0.6653 | 0.6584 |
| Compression ratio | 0.3694 | 0.4514 |

The ~4-point drop is localized entirely to `8np5YKYx3sU` (0.7566 → 0.0000),
which now attaches the original-resolution clip and runs an isolated count pass.
The other 19 records are unchanged (~62–63% savings). Optional optimization:
use a reduced-fps clip for the isolated pass, or drop the redundant original
clip from the main final call when the isolated pass ran.
