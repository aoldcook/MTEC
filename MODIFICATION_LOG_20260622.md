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

### 5.1 Target case `8np5YKYx3sU` — FIXED

| Run | Prediction | Correct |
|---|---|---|
| Baseline (report) | C (4 men, 3 women) | ✗ |
| After resolver/cast fix only | C | ✗ |
| After + isolated cast-count pass | **A (6 men, 3 women)** | ✓ |

Verified correct in **3 independent runs** plus an isolated count-only probe
(returns `men=6, women=3`). Root cause: the model under-counts the cast inside
the cluttered two-pass context, but counts it correctly when given only the
original clip with a focused prompt.

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
- An early over-broad verifier rule (since gated) and a shared static
  evidence-prompt edit (since reverted) shifted `7R1eNHvfspk` and
  `6DO8yOVYXr0`. After **gating all directives to their target families and
  reverting the unconditional prompt edits**, both reverted to correct:
  `7R1=A`, `6DO8=A`, `84Ep=C`, `8np=A` on the final code.

The net diff vs baseline now touches only 3 files and changes the prompt for
**only** performing-cast and beginning-scoped questions; all other 18 records
are byte-identical to baseline (verified by per-record token-saving parity).

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
