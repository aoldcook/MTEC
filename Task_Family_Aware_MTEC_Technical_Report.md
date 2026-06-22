# Technical Report: Task-Family-Aware MTEC Video Evidence Resolver

Date: June 22, 2026  
Repository snapshot: `24e5eb0 add stronger visual counting executor anchors`  
Project root on AutoDL: `/root/autodl-tmp/MTEC`

## 1. Executive Summary

This project extends the MTEC/ZoomRefine-style multimodal evaluation pipeline with a task-family-aware video evidence resolver. The central idea is to avoid treating all video questions as generic video understanding problems. Instead, each question is first routed into a task family, then processed with a resolver-specific evidence template before the final multiple-choice verifier maps evidence to an answer option.

The current system combines:

- Low-resolution global video timeline anchors.
- Low-FPS video anchors for compact visual context.
- High-detail keyframe crops.
- Tubelet storyboards for before/during/after motion evidence.
- OCR/object/motion evidence extraction.
- Transcript and ASR anchors when available.
- A task-family resolver registry.
- A two-pass model flow: evidence extraction followed by final option verification.
- Per-record input auditing to prevent false improvements caused by missing media.

On a 20-video regression set, the current task-family-aware system reached `18/20` correct with `0` runtime failures and an average token saving ratio of `0.6321`. Two difficult visual-counting cases remain unresolved.

## 2. Motivation

Earlier failures were not isolated single-question bugs. They exposed recurring video question failure modes:

- Cross-shot entity counting.
- Stage or scene group attribute counting.
- Container/object counting.
- Missing-set reasoning.
- Stateful OCR, such as scores or screen values.
- Ordinal clip action recognition.
- Domain-specific intent reasoning.

The project therefore moved from ad hoc fixes toward a generalized resolver framework:

```text
Question
-> Task-family router
-> Specialized resolver guidance
-> Structured evidence packet
-> Confidence-gated final verifier
-> Option answer
```

This design aims to make the system more explainable and transferable than a collection of sample-specific patches.

## 3. High-Level Architecture

The main runner is:

```text
scripts/run_modelscope_mtec_anchor_api_full.py
```

Core media and anchor generation logic is in:

```text
zoomrefine/mtec_media_pipeline.py
```

Prompt and evidence formatting logic is in:

```text
zoomrefine/mtec_prompt_plus.py
```

The new task-family routing and resolver registry are in:

```text
zoomrefine/mtec_task_resolvers.py
```

The current video pipeline operates as follows:

1. Load the Video-MME row and resolve the source video.
2. Extract or reuse subtitle/ASR evidence.
3. Build compressed visual anchors:
   - Full-video low-resolution global timeline.
   - Low-FPS video anchor.
   - High-detail keyframe crops.
   - Tubelet storyboards.
   - OCR regions.
   - Object-detection crops.
   - Motion evidence.
4. Route the question to a task family.
5. Attach resolver-specific guidance and evidence templates.
6. Run a global timeline pass when enabled.
7. Run a structured evidence extraction pass.
8. Build a final verifier prompt.
9. Attach media anchors and call the answer model.
10. Parse the final option letter.
11. Log input audits, compression statistics, model metadata, and correctness.

## 4. Task-Family Router and Resolver Registry

The task-family router is implemented in:

```text
zoomrefine/mtec_task_resolvers.py
```

It currently supports these task families:

| Task Family | Resolver Class | Purpose |
|---|---|---|
| `cross_shot_entity_count` | `EntityBankCounter` | Count entities across shots while avoiding duplicates. |
| `scene_group_attribute_count` | `PanoramaAttributeCounter` | Count people or attributes in one scene, such as men/women on a stage. |
| `container_object_count` | `ContainerObjectCounter` | Count visible objects inside a container or on a scoped surface. |
| `missing_set` | `OptionConditionedVisibleSetResolver` | Determine which option is absent by checking each option independently. |
| `stateful_ocr` | `StatefulOCRTracker` | Track stable OCR states such as scores, timers, prices, or displayed text. |
| `ordinal_clip_action` | `OrdinalClipActionResolver` | Identify actions in first/second/third/last logical clips. |
| `domain_intention` | `DomainIntentResolver` | Rank domain-specific intents, such as MOBA warding or attack intent. |
| `scene_conditioned_attribute` | `SceneConditionedAttributeResolver` | Locate the relevant scene before judging attributes. |
| `generic_video_evidence` | `GenericVideoEvidenceResolver` | Fallback for questions that do not match a specialized family. |

The router is currently rule-based. It uses keywords from the question and options to choose a family and emits:

- `task_family`
- `resolver_class`
- `route_confidence`
- `required_evidence`
- `question_scope_guard`
- `evidence_template`
- resolver-specific rules

## 5. Evidence Construction

### 5.1 Low-Resolution Global Timeline

The full-video low-resolution anchor gives the model a compressed but chronological view of the entire video. It is important for:

- Event order.
- Scene transitions.
- Long-range context.
- Recovery when local crops are misleading.

### 5.2 Low-FPS Evidence Video

A low-FPS video anchor preserves motion and scene context at reduced cost. The selected policy depends on the question:

- `tiny`
- `light`
- `medium`
- `full`
- `dense`
- task-aware variants such as `full_scene_group_count`

### 5.3 High-Detail Crops

For OCR, small objects, counts, and local details, the pipeline creates high-resolution keyframe crops. These are attached as image anchors with timestamps and region hints.

### 5.4 Tubelet Storyboards

Tubelets preserve before/during/after continuity around selected moments. They are useful for:

- Action recognition.
- Temporal ordering.
- State changes.
- Avoiding single-frame misunderstandings.

### 5.5 OCR, Object, Motion, and Transcript Evidence

The media pipeline generates deterministic evidence:

- OCR regions and text candidates.
- Object detections and crops.
- Motion saliency boxes.
- Scene segments.
- Transcript/ASR segments and query-relevant windows.

This evidence is not treated as final truth. It is used as structured support for the model verifier.

## 6. Task-Family Evidence Templates

Each task family provides an evidence template. For example:

### Cross-Shot Entity Counting

```json
{
  "target_entity": "question_target",
  "role_filter": ["foreground", "participant", "interacts_with_task"],
  "exclude": ["audience", "background", "host", "referee_if_not_target"],
  "entities": [],
  "count_value": null,
  "count_confidence": 0.0
}
```

### Scene Group Attribute Counting

```json
{
  "scene": "question_scene",
  "count_method": "best_wide_shot_not_sum_across_shots",
  "selected_frame_or_range": "",
  "total_people": null,
  "attribute_breakdown": {},
  "ignored_closeups": []
}
```

### Container Object Counting

```json
{
  "container": "question_container",
  "scope": "beginning_or_question_scope",
  "container_roi": [],
  "visible_items_inside_container": [],
  "excluded_outside_items": [],
  "count_value": null
}
```

### Missing Set

```json
{
  "option_visibility": {},
  "visible_set": [],
  "missing_candidates": [],
  "rule": "absence_by_option_wise_full_scope_aggregation"
}
```

### Stateful OCR

```json
{
  "target_state": "score_or_displayed_text",
  "stable_roi": {},
  "ocr_sequence": [],
  "selected_state": "",
  "state_confidence": 0.0,
  "state_machine_notes": []
}
```

## 7. Scope Guards

A key lesson from the regression set is that temporal scope must not be expanded casually.

For questions containing terms such as:

```text
beginning, at the start, start of, opening, initially, displayed at the beginning
```

the router emits a `beginning_locked` scope guard. This tells the evidence extractor and verifier:

- Opening evidence is primary.
- Later reveal shots are not primary evidence.
- Later ASR or transcript numbers must be treated as out-of-scope unless they directly describe the opening frame.

This was introduced after the system incorrectly used a later product-list narration to answer a beginning-scoped container count question.

## 8. Stronger Visual Counting Executor

The latest commit adds stronger visual counting anchors for difficult count questions.

### 8.1 Scene Group Attribute Count

For scene group counting, if the original video is small enough for the API, the pipeline can attach:

```text
video_visual_count_clip
```

This is an original-resolution video attachment used specifically for count verification. The intended behavior is:

- Count directly from the original-resolution clip.
- Include stage-edge people and presenters.
- Exclude the audience.
- Verify each multiple-choice option independently.

### 8.2 Container Object Count

For container count questions, the pipeline creates:

```text
video_visual_count_sheet
```

and an additional zoom crop over the foreground container/package area. These are intended to help count:

- visible items inside a container,
- visible items on a surface,
- printed or embossed product shapes on a displayed box when the question refers to the displayed package.

### 8.3 Input Audit

The final input audit now records:

```text
visual_count_video_anchor_count
visual_count_sheet_count
```

The runtime log also prints:

```text
visual_count_video=<n>
visual_count=<n>
```

This helps verify that the stronger counting evidence was actually attached before each model call.

## 9. Model Flow

The current default flow uses a two-stage model process:

1. Evidence pass:
   - Builds compact structured JSON evidence.
   - Must not output the final answer.
   - Uses task-family templates.

2. Final answer pass:
   - Receives media anchors, task-family guidance, structured evidence, and input audit context.
   - Evaluates each option independently.
   - Returns exactly one option letter.

The system was tested mainly with Bailian/DashScope-compatible Qwen models. SiliconFlow was also configured in the script, but the provided SiliconFlow key returned `HTTP 401 Invalid token` during this session, so the major evaluation runs used Bailian for both the evidence and final-answer stages.

## 10. Input Auditing

Before final model calls, the system checks that required media exists and is non-empty.

The audit includes:

- Source video path and byte size.
- Subtitle path.
- Video anchor count.
- Full-timeline anchor count.
- Low-FPS frame count.
- Detail crop count.
- Tubelet count.
- OCR crop count.
- Object crop count.
- Visual count sheet count.
- Visual count video count.
- Transcript segment count.
- Media content counts attached to the API call.
- Missing files.
- Empty files.
- Warnings.

This prevents false-positive compression results where the model appears efficient only because critical inputs were missing.

## 11. Testing Methodology

The project was evaluated with a regression-oriented testing workflow rather than a single aggregate benchmark run. The purpose was to verify both overall behavior and specific failure modes exposed by previous runs.

### 11.1 Test Environment

The main experiments were run on the AutoDL server:

```text
/root/autodl-tmp/MTEC
```

The active repository snapshot for the report is:

```text
24e5eb0 add stronger visual counting executor anchors
```

The server environment included:

- Ubuntu 22.04.
- NVIDIA A800 80GB GPU.
- Conda environment: `/root/miniconda3/envs/venv`.
- Video-MME metadata and video zip files stored under the project data directories.
- Precomputed subtitle/ASR files under the project output directories.

### 11.2 API and Model Setup

The evaluation scripts support SiliconFlow and Bailian/DashScope-compatible APIs. During this testing session:

- The SiliconFlow key returned `HTTP 401 Invalid token`, so it was not used for the main evaluation.
- Bailian/DashScope-compatible Qwen was used for both evidence extraction and final answer verification.
- API keys were supplied as environment variables and were not written into the report or project files.

The main model configuration used:

```text
model: qwen3.7-plus
answer_model: qwen3.7-plus
temperature: 0
evidence_pass: true
global_timeline_pass: true
video_anchor_policy: auto
video_global_anchor: true
video_query_retrieval: true
```

### 11.3 Main Regression Set

The main test used a fixed 20-video regression set selected from Video-MME. This set includes ordinary examples and previously observed hard failures, including:

- `7R1eNHvfspk`: cross-shot challenger counting.
- `84EpEwIVFdU`: missing-set reasoning.
- `6DO8yOVYXr0`: current-score OCR.
- `8np5YKYx3sU`: stage men/women counting.
- `6NVr0cNiHPM`: beginning-scoped box/item counting.

The run used explicit record keys rather than random sampling. This makes results comparable across code changes.

The primary output directory was:

```text
outputs/video20_task_family_resolver_bailian_20260622
```

### 11.4 Input Completeness Checks

Before relying on any result, the project verifies that each example has the required inputs:

- The Video-MME metadata row exists.
- The corresponding video file exists in the video zip pool.
- The extracted video file exists and is non-empty.
- Video anchors were generated.
- The full-timeline low-resolution anchor was generated.
- Tubelet storyboards were generated.
- Detail crops, OCR crops, object crops, and visual count anchors are counted.
- Transcript/subtitle availability is recorded.
- The final API request has attached media content.

The runtime log prints an `INPUT_CHECK` line for every record. A typical line includes:

```text
video_media=<n>
image_media=<n>
frames=<n>
global_full=<n>
global_samples=<n>
visual_count_video=<n>
tubelets=<n>
details=<n>
visual_count=<n>
ocr=<n>
motion=<n>
objects=<n>
transcript_segments=<n>
warnings=<...>
```

This prevents a misleading result where token saving appears high only because important visual evidence was missing.

### 11.5 Correctness Evaluation

For multiple-choice examples, the final model response is parsed into an option letter. The prediction is compared with the ground-truth answer from the dataset.

Each record stores:

- `record_key`
- `status`
- `Answer`
- `ground_truth`
- `correct`
- raw model response
- computed evidence response
- final input audit
- compression statistics
- elapsed time

The summary file aggregates:

- total records
- completed records
- failed records
- correct records
- accuracy
- average compression ratio
- average token saving ratio
- elapsed seconds

### 11.6 Error Analysis Method

For each wrong answer, the analysis distinguishes between:

- API or platform failure.
- Invalid or malformed evidence JSON.
- Missing input or missing media anchor.
- Wrong task-family routing.
- Scope contamination from out-of-scope transcript or later video evidence.
- Visual understanding failure.
- Final verifier mapping failure.

For the current two remaining wrong cases:

- `8np5YKYx3sU` routed correctly and had complete inputs, but the model repeatedly under-counted stage performers.
- `6NVr0cNiHPM` routed correctly and had complete inputs, but the model was unstable on the beginning box/package count and was easily pulled toward later product-list evidence.

### 11.7 Direct-Video Upper-Bound Probes

The project also used direct-video probes to test whether the answer model could solve a case without the compressed evidence pipeline.

For `8np5YKYx3sU`:

- The original video could be attached directly to Bailian.
- The model returned the correct answer `A`.
- This suggests the current full pipeline, not the base model alone, contributes to the remaining failure.

For `6NVr0cNiHPM`:

- The full original video could not be sent as a single data URI because of platform limits.
- An unrecompressed beginning segment was accepted.
- The model still did not return the ground-truth answer.
- This suggests the case is intrinsically hard for the model under the available API constraints.

### 11.8 Retry Experiments

After implementing stronger visual counting anchors, the project reran the two remaining wrong examples.

The main retry output directory was:

```text
outputs/video2_strong_visual_count_executor_v2_20260622
```

An additional 8np suppression run was stored in:

```text
outputs/video1_8np_visual_count_clip_suppress_evidence_20260622
```

These retries verified that the stronger visual count anchors were attached successfully, but the two answers still remained incorrect.

## 12. Experimental Results

### 12.1 Main 20-Video Regression Run

Output directory:

```text
outputs/video20_task_family_resolver_bailian_20260622
```

Summary:

```text
total: 20
completed: 20
failed: 0
correct: 18
accuracy: 0.90
avg_compression_ratio: 0.3694
avg_token_saving_ratio: 0.6321
avg_token_saving_ratio_positive_only: 0.6653
elapsed_seconds: 1827.267
```

Notable resolved cases:

| Video ID | Task | Result |
|---|---|---|
| `7R1eNHvfspk` | Cross-shot challenger count | Correct |
| `84EpEwIVFdU` | Missing-set reasoning | Correct |
| `6DO8yOVYXr0` | Stateful score OCR | Correct |

### 12.2 Strong Visual Counting Executor Retry

Output directory:

```text
outputs/video2_strong_visual_count_executor_v2_20260622
```

Summary:

```text
total: 2
completed: 2
failed: 0
correct: 0
accuracy: 0.0
```

Both difficult count cases remained incorrect even after attaching stronger count evidence.

### 12.3 Evidence Suppression Retry for Stage Count

Output directory:

```text
outputs/video1_8np_visual_count_clip_suppress_evidence_20260622
```

Summary:

```text
total: 1
completed: 1
failed: 0
correct: 0
```

The experiment suppressed intermediate computed evidence for the final verifier when an original-resolution visual count clip was attached, but the model still predicted the wrong option.

## 13. Current Remaining Problems

### 13.1 Stage Group Count: `8np5YKYx3sU`

Question:

```text
How many men and women are presenting on the stage?
A. Six men and three women.
B. Five men and two women.
C. Four men and three women.
D. Four men and four women.
```

Ground truth:

```text
A. Six men and three women.
```

Current system prediction:

```text
C. Four men and three women.
```

Diagnosis:

- The router correctly selects `scene_group_attribute_count`.
- The resolver class is `PanoramaAttributeCounter`.
- Input audit is clean.
- The system attaches an original-resolution visual count clip when possible.
- The model still repeatedly counts only `4 men + 3 women`.
- A direct original-video probe outside the normal two-stage pipeline returned the correct answer, but the full pipeline still failed.

Likely cause:

The final model is still influenced by the compressed context, intermediate timeline, or prompt structure, and does not reliably re-count all stage-edge performers from the attached original-resolution video.

Needed future work:

- A deterministic or semi-deterministic person detector/tracker specialized for wide stage frames.
- A model call dedicated only to counting people in a selected wide frame, isolated from generic transcript/evidence context.
- Explicit stage ROI extraction and possibly person bounding-box visualization.
- Option-conditioned verification: ask the model to verify `6 men + 3 women`, `5 men + 2 women`, etc., rather than freely generating a count.

### 13.2 Container/Box Count: `6NVr0cNiHPM`

Question:

```text
How many items are stored in the box displayed at the beginning of the video?
A. 7.
B. 8.
C. 10.
D. 9.
```

Ground truth:

```text
C. 10.
```

Current system prediction:

```text
B. 8.
```

Diagnosis:

- The router correctly selects `container_object_count`.
- The resolver class is `ContainerObjectCounter`.
- The input audit is clean.
- The stronger visual count sheet and container/package zoom crop are attached.
- The model tends to use a later narration or product reveal stating “eight full-size products”.
- When constrained to the beginning segment, the model interprets the box as closed or counts the displayed package artwork incorrectly.

Additional direct-probe finding:

- The full original video could not be sent as a single data URI because of platform size limits.
- An unrecompressed beginning segment was accepted by the model, but it predicted `D` rather than the ground truth `C`.

Likely cause:

The question requires interpreting the displayed box/package at the beginning, possibly including printed or embossed item shapes. Current visual evidence does not make this count robust, and the model is easily pulled toward later textual/narration evidence.

Needed future work:

- A better beginning-scene locator that finds the clearest package-facing frame, not the blank first frame.
- Super-resolution or high-quality crop extraction of the box face.
- Explicit segmentation of printed/embossed item shapes on the package.
- A hard rule that later “eight full-size products” narration must not override the beginning box count.
- A dedicated package-artwork counting pass.

### 13.3 Intermediate Evidence Can Become Overconfident

The evidence pass sometimes produces a plausible but wrong structured count, such as `4 men + 3 women`. Once this appears in the structured evidence, the final verifier may follow it even if stronger media is attached.

Partial mitigation was attempted:

- Suppressing computed evidence for final verification when an original visual count clip is attached.

Result:

- This did not fix `8np5YKYx3sU`.

Needed future work:

- Separate “evidence recording” from “count commitment”.
- For high-risk count families, force the final pass to ignore any intermediate count unless the count is accompanied by explicit per-instance evidence.

### 13.4 API and Platform Constraints

The system currently uses data URI attachments for media. This has practical limits:

- Large original videos may exceed request string length limits.
- Single data URI items may exceed byte limits.
- Some prompts or media combinations may trigger platform data inspection.

Observed examples:

- Full `6NVr0cNiHPM` original video exceeded data URI limits.
- A first strong visual count sheet attempt for `8np5YKYx3sU` triggered `data_inspection_failed`.

Needed future work:

- Use platform-supported file upload or object-storage URLs instead of base64 data URIs.
- Add automatic media-size routing.
- Add fallback modes when data inspection rejects a request.

## 14. Strengths of the Current System

The current project already has several important strengths:

- Task-family routing is explicit and inspectable.
- Resolver templates make evidence requirements clear.
- Input audit prevents silent missing-media failures.
- The pipeline records detailed compression and token-saving metrics.
- Scope guards reduce obvious temporal leakage.
- Missing-set and OCR-style questions improved in the regression set.
- The architecture is modular enough to add stronger deterministic executors.

## 15. Limitations

The current implementation still relies heavily on the multimodal model for final visual counting. This creates several limitations:

- People in dark stage backgrounds may be missed.
- Stage-edge performers may be ignored.
- Printed package artwork may be interpreted inconsistently.
- Later transcript evidence can still influence answers.
- Intermediate evidence can become a wrong but persuasive bottleneck.
- Large original media cannot always be attached because of API limits.

## 16. Conclusion

This project has evolved from a generic compressed-video QA pipeline into a task-family-aware video evidence resolver. The current implementation successfully routes questions into interpretable task families and improves multiple recurring failure modes, reaching `18/20` on the current regression set.

The remaining failures are concentrated in hard visual counting. They are not caused by missing files or invalid output formats. They require stronger task-specific visual executors that can produce instance-level or ROI-level evidence before the final model performs option selection.

The next technical milestone is therefore to move from prompt-guided visual counting to detector/ROI-assisted visual counting with explicit per-instance evidence.
