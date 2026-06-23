# Deterministic / Instance-Level Visual Counting — Implementation & Results

Date: 2026-06-23
Flag: `--deterministic-count` (default **OFF**).

## What was implemented
Per the request to "preserve continuous segments where the target appears (not
isolated keyframes) + retain task-relevant local regions + instance-level
deterministic counting", for counting questions the pipeline now optionally:
1. **Continuous dense segment** — samples a dense, evenly-spaced set of frames over
   the counting window (beginning-locked -> opening 8 s; else full duration) and
   attaches them as one montage anchor, so the verifier sees uninterrupted motion
   instead of sparse keyframes.
2. **Instance-level detector** — YOLO person detection per frame -> max simultaneous
   persons + an annotated densest frame (boxes + indices).
3. **Repetition counter** — for action counts, a frame-to-frame motion-energy
   peak count (deterministic, approximate).
These are surfaced as `deterministic_count_evidence` + a verifier directive.

## Results on the 6 residual counting cases (with --deterministic-count)

| Record | gt | prompt-only fix | + deterministic | note |
|---|---|---|---|---|
| 8np5YKYx3sU (cast) | A | A ✓ | A ✓ | already fixed by prompt/cast pass |
| zOgYnntFl-k (raised hands) | D | D ✓ | C ✗ | regressed, then nondeterministic |
| 5kmnEgBSCfg (sets of jumps) | D(5) | B(6) ✗ | B(6) ✗ | motion-rep estimated 4; model said 6 |
| 5Knkqo-lYF0 (rolls) | A | B ✗ | B ✗ | unchanged |
| 6NVr0cNiHPM (box items) | C | B ✗ | B ✗ | items aren't detector classes; occluded |
| zbvamKv81o0 (fewest acrobats) | A | C ✗ | D ✗ | comparison; nondeterministic |

**Deterministic counting scored 1/6 — no improvement over the prompt-only fix
(2/6), and it regressed one case.** Kept default-OFF; the shipped pipeline is
unaffected.

## Why it did not help (root causes)
1. **Total-person count != "how many people did ACTION X".** For `zOgYnntFl-k`
   ("how many raised their right hands") YOLO counted 14 persons over the whole
   video and the prior said "at least 14", pulling the answer away from the true 7.
   (Now gated out: the person prior is restricted to group-size scene questions and
   excludes cross_shot / action-qualified questions.)
2. **Motion-energy repetition counting is too coarse** for exact MCQ counts
   (5kmnEgBSCfg: estimated 4 vs true 5; off-by-one is decisive for the option).
   Exact rep counting needs pose tracking / temporal action-detection, not a generic
   1-D motion signal.
3. **Object counts outside detector vocabulary** (`6NVr0cNiHPM` beauty products in a
   partly-occluded box) get no useful detector signal.
4. **Residual misses are dominated by run-to-run nondeterminism** on borderline
   adjacent options (zOg, zbvam swing between runs).

## Conclusion / recommendation
Generic YOLO + motion-energy is not sufficient for these instance/repetition
counts. The continuous-segment idea is sound and low-risk, but the deterministic
*count* needs task-specific models:
- **Repetition counting**: a pose estimator (e.g. keypoint vertical-oscillation
  cycles) or a temporal repetition-counting model (RepNet-style).
- **Action-subset counts** ("raised hands"): hand/pose action detection, not total
  person detection.
- **Arbitrary object counts**: open-vocabulary detector (e.g. GroundingDINO) for
  the question's specific object.
Until then, `--deterministic-count` stays default-OFF; the prompt-level
temporal_event_count routing + anti-undercount directive remain the best safe
counting improvement (2/6 recovered, no regressions).
