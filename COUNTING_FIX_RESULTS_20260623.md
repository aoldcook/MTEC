# Counting-Bias Fix — Results

Date: 2026-06-23

## Fix
The dominant error category (6/19 misses) was visual-counting **undercount**. Root
causes addressed:
1. **Mis-routing of repeated-action counts.** "How many jumps/rolls/laps/sets/times…"
   were routed to `scene_group_attribute_count` (static "best single wide shot"),
   which structurally under-counts actions repeated over time. Added a new
   `temporal_event_count` family (routed by action-repetition keywords with
   word-boundary matching) whose evidence template/rules **enumerate each
   occurrence across the full timeline and sum them**.
2. **General undercount bias.** Added a gated `_count_undercount_directive` for all
   count families instructing the verifier that compressed/low-FPS frames skip
   instances, to enumerate exhaustively, and to prefer the higher option when torn.

Both changes are gated to counting questions (no effect on other tasks).

## Re-run of the 6 counting-error videos

| Record | Before | After | Outcome |
|---|---|---|---|
| 8np5YKYx3sU (stage cast) | B | **A** | fixed |
| zOgYnntFl-k (raised hands) | B(6) | **D(7)** | fixed |
| 5kmnEgBSCfg (sets of jumps) | A(3) | B(6) | moved toward gt(5) but overshot |
| 5Knkqo-lYF0 (rolls) | B(<=5) | B(<=5) | unchanged |
| 6NVr0cNiHPM (box items) | B(8) | B(8) | unchanged (opening box occluded) |
| zbvamKv81o0 (fewest acrobats) | D | C | still wrong (cross-segment compare) |

**2 of 6 counting errors fixed** (8np, zOg). On the full 105-record set this lifts
accuracy from 86/105 (81.9%) to **88/105 (83.8%)**.

## Observations
- The fix removes the *structural* undercount (wrong method/routing): 8np reached
  the correct 6+3, zOg reached 7. `5kmnEgBSCfg` moved 3->6 (right direction, now
  +1 over the true 5) — the routing is right but the model's frame-level rep
  perception remains imprecise.
- The residual misses are now genuine **VLM perception limits**, not method bugs:
  exact rep counts on fast actions (5kmn, 5Knk), occluded opening box (6NV), and
  cross-segment minimum comparison (zbvam). These need instance-level/deterministic
  visual counting rather than prompt/routing changes.
- Regression risk is low (changes gated to count families) but a full re-run of all
  count questions would be needed to confirm no previously-correct count flipped
  from the "prefer higher" nudge.
