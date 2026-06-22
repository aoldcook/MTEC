# Token-Savings Exploration (accuracy-first)

Date: 2026-06-22
Constraint: **accuracy is the top priority; any savings gain that degrades
accuracy is strictly prohibited.** Only accuracy-preserving optimizations are
acceptable.

## 1. Where the headroom actually is

On the 20-video regression set, **18 of 20 records already compress well
(~62–63% token saving) and are byte-identical to baseline** after the
blast-radius minimization. Only **one** record has negative headroom:

- `8np5YKYx3sU` (performing-cast count): representation is **expanded**
  (compression_ratio 1.85, token_saving 0.0). The main answer call carries
  **28,474 video tokens** (the dense `full_scene_group_count` anchor **plus**
  the original-resolution visual-count clip), and the dedicated isolated
  cast-count pass adds **21,122 more** — the full original video is effectively
  sent **twice**.

So the only meaningful savings headroom is the cast path, specifically the
duplicated original video.

## 2. Approaches evaluated

### A. Drop the redundant original clip from the final call once the isolated pass has the count — IMPLEMENTED (opt-in, default OFF)
Rationale: the isolated count-only pass already extracts the authoritative
`men=X, women=Y` from the full original clip. The main verifier then only needs
to **map that count to an option** (a text operation); it does not need the clip
attached again.

Measured (8np, flag ON):
- main-call video tokens **28,474 → 7,352**
- record token_saving_ratio **0.0 → 0.3715** (expansion eliminated)
- the clip is still consumed once (by the isolated pass), so counting accuracy
  is unaffected by the drop.

Accuracy status: **theoretically neutral** — in observed runs the main verifier
followed the isolated count whether or not the clip was attached (e.g. isolated
`5,3` → option B with the clip present, same as without). However, under the
nondeterministic backend (see §3) non-degradation **cannot be proven** with a
small sample, and two validation runs hit a platform content filter. Therefore
this optimization ships **behind a default-OFF flag**:

```
--drop-visual-count-clip-when-isolated
```

Default behavior is unchanged (clip retained) ⇒ zero accuracy risk by default.
Recommended to enable only after confirming parity on a deterministic backend or
via multi-run majority vote.

### B. Reduced-fps / lower-resolution clip for the isolated pass — REJECTED
The isolated count depends on resolution to distinguish backup performers;
downsampling risks lowering the count (5 instead of 6) → accuracy regression.
Not acceptable under the constraint.

### C. Lighten the main video-anchor policy for cast when the isolated pass ran — CANDIDATE (not implemented)
For cast questions the main call uses the dense `full_scene_group_count` policy.
Once counting is delegated to the isolated pass, the main call could use a
lighter policy (e.g. `light`). Larger savings than (A), but it removes visual
context the verifier may use as a sanity check ⇒ higher accuracy risk. Deferred
until (A) is validated.

### D. Trim verbose static verifier/evidence prompt text — REJECTED for now
The verifier prompt is long; trimming would cut prompt tokens on every record.
But the answer model is prompt-sensitive and nondeterministic, so any text change
can flip borderline cases. Not worth the accuracy risk under the current backend.

## 3. Backend caveats discovered

- **Nondeterminism:** `qwen3.7-plus` (DashScope) at temperature 0 with video is
  not deterministic; borderline answers and even the isolated count (`men=5` vs
  `men=6`) vary run-to-run. Reliable savings/accuracy validation needs a
  deterministic backend or multi-run majority vote.
- **Content inspection:** the celebrity-performance video intermittently triggers
  `data_inspection_failed` (HTTP 400) on full-video uploads — independent of these
  changes (also noted in the original report §13.4). A platform file-upload /
  object-storage path with a fallback would make the cast path more robust and
  also reduce base64 request size.

## 4. Recommendation

1. Keep (A) available and **enable it after** a parity check (deterministic
   backend or majority vote) — it removes the only token-expanding record with no
   change to which option the verifier selects.
2. Pursue the platform file-upload path to fix `data_inspection_failed` and
   further shrink request size.
3. Treat per-run accuracy deltas as noise; evaluate with majority vote.
