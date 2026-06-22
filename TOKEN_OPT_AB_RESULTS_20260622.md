# Token-Savings Modules — A/B Results & Findings

Date: 2026-06-22
Backend: Bailian/DashScope `qwen3.7-plus` (two-pass: evidence + final, plus
global-timeline and optional visual-context passes).

## Modules implemented (decoupled, independent default-OFF flags)

- **#1 `--enable-prompt-cache`** — reuse the evidence pass's exact media list for
  the final answer call so the provider can serve the repeated video/image prefix
  from context cache (identical visual evidence; only trailing text differs).
- **#2 `--strip-anchor-metadata`** — remove non-semantic anchor fields
  (compression/bytes/quality/strategy/paths/resolutions) from the prompt
  serialization only (saved records and attached media unchanged).

Both are independently toggleable for A/B; the prior `--drop-visual-count-clip-when-isolated`
remains a separate flag.

## Provider caching — what actually happens (probed directly)

- DashScope context caching **is real** for `qwen3.7-plus`: a repeated stable
  prefix ≥ ~2048 tokens caches (probe: 11k-token prefix → 10,880 cached; 4,425 →
  4,352). **`qwen-plus` does not** report caching.
- **Media caches too**: re-sending the same video cached **45,952 / 46,334**
  video tokens; partial-prefix caching works even when trailing text differs.
- **But the cache key includes request params**: `enable_thinking=false` vs
  omitted produced a cache **miss** between otherwise-identical calls. The
  evidence client (omit) and answer client (`false`) therefore had mismatched
  keys. Aligning them still left the in-pipeline final pass at `cached=0`.
- **Confound:** months/hours of prior testing on the 20-record set polluted the
  cache, so even the *baseline* shows ~33% cached (the evidence pass hits
  cross-run). This masks whether #1 helps on a cold cache. Per-record, the final
  pass shows `cached=0` in every in-pipeline configuration tried.

## A/B results on the 20-record regression set

| Config | n | Accuracy | mean token_saving_ratio | billed prompt tokens |
|---|---|---|---|---|
| baseline | 20 | 17/20 | 0.5926 | 983,499 |
| #1 cache | 20 | 17/20 | 0.5925 | 883,600 |
| #2 strip | (8, aborted) | 8/8 | n/a (partial) | n/a |

Conclusions:

1. **Module #2 is a no-op.** The prompt serializer (`format_compact_evidence_prompt`)
   already excludes all the metadata #2 targets — verified directly: the
   serialized prompt contains **0** occurrences of `compression`/`bytes`/`path`/
   `quality`/`resolution`/`strategy`, and stripping changes the prompt length by
   **0 characters** (27,700 → 27,700). The prompt is already lean; there is no
   metadata bloat to remove. (c2_strip was aborted once this was proven at the
   serialization level — a noisy run would add nothing.)
2. **Module #1 does not move the token-savings metric.** `token_saving_ratio` is
   computed from compressed *media bytes* vs raw video, so context caching (which
   only discounts billed API tokens) cannot change it (0.5926 vs 0.5925). The
   ~10% lower billed-token count for #1 is confounded by cross-run cache
   pollution and by the media-reuse altering content slightly; per-record the
   final pass never cached, so this is not a clean caching win.
3. **Accuracy is unchanged** by either module (17/20 both) — neither degrades
   accuracy, consistent with the design (identical visual evidence / no prompt
   content removed that the model uses).

## Best configuration

**Baseline defaults.** No module produces a real, accuracy-safe improvement of
the media-based `token_saving_ratio` on this pipeline/backend:
- #2 is a verified no-op (nothing to strip).
- #1 doesn't affect the media metric and cannot be cleanly shown to reduce billed
  tokens here (cache pollution); it also carries a small media-reuse risk.

The genuine token-savings headroom is in **media bytes** (video/image anchors
dominate), which cannot be reduced without accuracy risk — so baseline is the
safe optimum. Caching remains worth revisiting in a **cold-cache production**
setting with aligned request params (it provably reduces *billed* cost there,
without touching accuracy or the media metric).
