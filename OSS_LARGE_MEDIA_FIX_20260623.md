# Long-Video Payload Fix (OSS upload) + Re-runs — Summary

Date: 2026-06-23
Environment: project conda env `/root/miniconda3/envs/venv` (dashscope + oss2
installed there). All runs executed inside this env.

## Step 2 — The fix: DashScope temporary-OSS upload for large media

Long videos failed previously because the compressed anchors were attached as
base64 **data-URIs**, which hit DashScope's limits:
- `Exceeded limit on max bytes per data-uri item : 20971520` (20 MB/item)
- `String value length ... exceeds the maximum allowed (28000000)` (~28 MB/request)

**Fix (`--oss-media-upload auto`):** large media is uploaded to DashScope's
temporary OSS via `dashscope.utils.oss_utils.OssUtils.upload(...)`, returning an
`oss://` URL that is passed in the chat request with header
`X-DashScope-OssResourceResolve: enable`. The request body then carries only the
short URL instead of tens of MB of base64. Videos (and any media file >
`--oss-threshold-bytes`, default 8 MB) are uploaded; small image crops stay
inline. Implemented in `media_content()` + `generate()`; committed `2fe7e7d`.

Verified directly: a 414 MB video uploaded and was processed with no payload
error.

## Step 3 — Re-run of the 6 previously-errored records

| Record | Original error | With OSS | Token saving |
|---|---|---|---|
| `7D-gxaie6UI` | PAYLOAD (20 MB) | **correct (A)** | 0.875 |
| `zxKPjD8urG4` | PAYLOAD (28 MB) | **correct (A)** | 0.885 |
| `74TEQfw6L60` | PAYLOAD (28 MB) | **correct (D)** | 0.870 |
| `6DbsOZU8mBM` | DataInspectionFailed | still fails | — |
| `8-aI8Fp2bPU` | DataInspectionFailed | still fails | — |
| `5iA7wZfxglE` | DataInspectionFailed | still fails | — |

**All 3 payload errors fixed → 3/3 correct, mean token saving 87.7%.** The other
3 are content-moderation rejections (`DataInspectionFailed`) — a separate issue
OSS does not address.

## Step 4 — 50 new videos (previously-unevaluated records), OSS enabled

The OSS fix eliminated payload failures across the long videos. An initial run
hit a server **disk-quota** wall (the 50 GB `/root/autodl-tmp` filled from
accumulated extracted media); after freeing this session's run media and
re-running the affected records with `--cleanup-record-artifacts` (deletes each
record's extracted media after metrics are written), the set completed.

**Final new-50 result (48 of 50 completed; 2 = content-moderation rejections):**

| Metric | Value |
|---|---|
| **Accuracy** | **37 / 48 = 77.1%** |
| **Mean token saving** | **70.6%** |
| **Positive-only token saving** | **73.6%** |
| **Median token saving** | **79.2%** |
| Failures | 2 (`DataInspectionFailed` content filter; 0 payload, 0 disk) |

## Observations

1. **The OSS fix fully resolves the long-video payload-limit failures** — every
   payload error became a successful, correct answer. Long videos also yield the
   highest token savings (~87%), since their raw size dwarfs the compressed
   anchors.
2. **Remaining failures are content moderation, not the pipeline.** DashScope's
   `DataInspectionFailed` rejects certain video content regardless of transport;
   OSS cannot change that. These are a small minority (2/50 here, 3/6 in the
   targeted re-run which was biased toward already-flagged videos).
3. **Operational note:** long videos generate large local anchors; run with
   `--cleanup-record-artifacts` for big sweeps to avoid filling the disk. The
   `token_saving_ratio` metric is unaffected by OSS (it measures compressed media
   bytes vs raw; OSS only changes transport).
4. **Accuracy at scale is consistent** (~77–80%) across the 100-record and 50-new
   representative samples, vs ~85% on the hard-case 20-set.
