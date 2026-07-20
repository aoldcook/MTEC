#!/usr/bin/env python
"""Create pruned key subsets (deterministic first-N of the fixed subsets) to speed
up the ablation. Verifies every pruned key is covered by the archive Config A so
comparisons stay valid. Writes outputs/ablation_20260701/keys_pruned/."""
import os
import json

ROOT = "/root/autodl-tmp/MTEC"
KD = os.path.join(ROOT, "outputs/ablation_20260701/keys")
PD = os.path.join(ROOT, "outputs/ablation_20260701/keys_pruned")
ARCH = os.path.join(ROOT, "outputs/PAPER_ARCHIVE_20260625")

PRUNE = {
    "sf_vmme_short": 50,
    "sf_vmme_medium": 30,
    "sf_nextqa": 50,
    "bailian_mlvu_long": 30,
}
ARCHIVE_FOR = {
    "sf_vmme_short": "videomme_300_compressed",
    "sf_vmme_medium": "videomme_300_compressed",
    "sf_nextqa": "nextqa_300_compressed",
    "bailian_mlvu_long": "mlvu_long_180_compressed",
}


def archive_completed_keys(sub):
    p = os.path.join(ARCH, ARCHIVE_FOR[sub], "results.jsonl")
    ks = set()
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        if r.get("status") == "completed":
            ks.add(r.get("record_key"))
    return ks


os.makedirs(PD, exist_ok=True)
for sub, n in PRUNE.items():
    keys = open(os.path.join(KD, sub + ".txt")).read().split()
    arch = archive_completed_keys(sub)
    # take first-N keys that ARE covered by the archive (skip the few uncovered)
    covered = [k for k in keys if k in arch]
    pruned = covered[:n]
    miss = [k for k in keys[:n] if k not in arch]
    open(os.path.join(PD, sub + ".txt"), "w").write(" ".join(pruned))
    print("%s: pruned=%d (all archive-covered), skipped_uncovered_in_first_%d=%d" % (
        sub, len(pruned), n, len(miss)))

# combined convenience file
short = open(os.path.join(PD, "sf_vmme_short.txt")).read().split()
med = open(os.path.join(PD, "sf_vmme_medium.txt")).read().split()
print("TOTALS: vmme=%d nextqa=%d mlvu=%d" % (
    len(short) + len(med),
    len(open(os.path.join(PD, "sf_nextqa.txt")).read().split()),
    len(open(os.path.join(PD, "bailian_mlvu_long.txt")).read().split())))
