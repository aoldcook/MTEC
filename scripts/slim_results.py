#!/usr/bin/env python
"""Build a git-committable copy of the experiment outputs.

Keeps everything needed to recompute the paper's numbers (predictions, ground
truth, token usage, timings, all summary/report text) and drops the bulky
compressed-prompt payloads and media, which stay on the server.
"""
import json
import os
import shutil
import sys

SRC_ROOT = "/root/autodl-tmp/MTEC/outputs"
DST_ROOT = "/root/autodl-tmp/MTEC/results_public"

# Per-record fields that carry the bulk (compressed prompt payloads + audits).
DROP_FIELDS = {
    "low_resolution_anchor",
    "structured_evidence_prompt",
    "computed_evidence_prompt",
    "pre_api_input_audit",
    "final_input_audit",
}

# Directories to publish.
INCLUDE_DIRS = ["ablation_20260701", "PAPER_ARCHIVE_20260625"]

# Text artifacts copied verbatim.
COPY_EXT = {".txt", ".md", ".csv", ".log", ".srt", ".yaml", ".yml", ".sh"}
SLIM_EXT = {".json", ".jsonl"}
SKIP_EXT = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".wav", ".mp3",
            ".gz", ".tar", ".zip", ".pt", ".pth", ".bin"}

stats = {"copied": 0, "slimmed": 0, "skipped": 0,
         "bytes_in": 0, "bytes_out": 0, "errors": 0}


def slim_json(src, dst):
    """Rewrite a .json/.jsonl file without the heavy fields."""
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    stripped = text.lstrip()

    def clean(rec):
        if isinstance(rec, dict):
            return {k: v for k, v in rec.items() if k not in DROP_FIELDS}
        return rec

    if stripped.startswith("["):
        recs = json.loads(text)
        out = json.dumps([clean(r) for r in recs], ensure_ascii=False, indent=1)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(out)
    elif stripped.startswith("{") and "\n{" not in stripped.strip():
        # single JSON object
        obj = json.loads(text)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(clean(obj), f, ensure_ascii=False, indent=1)
    else:
        # JSONL
        with open(dst, "w", encoding="utf-8") as f:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    f.write(json.dumps(clean(json.loads(line)),
                                       ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    f.write(line + "\n")


def main():
    if os.path.isdir(DST_ROOT):
        shutil.rmtree(DST_ROOT)
    for d in INCLUDE_DIRS:
        src_dir = os.path.join(SRC_ROOT, d)
        if not os.path.isdir(src_dir):
            print("MISSING: %s" % src_dir)
            continue
        for root, _dirs, files in os.walk(src_dir):
            rel = os.path.relpath(root, SRC_ROOT)
            out_dir = os.path.join(DST_ROOT, rel)
            for name in files:
                src = os.path.join(root, name)
                ext = os.path.splitext(name)[1].lower()
                try:
                    size = os.path.getsize(src)
                except OSError:
                    continue
                stats["bytes_in"] += size
                if ext in SKIP_EXT:
                    stats["skipped"] += 1
                    continue
                os.makedirs(out_dir, exist_ok=True)
                dst = os.path.join(out_dir, name)
                try:
                    if ext in SLIM_EXT:
                        slim_json(src, dst)
                        stats["slimmed"] += 1
                    elif ext in COPY_EXT or ext == "":
                        shutil.copy2(src, dst)
                        stats["copied"] += 1
                    else:
                        stats["skipped"] += 1
                        continue
                    stats["bytes_out"] += os.path.getsize(dst)
                except Exception as exc:
                    stats["errors"] += 1
                    print("ERROR %s: %s" % (src, exc), file=sys.stderr)

    print("copied   : %d" % stats["copied"])
    print("slimmed  : %d" % stats["slimmed"])
    print("skipped  : %d (media/archives)" % stats["skipped"])
    print("errors   : %d" % stats["errors"])
    print("bytes in : %.2f GB" % (stats["bytes_in"] / 1e9))
    print("bytes out: %.1f MB" % (stats["bytes_out"] / 1e6))


if __name__ == "__main__":
    main()
