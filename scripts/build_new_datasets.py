#!/usr/bin/env python
"""Conform NExT-QA (MTEB, embedded video) and TempCompass (lmms-lab) to the
Video-MME metadata + videos_chunked_*.zip interface that the MTEC runner expects.

Outputs per dataset:
  data/datasets/<name>/<name>_meta.parquet     (videomme-style columns)
  data/datasets/<name>/videos/videos_chunked_00.zip  (<videoID>.mp4 members)
"""
import io
import os
import re
import sys
import glob
import random
import zipfile
import pandas as pd

LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
ROOT = "/root/autodl-tmp/MTEC"
SEED = 42
N_SAMPLE = 300


def vm_row(videoID, question_id, question, options, answer, duration, domain, sub_category, task_type):
    return {
        "video_id": str(question_id),
        "duration": duration,
        "domain": domain,
        "sub_category": sub_category,
        "url": "",
        "videoID": str(videoID),
        "question_id": str(question_id),
        "task_type": task_type,
        "question": question,
        "options": options,
        "answer": answer,
    }


def build_nextqa():
    name = "nextqa"
    base = os.path.join(ROOT, "data/datasets", name)
    shards = sorted(glob.glob(os.path.join(base, "shards", "*.parquet")))
    print("[nextqa] shards:", [os.path.basename(s) for s in shards])
    rows = []
    video_bytes = {}  # video_id -> bytes
    for shard in shards:
        df = pd.read_parquet(shard)
        sidx = re.search(r"test-(\d+)-", os.path.basename(shard)).group(1)
        for i, r in df.iterrows():
            vid = str(r["video_id"])
            cands = list(r["candidates"])
            ans_text = str(r["answer"])
            if ans_text not in cands:
                continue  # cannot resolve letter
            ans_letter = LETTERS[cands.index(ans_text)]
            options = [f"{LETTERS[j]}. {c}" for j, c in enumerate(cands)]
            qid = f"{vid}_{sidx}_{i}"
            rows.append((vm_row(vid, qid, str(r["question"]), options, ans_letter,
                                "short", "NExT-QA", "causal_temporal", "nextqa"),
                         r["video"]["bytes"]))
    print("[nextqa] total resolvable questions:", len(rows))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:N_SAMPLE]
    meta = [m for m, _ in rows]
    # collect unique videos needed
    for m, vb in rows:
        video_bytes.setdefault(m["videoID"], vb)
    print(f"[nextqa] sampled {len(meta)} questions over {len(video_bytes)} videos")
    # write videos zip
    vdir = os.path.join(base, "videos")
    os.makedirs(vdir, exist_ok=True)
    zpath = os.path.join(vdir, "videos_chunked_00.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for vid, vb in video_bytes.items():
            z.writestr(f"{vid}.mp4", vb)
    # write meta
    mpath = os.path.join(base, f"{name}_meta.parquet")
    pd.DataFrame(meta).to_parquet(mpath, index=False)
    print(f"[nextqa] wrote {mpath} and {zpath} ({os.path.getsize(zpath)/1e6:.0f}MB)")
    return meta


def parse_tempcompass_question(qtext):
    lines = qtext.split("\n")
    opt_re = re.compile(r"^\s*([A-G])\.\s*(.*)$")
    stem_lines, options = [], []
    for ln in lines:
        m = opt_re.match(ln)
        if m:
            options.append(f"{m.group(1)}. {m.group(2).strip()}")
        else:
            if ln.strip():
                stem_lines.append(ln.strip())
    return " ".join(stem_lines), options


def build_tempcompass():
    name = "tempcompass"
    base = os.path.join(ROOT, "data/datasets", name)
    df = pd.read_parquet(os.path.join(base, "multi-choice.parquet"))
    # available video members
    src_zip = os.path.join(base, "tempcompass_videos.zip")
    with zipfile.ZipFile(src_zip) as z:
        members = {os.path.splitext(os.path.basename(n))[0]: n
                   for n in z.namelist() if n.lower().endswith(".mp4")}
    rows = []
    for i, r in df.iterrows():
        vid = str(r["video_id"])
        if vid not in members:
            continue
        stem, options = parse_tempcompass_question(str(r["question"]))
        if len(options) < 2:
            continue
        ans = str(r["answer"]).strip()
        m = re.match(r"^\s*([A-G])\.", ans)
        if not m:
            # match by text
            ans_letter = None
            for o in options:
                if o.split(". ", 1)[-1].strip().lower() == ans.lower():
                    ans_letter = o[0]
                    break
            if ans_letter is None:
                continue
        else:
            ans_letter = m.group(1)
        dim = str(r["dim"])
        rows.append((vm_row(vid, f"tc{i}", stem, options, ans_letter,
                            "short", "TempCompass", dim, dim), vid))
    print("[tempcompass] total resolvable questions:", len(rows))
    # stratified-ish sample by dim
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:N_SAMPLE]
    meta = [m for m, _ in rows]
    needed = sorted({v for _, v in rows})
    print(f"[tempcompass] sampled {len(meta)} questions over {len(needed)} videos")
    # repackage only needed videos
    vdir = os.path.join(base, "videos")
    os.makedirs(vdir, exist_ok=True)
    zpath = os.path.join(vdir, "videos_chunked_00.zip")
    with zipfile.ZipFile(src_zip) as zin, zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zout:
        for vid in needed:
            with zin.open(members[vid]) as fh:
                zout.writestr(f"{vid}.mp4", fh.read())
    mpath = os.path.join(base, f"{name}_meta.parquet")
    pd.DataFrame(meta).to_parquet(mpath, index=False)
    print(f"[tempcompass] wrote {mpath} and {zpath} ({os.path.getsize(zpath)/1e6:.0f}MB)")
    # dim distribution
    print("[tempcompass] dim dist:", pd.DataFrame(meta)["task_type"].value_counts().to_dict())
    return meta


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "tempcompass"):
        build_tempcompass()
    if which in ("all", "nextqa"):
        build_nextqa()
    print("DONE")
