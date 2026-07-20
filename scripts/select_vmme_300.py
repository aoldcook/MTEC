#!/usr/bin/env python
"""Select balanced new Video-MME record keys to bring the total to 300 (100 short /
100 medium / 100 long), excluding already-completed records, and split them into N
runtime-balanced shard keyfiles (interleaved by duration so long records spread evenly).
"""
import os, json, glob, zipfile, pathlib
import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
POOL = os.path.join(ROOT, "data/modelscope/video-mme-zips")
EXISTING_DIRS = ["eval100_baseline_20260622", "new50_oss_20260623", "new50_redo_20260623",
                 "payload3_oss_20260623", "err6_oss_20260623", "ab_baseline_20260622"]
TARGET = {"short": 100, "medium": 100, "long": 100}
NSHARDS = 6
OUTDIR = os.path.join(ROOT, "outputs/vmme300_shards")


def done_keys():
    ks = set()
    for d in EXISTING_DIRS:
        for f in glob.glob(os.path.join(ROOT, "outputs", d, "*results.jsonl")):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("status") == "completed" and str(r.get("record_key", "")).startswith("video:"):
                        ks.add(r["record_key"])
                except Exception:
                    pass
    return ks


def pool_videos():
    have = set()
    for z in glob.glob(os.path.join(POOL, "*.zip")):
        try:
            with zipfile.ZipFile(z) as a:
                for i in a.infolist():
                    if i.filename.endswith(".mp4"):
                        have.add(pathlib.Path(i.filename).stem)
        except Exception as e:
            print("  SKIP bad zip", os.path.basename(z), e)
    return have


def main():
    meta = pd.read_parquet(META)
    have = pool_videos()
    done = done_keys()
    meta = meta[meta["videoID"].astype(str).isin(have)].copy()
    meta["rk"] = meta.apply(lambda r: f"video:{r['videoID']}:{r['question_id']}", axis=1)
    done_by_dur = meta[meta["rk"].isin(done)]["duration"].value_counts().to_dict()
    print("pool videos:", len(have), " completed:", len(done), " done_by_dur:", done_by_dur)

    new_keys = {}
    for dur, tgt in TARGET.items():
        have_done = sum(1 for k in done if k in set(meta[meta.duration == dur]["rk"]))
        need = tgt - have_done
        avail = meta[(meta.duration == dur) & (~meta["rk"].isin(done))].sort_values(["videoID", "question_id"])
        take = avail["rk"].tolist()[: max(need, 0)]
        new_keys[dur] = take
        print(f"  {dur}: target={tgt} done={have_done} need={need} available={len(avail)} selected={len(take)}")

    # assign EACH duration's keys round-robin across all shards so long records
    # (the slow ones) spread evenly -> balanced shard runtime
    os.makedirs(OUTDIR, exist_ok=True)
    shards = [[] for _ in range(NSHARDS)]
    allnew = []
    for dur in ["long", "medium", "short"]:
        for j, k in enumerate(new_keys[dur]):
            shards[j % NSHARDS].append(k)
            allnew.append(k)
    print("total new keys:", len(allnew), " => grand total:", len(done) + len(allnew))
    for i, s in enumerate(shards):
        open(os.path.join(OUTDIR, f"shard_{i}.txt"), "w").write(" ".join(s))
        print(f"  shard {i}: {len(s)} keys")
    open(os.path.join(OUTDIR, "all_new_keys.txt"), "w").write("\n".join(allnew))
    json.dump({"done": sorted(done), "new": allnew, "target": TARGET},
              open(os.path.join(OUTDIR, "plan.json"), "w"))


if __name__ == "__main__":
    main()
