#!/usr/bin/env python
"""(1) Record a video<->sample manifest for the existing on-disk datasets so the
original videos can be retrieved later, and (2) build the MLVU long-video sample
(meta parquet + per-video ModelScope repo-path stream map), appending it to the manifest.
"""
import os, re, json, glob, random, zipfile, urllib.request
import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
DS = os.path.join(ROOT, "data/datasets")
MANIFEST = os.path.join(DS, "VIDEO_SAMPLE_MANIFEST.json")
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
SEED = 42
MLVU_N = 180
MLVU_MIN, MLVU_MAX = 600, 3600  # 10-60 min

manifest = {"generated": "2026-06-25", "datasets": {}}


def videomme_manifest():
    meta = pd.read_parquet(os.path.join(DS, "video-mme/videomme/test-00000-of-00001.parquet"))
    # which videoIDs are local, and in which zip
    loc = {}
    for z in glob.glob(os.path.join(ROOT, "data/modelscope/video-mme-zips/*.zip")):
        with zipfile.ZipFile(z) as a:
            for i in a.infolist():
                if i.filename.endswith(".mp4"):
                    loc[os.path.splitext(os.path.basename(i.filename))[0]] = os.path.basename(z)
    rows = []
    sub = meta[meta["videoID"].astype(str).isin(loc.keys())]
    for _, r in sub.iterrows():
        vid = str(r["videoID"])
        rows.append({
            "record_key": f"video:{vid}:{r['question_id']}",
            "videoID": vid, "question_id": str(r["question_id"]),
            "duration_class": str(r["duration"]), "task_type": str(r["task_type"]),
            "original_source": str(r["url"]), "local_zip": loc.get(vid),
            "retrieval": f"YouTube {r['url']} ; or ModelScope Video-MME zip {loc.get(vid)}",
        })
    manifest["datasets"]["video-mme"] = {"n_questions": len(rows), "n_videos": sub["videoID"].nunique(),
                                          "source": "lmms-lab/Video-MME (YouTube urls)", "samples": rows}
    print("[manifest] video-mme:", len(rows), "questions")


def simple_manifest(name, parquet, source, retrieval_tmpl):
    df = pd.read_parquet(parquet)
    rows = []
    for _, r in df.iterrows():
        vid = str(r["videoID"]); qid = str(r["question_id"])
        rows.append({
            "record_key": f"video:{vid}:{qid}", "videoID": vid, "question_id": qid,
            "sub_category": str(r.get("sub_category")), "original_source": source,
            "retrieval": retrieval_tmpl.format(vid=vid),
        })
    manifest["datasets"][name] = {"n_questions": len(rows), "n_videos": df["videoID"].nunique(),
                                  "source": source, "samples": rows}
    print(f"[manifest] {name}:", len(rows), "questions")


def tree_paths():
    allf = []
    for pg in range(1, 12):
        d = json.load(urllib.request.urlopen(
            "https://modelscope.cn/api/v1/datasets/AI-ModelScope/MLVU/repo/tree?Revision=master&Recursive=true&PageSize=200&PageNumber=%d" % pg, timeout=30))
        f = (d.get("Data") or {}).get("Files") or []
        if not f:
            break
        allf += f
        if len(f) < 200:
            break
    byname = {}
    size = {}
    for x in allf:
        p = x.get("Path", "")
        if p.lower().endswith(".mp4"):
            parts = p.split("/")
            if len(parts) >= 4:
                byname[(parts[2], parts[3])] = p
                size[p] = x.get("Size", 0)
    return byname, size


def build_mlvu():
    byname, size = tree_paths()
    tasks = {"1_plotQA": "1_plotQA", "2_needle": "2_needle", "3_ego": "3_ego",
             "5_order": "5_order", "6_anomaly_reco": "6_anomaly_reco", "7_topic_reasoning": "7_topic_reasoning"}
    pool = []
    for jf in glob.glob(os.path.join(DS, "mlvu/json/*.json")):
        t = os.path.basename(jf)[:-5]
        if t not in tasks:
            continue
        for e in json.load(open(jf)):
            dur = e.get("duration", 0)
            if not (MLVU_MIN <= dur <= MLVU_MAX):
                continue
            path = byname.get((tasks[t], e["video"]))
            if not path:
                continue
            cands = list(e["candidates"])
            ans = str(e["answer"])
            if ans not in cands:
                continue
            pool.append({"task": t, "video": e["video"], "repo_path": path, "duration": dur,
                         "size": size.get(path, 0), "question": str(e["question"]),
                         "candidates": cands, "answer_letter": LETTERS[cands.index(ans)]})
    print("[mlvu] candidate long Qs in [%d,%d]s:" % (MLVU_MIN, MLVU_MAX), len(pool))
    # sample DISTINCT videos, balanced across tasks
    rng = random.Random(SEED)
    rng.shuffle(pool)
    seen_video = set()
    by_task = {}
    for e in pool:
        if e["video"] in seen_video:
            continue
        by_task.setdefault(e["task"], []).append(e)
        seen_video.add(e["video"])
    # round-robin across tasks until MLVU_N
    chosen = []
    idx = {t: 0 for t in by_task}
    tlist = sorted(by_task)
    while len(chosen) < MLVU_N and any(idx[t] < len(by_task[t]) for t in tlist):
        for t in tlist:
            if idx[t] < len(by_task[t]):
                chosen.append(by_task[t][idx[t]])
                idx[t] += 1
                if len(chosen) >= MLVU_N:
                    break
    print("[mlvu] chosen", len(chosen), "by task:",
          {t: sum(1 for c in chosen if c["task"] == t) for t in tlist})
    durs = sorted(c["duration"] for c in chosen)
    print("[mlvu] chosen duration sec: min%d med%d max%d  total_GB~%.1f" %
          (durs[0], durs[len(durs)//2], durs[-1], sum(c["size"] for c in chosen)/1e9))
    # videomme-style meta + stream map + manifest entries
    meta_rows, stream_map, man_rows = [], {}, []
    for i, c in enumerate(chosen):
        vid = os.path.splitext(c["video"])[0]
        qid = f"mlvu{i}"
        opts = [f"{LETTERS[j]}. {x}" for j, x in enumerate(c["candidates"])]
        meta_rows.append({"video_id": qid, "duration": "long", "domain": "MLVU",
                          "sub_category": c["task"], "url": "", "videoID": vid,
                          "question_id": qid, "task_type": c["task"],
                          "question": c["question"], "options": opts, "answer": c["answer_letter"]})
        stream_map[vid] = {"repo_path": c["repo_path"], "duration": c["duration"], "size": c["size"]}
        man_rows.append({"record_key": f"video:{vid}:{qid}", "videoID": vid, "question_id": qid,
                         "task_type": c["task"], "duration_sec": c["duration"],
                         "original_source": "AI-ModelScope/MLVU",
                         "retrieval": f"ModelScope AI-ModelScope/MLVU FilePath={c['repo_path']}"})
    pd.DataFrame(meta_rows).to_parquet(os.path.join(DS, "mlvu/mlvu_meta.parquet"), index=False)
    json.dump(stream_map, open(os.path.join(DS, "mlvu/mlvu_stream_map.json"), "w"), indent=1)
    manifest["datasets"]["mlvu-long"] = {"n_questions": len(man_rows), "n_videos": len(stream_map),
                                         "source": "AI-ModelScope/MLVU", "duration_range_sec": [MLVU_MIN, MLVU_MAX],
                                         "samples": man_rows}
    print("[mlvu] wrote mlvu_meta.parquet + mlvu_stream_map.json")


if __name__ == "__main__":
    videomme_manifest()
    simple_manifest("nextqa", os.path.join(DS, "nextqa/nextqa_meta.parquet"),
                    "MTEB/NExT-QA (video embedded in parquet)",
                    "ModelScope MTEB/NExT-QA, video_id={vid} in data/test-*.parquet")
    simple_manifest("tempcompass", os.path.join(DS, "tempcompass/tempcompass_meta.parquet"),
                    "lmms-lab/TempCompass",
                    "ModelScope lmms-lab/TempCompass tempcompass_videos.zip member {vid}.mp4")
    build_mlvu()
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    print("WROTE", MANIFEST)
