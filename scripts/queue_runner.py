#!/usr/bin/env python
"""Generic bounded-concurrency job queue for the ablation study.

Reads JSONL job specs {"name","cmd":[argv],"log":path,"env":{...},"cwd":path},
runs at most --max-concurrent at a time, writes a live status JSON, and appends
one line per launch/completion to stdout (captured by the daemon's own log).
"""
import os
import sys
import json
import time
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-file", required=True)
    ap.add_argument("--max-concurrent", type=int, default=3)
    ap.add_argument("--status-file", required=True)
    ap.add_argument("--poll-seconds", type=float, default=15.0)
    args = ap.parse_args()

    jobs = [json.loads(l) for l in open(args.queue_file) if l.strip()]
    print("loaded %d jobs" % len(jobs), flush=True)

    idx = 0
    completed = 0
    failed = 0
    procs = []  # list of [popen, job, logfile_handle]

    def launch(job):
        env = os.environ.copy()
        env.update(job.get("env", {}))
        os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
        logf = open(job["log"], "a")
        logf.write("\n===== LAUNCH %s =====\n" % time.ctime())
        logf.flush()
        p = subprocess.Popen(job["cmd"], cwd=job.get("cwd", "/root/autodl-tmp/MTEC"),
                              env=env, stdout=logf, stderr=subprocess.STDOUT)
        return p, logf

    while idx < len(jobs) or procs:
        while len(procs) < args.max_concurrent and idx < len(jobs):
            job = jobs[idx]
            idx += 1
            p, logf = launch(job)
            procs.append([p, job, logf])
            print("LAUNCH [%d/%d] %s pid=%d" % (idx, len(jobs), job["name"], p.pid), flush=True)

        still = []
        for p, job, logf in procs:
            rc = p.poll()
            if rc is None:
                still.append([p, job, logf])
            else:
                logf.close()
                if rc == 0:
                    completed += 1
                    status = "OK"
                else:
                    failed += 1
                    status = "FAIL"
                print("DONE [%s rc=%s] %s completed=%d failed=%d remaining=%d" %
                      (status, rc, job["name"], completed, failed, len(jobs) - idx + len(still)), flush=True)
        procs = still

        with open(args.status_file, "w") as f:
            json.dump({
                "total": len(jobs), "launched": idx, "completed": completed, "failed": failed,
                "running": [j["name"] for _, j, _ in procs], "ts": time.time(),
            }, f, indent=2)

        if idx >= len(jobs) and not procs:
            break
        time.sleep(args.poll_seconds)

    print("QUEUE_ALL_DONE total=%d completed=%d failed=%d" % (len(jobs), completed, failed), flush=True)


if __name__ == "__main__":
    main()
