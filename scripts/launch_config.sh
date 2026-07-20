#!/bin/bash
# Idempotent single-launch of one ablation config's queue daemon.
# Guards against duplicate daemons (SSH-retry storms): if a daemon for this
# label+queue is already running, it does nothing.
# Usage: launch_config.sh <LABEL> [max_concurrent] [queue_file] [status/daemon suffix]
set -e
LABEL="$1"
MAXC="${2:-3}"
QFILE="${3:-queue.txt}"
SUFFIX="${4:-}"
ROOT=/root/autodl-tmp/MTEC
PY=/root/miniconda3/envs/venv/bin/python
D="$ROOT/outputs/ablation_20260701/runs/$LABEL"

if pgrep -f "queue_runner.py.*$LABEL/$QFILE" >/dev/null 2>&1; then
  echo "ALREADY_RUNNING $LABEL/$QFILE"
  exit 0
fi

cd "$ROOT"
setsid "$PY" scripts/queue_runner.py \
  --queue-file "$D/$QFILE" \
  --max-concurrent "$MAXC" \
  --status-file "$D/status${SUFFIX}.json" \
  --poll-seconds 15 </dev/null > "$D/daemon${SUFFIX}.log" 2>&1 &
disown
sleep 2
echo "LAUNCHED $LABEL/$QFILE"
pgrep -f "queue_runner.py.*$LABEL/$QFILE" | head -1 | sed 's/^/daemon_pid=/'
