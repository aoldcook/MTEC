#!/bin/bash
[ -f "$(dirname "$0")/../.env" ] && . "$(dirname "$0")/../.env"
# Polls SiliconFlow balance; if it drops below FLOOR, kills all SF ablation jobs
# (A_sf/D_sf/E_sf daemons + runners) to prevent overspend. Logs to SF_BALANCE.log.
SF_KEY="${SF_API_KEY:?SF_API_KEY not set - see .env.example}"
FLOOR=3.0
LOG=/root/autodl-tmp/MTEC/outputs/ablation_20260701/SF_BALANCE.log
PY=/root/miniconda3/envs/venv/bin/python
echo "[sf_watchdog] started $(date) floor=$FLOOR" >> "$LOG"
while true; do
  BAL=$(curl -s https://api.siliconflow.cn/v1/user/info -H "Authorization: Bearer $SF_KEY" \
        | "$PY" -c 'import sys,json
try: print(json.load(sys.stdin)["data"]["totalBalance"])
except: print("ERR")' 2>/dev/null)
  echo "[sf_watchdog] $(date) balance=$BAL" >> "$LOG"
  if [ "$BAL" != "ERR" ] && [ -n "$BAL" ]; then
    LOW=$("$PY" -c "print(1 if float('$BAL')<$FLOOR else 0)" 2>/dev/null)
    if [ "$LOW" = "1" ]; then
      echo "[sf_watchdog] $(date) BALANCE LOW ($BAL < $FLOOR) -> killing SF jobs" >> "$LOG"
      pkill -9 -f 'queue_runner.py.*_sf/queue.txt'
      pkill -9 -f 'runs/A_sf/'; pkill -9 -f 'runs/D_sf/'; pkill -9 -f 'runs/E_sf/'
      break
    fi
  fi
  sleep 120
done
