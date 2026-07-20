#!/bin/bash
# Watches all ablation output logs for quota/balance failures and kills the
# affected API's runner processes to conserve credit. Runs forever in background.
cd /root/autodl-tmp/MTEC
LOGDIR=outputs/ablation_20260701
ALERT=$LOGDIR/BALANCE_ALERTS.log
mkdir -p "$LOGDIR"
touch "$ALERT"

echo "[balance_monitor] started $(date)" >> "$ALERT"

while true; do
  # Bailian check: scan any log/jsonl under the ablation tree used by bailian runs
  BAILIAN_HIT=$(grep -RslE "Access denied|insufficient_quota|good standing" "$LOGDIR" 2>/dev/null | xargs -r grep -lE "BAILIAN|bailian|dashscope" 2>/dev/null | head -1)
  # generic hit even without api-name context, still worth capturing
  ANY_HIT=$(grep -RlE "Access denied, please make sure your account is in good standing|insufficient_quota" "$LOGDIR" 2>/dev/null | head -1)

  if [ -n "$ANY_HIT" ]; then
    TS=$(date)
    echo "[balance_monitor] $TS QUOTA HIT detected in: $ANY_HIT" >> "$ALERT"
    grep -E "Access denied, please make sure your account is in good standing|insufficient_quota" "$ANY_HIT" | tail -3 >> "$ALERT"

    # Determine which API this file's run used, by checking the matching launch script /proc cmdline
    for pid in $(pgrep -f "run_modelscope_mtec_anchor_api_full.py"); do
      CMD=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
      OUTDIR=$(echo "$CMD" | grep -oE -- '--output-dir [^ ]+' | awk '{print $2}')
      if [ -n "$OUTDIR" ] && [[ "$ANY_HIT" == "$OUTDIR"* || "$ANY_HIT" == *"$OUTDIR"* ]]; then
        echo "[balance_monitor] $TS killing pid=$pid outdir=$OUTDIR to conserve credit" >> "$ALERT"
        kill -TERM "$pid" 2>/dev/null
      fi
    done
  fi
  sleep 30
done
