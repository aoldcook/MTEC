#!/bin/bash
[ -f "$(dirname "$0")/../.env" ] && . "$(dirname "$0")/../.env"
# Generic SF-regime (short/medium/nextqa) sharded launcher for the ablation study.
# Usage: launch_sf_job.sh <meta_parquet> <zips_dir> <keys_file> <output_dir> <n_shards> [extra runner args...]
set -e
cd /root/autodl-tmp/MTEC
source /root/miniconda3/etc/profile.d/conda.sh
conda activate venv
export SF_API_KEY="${SF_API_KEY:?SF_API_KEY not set - see .env.example}"

META="$1"; ZIPS="$2"; KEYSFILE="$3"; OUTDIR="$4"; NSHARDS="$5"; shift 5
EXTRA_ARGS=("$@")

mkdir -p "$OUTDIR"

# split keys round-robin into NSHARDS files
python - "$KEYSFILE" "$OUTDIR" "$NSHARDS" <<'PYEOF'
import sys
keys = open(sys.argv[1]).read().split()
outdir = sys.argv[2]
n = int(sys.argv[3])
shards = [[] for _ in range(n)]
for i, k in enumerate(keys):
    shards[i % n].append(k)
for i, s in enumerate(shards):
    open(f"{outdir}/shard_{i}_keys.txt", "w").write(" ".join(s))
    print(f"shard {i}: {len(s)} keys")
PYEOF

for i in $(seq 0 $((NSHARDS - 1))); do
  SHARD_KEYS=$(cat "$OUTDIR/shard_${i}_keys.txt")
  if [ -z "$SHARD_KEYS" ]; then continue; fi
  mkdir -p "$OUTDIR/s${i}"
  setsid python scripts/run_modelscope_mtec_anchor_api_full.py --modalities video \
    --model Qwen/Qwen3-VL-32B-Instruct --base-url https://api.siliconflow.cn/v1 --api-key-env SF_API_KEY \
    --answer-model Qwen/Qwen3-VL-32B-Instruct --answer-base-url https://api.siliconflow.cn/v1 --answer-api-key-env SF_API_KEY \
    --answer-enable-thinking omit \
    --evidence-pass true --prompt-style compact --evidence-prompt-style minimal \
    --video-anchor-policy auto --global-timeline-pass true \
    --precomputed-subtitles-dir outputs/videomme_asr_subtitles_base_en_no_vad \
    --oss-media-upload off --cleanup-record-artifacts \
    --videomme-metadata "$META" --video-zips-dir "$ZIPS" \
    "${EXTRA_ARGS[@]}" \
    --video-record-keys $SHARD_KEYS \
    --output-dir "$OUTDIR/s${i}" \
    < /dev/null > "$OUTDIR/s${i}.log" 2>&1 &
  disown
  echo "launched shard $i pid=$!"
done
