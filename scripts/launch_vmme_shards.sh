#!/bin/bash
[ -f "$(dirname "$0")/../.env" ] && . "$(dirname "$0")/../.env"
# Launch N parallel Video-MME runner shards over precomputed shard keyfiles.
source /root/miniconda3/etc/profile.d/conda.sh && conda activate venv
export BAILIAN_API_KEY="${BAILIAN_API_KEY:?BAILIAN_API_KEY not set - see .env.example}"
cd /root/autodl-tmp/MTEC
SH=outputs/vmme300_shards
for i in 0 1 2 3 4 5; do
  KEYS=$(cat $SH/shard_$i.txt)
  [ -z "$KEYS" ] && continue
  setsid bash -c "python scripts/run_modelscope_mtec_anchor_api_full.py \
    --modalities video --model qwen3.7-plus \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --api-key-env BAILIAN_API_KEY \
    --answer-model qwen3.7-plus --answer-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --answer-api-key-env BAILIAN_API_KEY \
    --evidence-pass true --prompt-style compact --evidence-prompt-style minimal \
    --video-anchor-policy auto --global-timeline-pass true \
    --precomputed-subtitles-dir outputs/videomme_asr_subtitles_base_en_no_vad \
    --oss-media-upload auto --cleanup-record-artifacts \
    --output-dir $SH/s$i --video-record-keys $KEYS > $SH/s${i}.log 2>&1" < /dev/null &
done
sleep 20
echo "STREAM_RUNNERS=$(ps aux|grep run_modelscope|grep -v grep|wc -l)"
