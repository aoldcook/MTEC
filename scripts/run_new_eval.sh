#!/bin/bash
[ -f "$(dirname "$0")/../.env" ] && . "$(dirname "$0")/../.env"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate venv
export BAILIAN_API_KEY="${BAILIAN_API_KEY:?BAILIAN_API_KEY not set - see .env.example}"
cd /root/autodl-tmp/MTEC
DS=$1   # tempcompass | nextqa
python scripts/run_modelscope_mtec_anchor_api_full.py \
  --modalities video --model qwen3.7-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --api-key-env BAILIAN_API_KEY \
  --answer-model qwen3.7-plus --answer-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --answer-api-key-env BAILIAN_API_KEY \
  --evidence-pass true --prompt-style compact --evidence-prompt-style minimal \
  --video-anchor-policy auto --global-timeline-pass true --video-transcript-backend none \
  --oss-media-upload auto --cleanup-record-artifacts \
  --videomme-metadata data/datasets/$DS/${DS}_meta.parquet \
  --video-zips-dir data/datasets/$DS/videos \
  --limit-per-modality 300 --output-dir outputs/eval_${DS}_300_20260624
