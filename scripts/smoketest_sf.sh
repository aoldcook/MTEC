#!/bin/bash
[ -f "$(dirname "$0")/../.env" ] && . "$(dirname "$0")/../.env"
set -e
cd /root/autodl-tmp/MTEC
source /root/miniconda3/etc/profile.d/conda.sh
conda activate venv
export SF_API_KEY="${SF_API_KEY:?SF_API_KEY not set - see .env.example}"
mkdir -p outputs/ablation_20260701/smoketest_sf

python scripts/run_modelscope_mtec_anchor_api_full.py --modalities video \
  --model Qwen/Qwen3-VL-32B-Instruct --base-url https://api.siliconflow.cn/v1 --api-key-env SF_API_KEY \
  --answer-model Qwen/Qwen3-VL-32B-Instruct --answer-base-url https://api.siliconflow.cn/v1 --answer-api-key-env SF_API_KEY \
  --answer-enable-thinking omit \
  --evidence-pass true --prompt-style compact --evidence-prompt-style minimal \
  --video-anchor-policy auto --global-timeline-pass true \
  --precomputed-subtitles-dir outputs/videomme_asr_subtitles_base_en_no_vad \
  --oss-media-upload off --cleanup-record-artifacts \
  --videomme-metadata data/datasets/video-mme/videomme/test-00000-of-00001.parquet \
  --video-zips-dir data/modelscope/video-mme-zips \
  --video-record-keys video:ZfNSxRiYfZQ:258-3 video:7R1eNHvfspk:147-2 video:1q-5IIyZL20:197-2 video:0ag_Qi5OEd0:522-3 video:7Hk9jct2ozY:321-2 video:1pHkv4KUiFY:314-2 \
  --output-dir outputs/ablation_20260701/smoketest_sf \
  > outputs/ablation_20260701/smoketest_sf.log 2>&1
echo "SMOKETEST_DONE rc=$?" >> outputs/ablation_20260701/smoketest_sf.log
