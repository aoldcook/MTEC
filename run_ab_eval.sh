#!/bin/bash
[ -f "$(dirname "$0")/.env" ] && . "$(dirname "$0")/.env"
set -u
source /root/miniconda3/etc/profile.d/conda.sh && conda activate venv
export BAILIAN_API_KEY="${BAILIAN_API_KEY:?BAILIAN_API_KEY not set - see .env.example}"
KEYS="video:5wLv3pCqZ9o:013-1 video:5_fXicEnKKk:016-1 video:zKyWRRJQbkM:033-1 video:5ksVshqVuiM:040-1 video:zBnKgwnn7i4:057-1 video:8TNPeimqOO0:060-1 video:7iXM5aq53Ts:073-1 video:Z-rHofd6g2Q:122-1 video:6Z_XNM_iT4g:134-1 video:80p80ynsZ78:139-1 video:7R1eNHvfspk:147-1 video:5fPtlxR3s3c:149-1 video:zNxi2s36tS0:163-1 video:84EpEwIVFdU:174-1 video:6DO8yOVYXr0:177-1 video:8np5YKYx3sU:181-1 video:5Knkqo-lYF0:216-1 video:zPx3EibuO_w:232-1 video:ZO12ZY38FEw:245-1 video:6NVr0cNiHPM:248-1"
run() {
  local name="$1"; shift
  echo "=== START $name $(date +%T) ==="
  python scripts/run_modelscope_mtec_anchor_api_full.py --modalities video     --model qwen3.7-plus --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --api-key-env BAILIAN_API_KEY     --answer-model qwen3.7-plus --answer-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --answer-api-key-env BAILIAN_API_KEY     --evidence-pass true --prompt-style compact --evidence-prompt-style minimal     --video-anchor-policy auto --global-timeline-pass true     --precomputed-subtitles-dir outputs/videomme_asr_subtitles_base_en_no_vad     "$@" --video-record-keys $KEYS --output-dir outputs/ab_${name}_20260622 > outputs/ab_${name}.log 2>&1
  echo "=== DONE $name $(date +%T) ==="
}
run baseline
run c1_cache --enable-prompt-cache
run c2_strip --strip-anchor-metadata
run c12_both --enable-prompt-cache --strip-anchor-metadata
echo ALL_DONE
