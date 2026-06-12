# MTEC-Prompt++

MTEC-Prompt++ 是一个面向多模态大模型评测与压缩输入实验的工程化原型。当前项目从原始的 Zoom-Refine 图像增强思路扩展为三模态流程：图像、视频、音频都先生成低成本结构锚点，再和结构化文本证据一起输入模型，从而在尽量保留任务关键信息的同时降低媒体输入体积和 token 成本。

当前重点验证的是：

- 图像：真实图像数据集上的低分辨率全局结构锚点。
- 视频：低 FPS 视频锚点、关键帧和事件边界摘要。
- 音频：真正低码率音频锚点和能量事件摘要，重点覆盖人声类音频。
- 报告：将每次评测结果输出为简单 HTML 表格，包含预测、标注、正确性、压缩率和 token 节省比例。

## 核心逻辑

MTEC-Prompt++ 的输入是双通道：

1. 低成本多模态结构锚点

   图像、视频、音频不会直接把完整原始媒体喂给模型，而是先转换为更小的结构锚点。

   - 图像：生成低分辨率 JPEG 全局锚点，保留整体布局和空间关系。
   - 视频：抽取低 FPS 帧，保留时间顺序、场景变化和事件边界，并生成低 FPS 视频片段。
   - 音频：生成低采样率、低码率单声道音频，同时提取能量窗口和显著事件段。

2. 结构化证据文本

   项目会根据问题类型生成紧凑的结构化 prompt，描述锚点类型、预算分配、关键媒体元信息和需要模型关注的证据。模型最终同时接收锚点媒体对象和结构化证据文本。

简化流程如下：

```text
原始图像/视频/音频
        |
        v
低成本结构锚点生成
        |
        +--> 图像低分辨率锚点
        +--> 视频低 FPS 锚点/关键帧
        +--> 音频低码率锚点/事件摘要
        |
        v
结构化证据 Prompt
        |
        v
Qwen2.5-VL / Qwen2.5-Omni 评测
        |
        v
JSON 结果 + HTML 表格报告
```

## 代码结构

```text
zoomrefine/
  mtec_prompt_plus.py        # 问题画像、预算分配、结构化证据 prompt
  mtec_media_pipeline.py     # 图像/视频/音频结构锚点生成

scripts/
  run_modelscope_mtec_anchor_7b.py   # Qwen2.5-Omni-7B 三模态 ModelScope 评测
  run_modelscope_multimodal_smoke.py # Qwen2.5-VL-3B 轻量烟测
  generate_mtec_table_report.py      # 简单 HTML 表格报告
  generate_mtec_report.py            # 较完整的 HTML 报告
```

本仓库只保存代码、配置和文档。模型权重、数据集和评测输出不应提交到 GitHub。

## 环境准备

建议使用 Python 3.9+ 和 CUDA 环境。AutoDL/A800 上可以使用已有 conda 环境，也可以新建环境：

```bash
conda create -n mtec python=3.10 -y
conda activate mtec
pip install -r requirements.txt
```

Qwen2.5-Omni 评测还需要安装对应版本的 `transformers`、`qwen-omni-utils`、`torch`、`pandas`、`pyarrow`、`opencv-python`、`soundfile` 等依赖。音频压缩优先使用系统 `ffmpeg`；没有 `ffmpeg` 时，项目会退回到 Python/soundfile 路径生成低采样率音频锚点。

## 本地目录约定

推荐目录结构：

```text
MTEC/
  models/
    qwen2.5-vl-3b/
    qwen2.5-omni-7b/
  data/
    modelscope/
      realworldqa/
      video-mme-zips/
      urbansound8k-noises/
      hearsed-dcase2016/
  outputs/
    modelscope_mtec_anchor_7b/
    reports/
```

这些目录已被 `.gitignore` 忽略：

- `models/`
- `data/`
- `outputs/`
- `.cache/`
- `.venv/`

## 3B 轻量烟测

3B 路径主要用于快速确认图像/视频/音频样本、结果格式和报告生成是否正常。

```bash
OMP_NUM_THREADS=1 python scripts/run_modelscope_multimodal_smoke.py \
  --model-path models/qwen2.5-vl-3b \
  --modelscope-root data/modelscope \
  --output-dir outputs/modelscope_multimodal_smoke \
  --dtype bfloat16
```

生成表格报告：

```bash
python scripts/generate_mtec_table_report.py \
  --inputs outputs/modelscope_multimodal_smoke/modelscope_multimodal_smoke_results.json \
  --output outputs/reports/mtec_modelscope_3b_table.html \
  --title "MTEC-Prompt++ ModelScope 3B Results Table"
```

## 7B 三模态评测

7B 路径使用 Qwen2.5-Omni，并按 MTEC-Prompt++ 双通道逻辑运行图像、视频和音频评测。

背景环境音评测：

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python scripts/run_modelscope_mtec_anchor_7b.py \
  --model-path models/qwen2.5-omni-7b \
  --modelscope-root data/modelscope \
  --image-parquet data/modelscope/realworldqa/data/test-00000-of-00002.parquet \
  --videomme-metadata data/datasets/video-mme/videomme/test-00000-of-00001.parquet \
  --audio-task background \
  --audio-parquet data/modelscope/urbansound8k-noises/data/test-00000-of-00001-40cf49999a374336.parquet \
  --output-dir outputs/modelscope_mtec_anchor_7b \
  --dtype bfloat16 \
  --max-new-tokens 64
```

人声音频评测：

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python scripts/run_modelscope_mtec_anchor_7b.py \
  --model-path models/qwen2.5-omni-7b \
  --modelscope-root data/modelscope \
  --image-parquet data/modelscope/realworldqa/data/test-00000-of-00002.parquet \
  --videomme-metadata data/datasets/video-mme/videomme/test-00000-of-00001.parquet \
  --audio-task voice \
  --voice-audio-parquet data/modelscope/hearsed-dcase2016/data/test-00000-of-00001.parquet \
  --output-dir outputs/modelscope_mtec_anchor_7b_voice \
  --dtype bfloat16 \
  --max-new-tokens 64
```

生成表格报告：

```bash
python scripts/generate_mtec_table_report.py \
  --inputs outputs/modelscope_mtec_anchor_7b_voice/modelscope_mtec_anchor_7b_results.json \
  --output outputs/reports/mtec_modelscope_mtec_anchor_7b_voice_table.html \
  --title "MTEC-Prompt++ ModelScope 7B Voice Audio Results Table"
```

表格字段包括：

- 模态和数据集
- 媒体样本链接
- 问题或 prompt
- 模型预测
- 标准答案
- 压缩率
- token 节省比例
- 是否正确
- 推理耗时
- 错误或备注

## 当前实验结论

截至当前实验，7B 路径已验证：

- RealWorldQA 真实图像样本可以通过低分辨率图像锚点完成回答。
- Video-MME 样本可以通过低 FPS 视频锚点完成回答。
- 背景环境音细分类不够稳定，尤其是空调、引擎怠速等持续低频声音容易混淆。
- 人声事件更符合当前音频能力边界；低码率音频锚点下可以识别 speech、cough、clearthroat 等主要人声事件。

因此，当前音频策略是：对背景音乐/环境音给出笼统描述即可，对人声类音频保留更高优先级。

## Git 注意事项

提交前建议检查：

```bash
git status --short
git check-ignore -v models data outputs
```

不要提交以下内容：

- 模型权重，如 `*.safetensors`、`*.bin`、`*.pt`、`*.pth`
- 数据集文件，如 parquet、zip、mp4、wav、mp3
- 评测输出，如 `outputs/` 下的 JSON、HTML、tar.gz

只提交代码、文档、轻量配置和必要脚本。

## License

本项目沿用原仓库的 Apache-2.0 License。
