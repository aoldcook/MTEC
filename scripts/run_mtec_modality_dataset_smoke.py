import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def patch_torch_compat() -> None:
    if not hasattr(torch, "compiler"):
        torch.compiler = types.SimpleNamespace()
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False
    try:
        import torch.utils._pytree as pytree
    except Exception:
        return
    if not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):
        def register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs):
            return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

        pytree.register_pytree_node = register_pytree_node


def resolve_path(path_text: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_text)))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def extract_letter(value: Any) -> Optional[str]:
    if value is None:
        return None
    match = re.search(r"\(?\s*([A-Fa-f])\s*\)?", str(value))
    return match.group(1).upper() if match else None


class QwenVLRunner:
    def __init__(self, model_path: Path, device: str, dtype: str):
        patch_torch_compat()
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.process_vision_info = process_vision_info
        self.device = device
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

    def generate(self, content: List[Dict[str, Any]], max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def save_pil_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, format="JPEG", quality=92)
    return path


def run_image_dataset(
    runner: QwenVLRunner,
    dataset_name: str,
    loader_args: List[str],
    output_dir: Path,
    max_new_tokens: int,
) -> Dict[str, Any]:
    from datasets import load_dataset

    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "modality": "image",
        "status": "started",
    }
    started = time.perf_counter()
    try:
        dataset = load_dataset(*loader_args, split="test", streaming=True)
        item = next(iter(dataset))
        image = item["image"]
        image_path = save_pil_image(image, output_dir / "media" / f"{dataset_name}_sample.jpg")
        question = item.get("prompt") or item.get("question") or ""
        if "Please answer" not in question:
            question = f"{question}\nRespond with only the option letter."
        response = runner.generate(
            [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
            max_new_tokens=max_new_tokens,
        )
        prediction = extract_letter(response)
        ground_truth = extract_letter(item.get("answer"))
        record.update(
            {
                "status": "completed",
                "image": str(image_path),
                "question": question,
                "ground_truth": ground_truth,
                "Answer": prediction,
                "raw_response": response,
                "correct": bool(prediction and ground_truth and prediction == ground_truth),
                "source_item": {
                    key: str(item.get(key))
                    for key in ("filename", "image_path", "source", "task")
                    if item.get(key) is not None
                },
            }
        )
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def download_video(url: str, output_template: str) -> Optional[str]:
    try:
        from yt_dlp import YoutubeDL
    except Exception:
        return None
    options = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": output_template,
        "socket_timeout": 12,
        "retries": 1,
        "fragment_retries": 1,
        "quiet": True,
        "noplaylist": True,
        "max_filesize": 200 * 1024 * 1024,
        "download_sections": ["*00:00:00-00:01:30"],
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    requested = info.get("requested_downloads") or []
    if requested and requested[0].get("filepath"):
        return requested[0]["filepath"]
    filepath = ydl.prepare_filename(info)
    return filepath if Path(filepath).exists() else None


def run_videomme(
    runner: QwenVLRunner,
    dataset_root: Path,
    output_dir: Path,
    max_new_tokens: int,
    try_download: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "video-mme", "modality": "video", "status": "started"}
    started = time.perf_counter()
    try:
        df = pd.read_parquet(dataset_root / "video-mme" / "videomme" / "test-00000-of-00001.parquet")
        row = df[df["duration"] == "short"].iloc[0].to_dict()
        question = row["question"] + "\n" + "\n".join(str(option) for option in row["options"])
        question += "\nRespond with only the option letter."
        record.update(
            {
                "question_id": row.get("question_id"),
                "videoID": row.get("videoID"),
                "url": row.get("url"),
                "question": question,
                "ground_truth": extract_letter(row.get("answer")),
            }
        )
        if not try_download:
            raise RuntimeError("Video download disabled.")
        video_path = download_video(
            str(row["url"]),
            str(output_dir / "media" / "videomme_%(id)s.%(ext)s"),
        )
        if not video_path:
            raise RuntimeError("Video download did not produce a local file.")
        response = runner.generate(
            [
                {"type": "video", "video": video_path, "fps": 1.0},
                {"type": "text", "text": question},
            ],
            max_new_tokens=max_new_tokens,
        )
        prediction = extract_letter(response)
        record.update(
            {
                "status": "completed",
                "video": video_path,
                "Answer": prediction,
                "raw_response": response,
                "correct": bool(prediction and prediction == record["ground_truth"]),
            }
        )
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def run_music_avqa(dataset_root: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "music-avqa", "modality": "audio_video", "status": "started"}
    try:
        items = json.loads((dataset_root / "music-avqa" / "avqa-val.json").read_text(encoding="utf-8"))
        item = items[0]
        video_id = str(item.get("video_id"))
        media_matches = []
        for suffix in ("*.mp4", "*.wav", "*.mp3", "*.flac", "*.m4a"):
            media_matches.extend((dataset_root / "music-avqa").rglob(f"*{video_id}*{suffix[-4:]}"))
        record.update(
            {
                "question_id": item.get("question_id"),
                "video_id": video_id,
                "question": item.get("question_content"),
                "ground_truth": item.get("anser") or item.get("answer"),
            }
        )
        if not media_matches:
            record.update(
                {
                    "status": "skipped_no_media",
                    "Error": "MUSIC-AVQA metadata is present, but no local audio/video media files were found.",
                }
            )
        else:
            record.update({"status": "media_found", "media": [str(path) for path in media_matches[:5]]})
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-stage MTEC-Prompt++ dataset smoke tests by modality.")
    parser.add_argument("--model-path", default="models/qwen2.5-vl-3b")
    parser.add_argument("--datasets-root", default="data/datasets")
    parser.add_argument("--output-dir", default="outputs/modality_dataset_smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--try-video-download", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = resolve_path(args.datasets_root)

    records: List[Dict[str, Any]] = []
    runner = QwenVLRunner(resolve_path(args.model_path), args.device, args.dtype)
    records.append(
        run_image_dataset(runner, "cv-bench-2d", ["nyu-visionx/CV-Bench", "2D"], output_dir, args.max_new_tokens)
    )
    records.append(
        run_image_dataset(runner, "realworldqa", ["lmms-lab/RealWorldQA"], output_dir, args.max_new_tokens)
    )
    records.append(run_videomme(runner, dataset_root, output_dir, args.max_new_tokens, args.try_video_download))
    records.append(run_music_avqa(dataset_root))

    summary = {
        "model": str(resolve_path(args.model_path)),
        "records": records,
        "counts": {
            "completed": sum(1 for item in records if item.get("status") == "completed"),
            "failed": sum(1 for item in records if item.get("status") == "failed"),
            "skipped": sum(1 for item in records if str(item.get("status", "")).startswith("skipped")),
        },
    }
    result_path = output_dir / "mtec_modality_dataset_smoke_results.json"
    result_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_dir / "mtec_modality_dataset_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {result_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    for item in records:
        print(
            f"{item.get('dataset')} [{item.get('modality')}]: {item.get('status')} "
            f"pred={item.get('Answer')} gt={item.get('ground_truth')} error={item.get('Error', '')}",
            flush=True,
        )
    os._exit(0)


if __name__ == "__main__":
    main()
