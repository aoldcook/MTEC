import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import types
import zipfile
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


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


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def estimated_tokens(byte_count: int) -> int:
    return max(1, ceil(max(0, int(byte_count)) / 4))


def add_compression_metrics(
    record: Dict[str, Any],
    source_bytes: int,
    evidence_bytes: int,
    note: str,
) -> None:
    original_tokens = estimated_tokens(source_bytes)
    compressed_tokens = estimated_tokens(evidence_bytes)
    record.update(
        {
            "source_bytes": int(source_bytes),
            "evidence_bytes": int(evidence_bytes),
            "compression_ratio": round(evidence_bytes / source_bytes, 4) if source_bytes > 0 else None,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_saving_ratio": round(max(0.0, 1.0 - compressed_tokens / original_tokens), 4),
            "compression_note": note,
        }
    )


class QwenVLRunner:
    def __init__(self, model_path: Path, device: str, dtype: str):
        patch_torch_compat()
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32
        self.device = device
        self.process_vision_info = process_vision_info
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


def run_image_cv_bench(runner: QwenVLRunner, output_dir: Path, max_new_tokens: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "cv-bench-2d", "modality": "image", "status": "started"}
    started = time.perf_counter()
    try:
        from datasets import load_dataset

        dataset = load_dataset("nyu-visionx/CV-Bench", "2D", split="test", streaming=True)
        item = next(iter(dataset))
        image_path = save_pil_image(item["image"], output_dir / "media" / "cv_bench_2d_sample.jpg")
        question = item.get("prompt") or item.get("question") or ""
        question = f"{question}\nRespond with only the option letter."
        response = runner.generate(
            [{"type": "image", "image": str(image_path)}, {"type": "text", "text": question}],
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
            }
        )
        add_compression_metrics(
            record,
            image_path.stat().st_size,
            image_path.stat().st_size,
            "Image smoke passes the local image directly; token counts are byte/4 proxy estimates.",
        )
    except Exception as exc:
        fallback_error = f"{type(exc).__name__}: {exc}"
        try:
            image_path = REPO_ROOT / "asset" / "Example.jpg"
            question = "Describe the main visible subject in this image and answer concisely."
            response = runner.generate(
                [{"type": "image", "image": str(image_path)}, {"type": "text", "text": question}],
                max_new_tokens=max_new_tokens,
            )
            record.update(
                {
                    "dataset": "repo-example-image-fallback",
                    "status": "completed",
                    "image": str(image_path),
                    "question": question,
                    "ground_truth": None,
                    "Answer": response.strip(),
                    "raw_response": response,
                    "correct": None,
                    "note": f"CV-Bench streaming failed, so the local repository image was used. Original error: {fallback_error}",
                }
            )
            add_compression_metrics(
                record,
                image_path.stat().st_size,
                image_path.stat().st_size,
                "Image fallback passes the local image directly; token counts are byte/4 proxy estimates.",
            )
        except Exception as fallback_exc:
            record.update(
                {
                    "status": "failed",
                    "Error": f"CV-Bench failed with {fallback_error}; fallback failed with {type(fallback_exc).__name__}: {fallback_exc}",
                }
            )
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def find_video_sample(zips_dir: Path, metadata_path: Path) -> Tuple[Path, Dict[str, Any], zipfile.ZipInfo]:
    df = pd.read_parquet(metadata_path)
    zip_infos: List[Tuple[Path, zipfile.ZipInfo]] = []
    for zip_path in sorted(zips_dir.glob("videos_chunked_*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.filename.endswith(".mp4") and info.file_size > 0:
                    zip_infos.append((zip_path, info))
    by_id = {Path(info.filename).stem: (zip_path, info) for zip_path, info in zip_infos}
    hits = df[df["videoID"].isin(by_id.keys())].copy()
    if "duration" in hits.columns:
        hits = hits[hits["duration"].astype(str).str.lower() == "short"]
    if hits.empty:
        raise RuntimeError("No Video-MME metadata rows matched the downloaded zip chunks.")
    hits["local_size"] = hits["videoID"].map(lambda video_id: by_id[str(video_id)][1].file_size)
    row = hits.sort_values(["local_size", "question_id"]).iloc[0].to_dict()
    zip_path, info = by_id[str(row["videoID"])]
    return zip_path, row, info


def extract_zip_member(zip_path: Path, member: zipfile.ZipInfo, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / Path(member.filename).name
    if out_path.exists() and out_path.stat().st_size == member.file_size:
        return out_path
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as source, out_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return out_path


def run_videomme_local(
    runner: QwenVLRunner,
    zips_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    max_new_tokens: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "video-mme-modelscope", "modality": "video", "status": "started"}
    started = time.perf_counter()
    try:
        zip_path, row, member = find_video_sample(zips_dir, metadata_path)
        video_path = extract_zip_member(zip_path, member, output_dir / "media" / "video_mme")
        options = "\n".join(str(option) for option in row["options"])
        question = f"{row['question']}\n{options}\nRespond with only the option letter."
        response = runner.generate(
            [{"type": "video", "video": str(video_path), "fps": 1.0}, {"type": "text", "text": question}],
            max_new_tokens=max_new_tokens,
        )
        prediction = extract_letter(response)
        ground_truth = extract_letter(row.get("answer"))
        record.update(
            {
                "status": "completed",
                "video": str(video_path),
                "zip": str(zip_path),
                "zip_member": member.filename,
                "videoID": row.get("videoID"),
                "question_id": row.get("question_id"),
                "question": question,
                "ground_truth": ground_truth,
                "Answer": prediction,
                "raw_response": response,
                "correct": bool(prediction and ground_truth and prediction == ground_truth),
            }
        )
        add_compression_metrics(
            record,
            member.file_size,
            video_path.stat().st_size,
            "Video smoke passes the extracted local clip directly; token counts are byte/4 proxy estimates.",
        )
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def decode_audio(audio_cell: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    audio_bytes = audio_cell.get("bytes")
    if audio_bytes is None:
        raise RuntimeError("Audio cell does not contain bytes.")
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sample_rate)


def make_audio_spectrogram(audio: np.ndarray, sample_rate: int, output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio.size == 0:
        raise RuntimeError("Audio sample is empty.")
    audio = audio[: sample_rate * 12]
    audio = audio / max(float(np.max(np.abs(audio))), 1e-6)
    n_fft = 1024
    hop = 512
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    frames = []
    window = np.hanning(n_fft).astype(np.float32)
    for start in range(0, audio.size - n_fft + 1, hop):
        frame = audio[start : start + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    spec = np.stack(frames, axis=1)
    spec = np.log1p(spec)
    spec = spec / max(float(spec.max()), 1e-6)
    spec_img = (255 - (spec * 255)).astype(np.uint8)
    spec_img = np.flipud(spec_img)
    image = Image.fromarray(spec_img, mode="L").resize((960, 540))
    image = Image.merge("RGB", (image, image, image))
    draw = ImageDraw.Draw(image)
    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(np.square(audio))))
    freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)
    mean_power = spec.mean(axis=1)
    top_freqs = freqs[np.argsort(mean_power)[-5:]]
    caption = f"Spectrogram | {duration:.1f}s | {sample_rate}Hz | RMS {rms:.3f}"
    draw.rectangle((0, 0, 960, 30), fill=(255, 255, 255))
    draw.text((10, 8), caption, fill=(0, 0, 0))
    image.save(output_path)
    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "rms": round(rms, 5),
        "dominant_frequency_hz": [round(float(x), 1) for x in sorted(top_freqs)],
    }


def run_audio_spectrogram(
    runner: QwenVLRunner,
    audio_parquet: Path,
    output_dir: Path,
    max_new_tokens: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "urbansound8k-noises-modelscope", "modality": "audio_spectrogram", "status": "started"}
    started = time.perf_counter()
    try:
        df = pd.read_parquet(audio_parquet)
        row = df.iloc[0].to_dict()
        source_audio_bytes = len(row["audio"].get("bytes") or b"")
        audio, sample_rate = decode_audio(row["audio"])
        spectrogram_path = output_dir / "media" / "audio" / f"{Path(str(row.get('file', 'sample'))).stem}_spectrogram.png"
        stats = make_audio_spectrogram(audio, sample_rate, spectrogram_path)
        instruction = str(row.get("instruction") or "")
        prompt = (
            "Qwen2.5-VL-3B has no native audio input here, so use the spectrogram image and numeric audio summary "
            "as compressed audio evidence.\n"
            f"Audio summary: {json.dumps(stats, ensure_ascii=False)}\n"
            f"Task: {instruction}\n"
            "Return the most likely label only."
        )
        response = runner.generate(
            [{"type": "image", "image": str(spectrogram_path)}, {"type": "text", "text": prompt}],
            max_new_tokens=max_new_tokens,
        )
        label = str(row.get("label") or "")
        record.update(
            {
                "status": "completed",
                "image": str(spectrogram_path),
                "audio_file": str(row.get("file") or ""),
                "question": prompt,
                "ground_truth": label,
                "Answer": response.strip(),
                "raw_response": response,
                "audio_stats": stats,
                "correct": normalize_label(label) in normalize_label(response),
                "note": "Audio is evaluated through a generated spectrogram because Qwen2.5-VL-3B is not a native audio model.",
            }
        )
        add_compression_metrics(
            record,
            source_audio_bytes,
            spectrogram_path.stat().st_size + len(prompt.encode("utf-8")),
            "Audio source bytes are WAV bytes from parquet; evidence bytes are spectrogram PNG plus text summary prompt. Token counts are byte/4 proxy estimates.",
        )
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ModelScope-backed image/video/audio smoke tests with Qwen2.5-VL-3B.")
    parser.add_argument("--model-path", default="models/qwen2.5-vl-3b")
    parser.add_argument("--modelscope-root", default="data/modelscope")
    parser.add_argument("--videomme-metadata", default="data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
    parser.add_argument(
        "--audio-parquet",
        default="data/modelscope/urbansound8k-noises/data/test-00000-of-00001-40cf49999a374336.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/modelscope_multimodal_smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modelscope_root = resolve_path(args.modelscope_root)

    started = time.perf_counter()
    runner = QwenVLRunner(resolve_path(args.model_path), args.device, args.dtype)
    records = [
        run_image_cv_bench(runner, output_dir, args.max_new_tokens),
        run_videomme_local(
            runner,
            modelscope_root / "video-mme-zips",
            resolve_path(args.videomme_metadata),
            output_dir,
            args.max_new_tokens,
        ),
        run_audio_spectrogram(runner, resolve_path(args.audio_parquet), output_dir, args.max_new_tokens),
    ]
    summary = {
        "model": str(resolve_path(args.model_path)),
        "modelscope_root": str(modelscope_root),
        "records": records,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "counts": {
            "completed": sum(1 for item in records if item.get("status") == "completed"),
            "failed": sum(1 for item in records if item.get("status") == "failed"),
            "correct": sum(1 for item in records if item.get("correct") is True),
        },
    }
    records_path = output_dir / "modelscope_multimodal_smoke_results.json"
    summary_path = output_dir / "modelscope_multimodal_smoke_summary.json"
    records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {records_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    for item in records:
        print(
            f"{item.get('dataset')} [{item.get('modality')}]: {item.get('status')} "
            f"pred={item.get('Answer')} gt={item.get('ground_truth')} correct={item.get('correct')} "
            f"error={item.get('Error', '')}",
            flush=True,
        )
    os._exit(0)


if __name__ == "__main__":
    main()
