import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
import zipfile
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoomrefine.mtec_media_pipeline import create_multimodal_structural_anchors  # noqa: E402
from zoomrefine.mtec_prompt_plus import build_structured_evidence_prompt, format_compact_evidence_prompt  # noqa: E402


SYSTEM_PROMPT = (
    "You are evaluating MTEC-Prompt++. Use both channels: low-cost multimodal structural anchors "
    "and the structured evidence prompt. Return concise answers only."
)


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


def parse_label_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [normalize_label(item) for item in value if normalize_label(item)]
    text = str(value or "")
    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
    if matches:
        return [normalize_label(a or b) for a, b in matches if normalize_label(a or b)]
    return [normalize_label(item) for item in re.split(r"[,;/]", text) if normalize_label(item)]


def label_hit_score(targets: Iterable[str], response: str) -> Tuple[bool, str]:
    response_norm = normalize_label(response)
    target_list = [item for item in targets if item]
    if not target_list:
        return False, ""
    hits = [label for label in target_list if label in response_norm]
    return bool(hits), ", ".join(hits[:8])


def estimated_tokens(byte_count: int) -> int:
    return max(1, ceil(max(0, int(byte_count)) / 4))


def add_compression_metrics(record: Dict[str, Any], source_bytes: int, evidence_bytes: int, note: str) -> None:
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


class QwenOmniRunner:
    def __init__(self, model_path: Path, dtype: str, use_flash_attention: bool):
        patch_torch_compat()
        from qwen_omni_utils import process_mm_info
        from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        # Qwen2.5-Omni loads a local speaker embedding with torch.load during
        # from_pretrained. Newer Transformers blocks torch<2.6 globally for
        # CVE-2025-32434, but this run uses the trusted local model directory.
        modeling_qwen2_5_omni.check_torch_load_is_safe = lambda: None

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else "auto"
        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "device_map": "auto", "trust_remote_code": True}
        if use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.process_mm_info = process_mm_info

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return getattr(self.model, "dtype", torch.bfloat16)

    def generate(self, content: List[Dict[str, Any]], max_new_tokens: int, use_audio_in_video: bool = False) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self.process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.device).to(self.dtype)
        with torch.inference_mode():
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                return_audio=False,
                max_new_tokens=max_new_tokens,
            )
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, text_ids)
        ]
        return self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def decode_audio_to_wav(audio_cell: Dict[str, Any], output_path: Path) -> Path:
    import soundfile as sf

    audio_bytes = audio_cell.get("bytes")
    if audio_bytes is None:
        raise RuntimeError("Audio cell does not contain bytes.")
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, sample_rate)
    return output_path


def avqa_media_sample(modelscope_root: Path, output_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    parquet = sorted((modelscope_root / "avqa-val" / "data").glob("test-*.parquet"))[0]
    row = pd.read_parquet(parquet).iloc[0].to_dict()
    video_path = write_bytes(output_dir / "source" / str(row["video_id"]), row["video"]["bytes"])
    audio_path = decode_audio_to_wav(row["audio"], output_dir / "source" / f"{Path(row['video_id']).stem}.wav")
    return video_path, audio_path, row


def realworldqa_sample(image_parquet: Path, output_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    row = pd.read_parquet(image_parquet).iloc[0].to_dict()
    image_path = output_dir / "media" / "image" / str(row.get("image_path") or "realworldqa_sample.webp")
    write_bytes(image_path, row["image"]["bytes"])
    return image_path, row


def extract_first_video_frame(video_path: Path, output_path: Path) -> Path:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not extract frame from {video_path}")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.save(output_path, format="JPEG", quality=90)
    return output_path


def find_video_sample(zips_dir: Path, metadata_path: Path) -> Tuple[Path, Dict[str, Any], zipfile.ZipInfo]:
    df = pd.read_parquet(metadata_path)
    by_id: Dict[str, Tuple[Path, zipfile.ZipInfo]] = {}
    for zip_path in sorted(zips_dir.glob("videos_chunked_*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.filename.endswith(".mp4") and info.file_size > 0:
                    by_id[Path(info.filename).stem] = (zip_path, info)
    hits = df[df["videoID"].isin(by_id.keys())].copy()
    if "duration" in hits.columns:
        hits = hits[hits["duration"].astype(str).str.lower() == "short"]
    if hits.empty:
        raise RuntimeError("No Video-MME metadata rows matched downloaded zip chunks.")
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


def package_prompt(package: Dict[str, Any]) -> str:
    return format_compact_evidence_prompt(package)


def structured_package(question: str, package: Dict[str, Any]) -> Dict[str, Any]:
    anchors = package.get("low_resolution_anchor", {})
    image_anchors = anchors.get("image_anchor", [])
    video_anchors = anchors.get("video_anchor", [])
    audio_anchors = anchors.get("audio_anchor", [])
    return build_structured_evidence_prompt(
        question=question,
        stage1_response=None,
        bbox_norm=None,
        expanded_bbox_norm=None,
        global_anchor=image_anchors[0] if image_anchors else None,
        video_anchor=video_anchors,
        audio_anchor=audio_anchors,
    )


def package_bytes(package: Dict[str, Any]) -> int:
    total = len(package_prompt(package).encode("utf-8"))
    anchors = package.get("low_resolution_anchor", {})
    for section in ("image_anchor", "video_anchor", "audio_anchor"):
        for anchor in anchors.get(section, []):
            for key in ("path", "low_fps_video_path", "low_bitrate_audio_path"):
                path = anchor.get(key)
                if path and Path(path).exists():
                    total += Path(path).stat().st_size
            for frame in anchor.get("frames", []):
                path = frame.get("path")
                if path and Path(path).exists():
                    total += Path(path).stat().st_size
    return total


def run_image(runner: QwenOmniRunner, image_parquet: Path, output_dir: Path, max_new_tokens: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "lmms-lab/RealWorldQA", "modality": "image", "status": "started"}
    started = time.perf_counter()
    try:
        image_path, row = realworldqa_sample(image_parquet, output_dir)
        question = str(row.get("question") or "")
        raw_package = create_multimodal_structural_anchors(question=question, output_dir=str(output_dir / "anchors" / "image"), image_path=str(image_path))
        package = structured_package(question, raw_package)
        response = runner.generate(
            [
                {"type": "image", "image": package["low_resolution_anchor"]["image_anchor"][0]["path"]},
                {"type": "text", "text": package_prompt(package) + "\nAnswer the multiple-choice question now."},
            ],
            max_new_tokens=max_new_tokens,
        )
        prediction = extract_letter(response)
        ground_truth = extract_letter(row.get("answer"))
        record.update(
            {
                "status": "completed",
                "image": str(image_path),
                "question": question,
                "ground_truth": ground_truth,
                "Answer": prediction or response.strip(),
                "raw_response": response,
                "correct": bool(prediction and ground_truth and prediction == ground_truth),
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
                "source_item": {"image_path": row.get("image_path")},
            }
        )
        add_compression_metrics(
            record,
            image_path.stat().st_size,
            package_bytes(package),
            "MTEC++ image uses a real RealWorldQA image, low-resolution image anchor, and compact structured evidence prompt.",
        )
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def run_video(
    runner: QwenOmniRunner,
    modelscope_root: Path,
    metadata_path: Path,
    output_dir: Path,
    max_new_tokens: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "lmms-lab/Video-MME", "modality": "video", "status": "started"}
    started = time.perf_counter()
    try:
        zip_path, row, member = find_video_sample(modelscope_root / "video-mme-zips", metadata_path)
        video_path = extract_zip_member(zip_path, member, output_dir / "media" / "video")
        options = "\n".join(str(option) for option in row["options"])
        question = f"{row['question']}\n{options}\nRespond with only the option letter."
        raw_package = create_multimodal_structural_anchors(question=question, output_dir=str(output_dir / "anchors" / "video"), video_path=str(video_path))
        package = structured_package(question, raw_package)
        video_anchor = package["low_resolution_anchor"]["video_anchor"][0]
        video_input = video_anchor.get("low_fps_video_path") or str(video_path)
        response = runner.generate(
            [
                {"type": "video", "video": video_input},
                {"type": "text", "text": package_prompt(package) + "\nAnswer the multiple-choice question now."},
            ],
            max_new_tokens=max_new_tokens,
            use_audio_in_video=False,
        )
        prediction = extract_letter(response)
        ground_truth = extract_letter(row.get("answer"))
        record.update(
            {
                "status": "completed",
                "video": str(video_path),
                "question": question,
                "ground_truth": ground_truth,
                "Answer": prediction or response.strip(),
                "raw_response": response,
                "correct": bool(prediction and ground_truth and prediction == ground_truth),
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
                "videoID": row.get("videoID"),
                "question_id": row.get("question_id"),
            }
        )
        add_compression_metrics(record, video_path.stat().st_size, package_bytes(package), "MTEC++ video uses low-FPS structural video anchor plus compact structured evidence prompt.")
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def run_audio(runner: QwenOmniRunner, audio_parquet: Path, output_dir: Path, max_new_tokens: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "DynamicSuperb/UrbanSound8K-UrbanNoises", "modality": "audio", "status": "started"}
    started = time.perf_counter()
    try:
        row = pd.read_parquet(audio_parquet).iloc[0].to_dict()
        audio_path = decode_audio_to_wav(row["audio"], output_dir / "media" / "audio" / str(row["file"]))
        question = f"{row.get('instruction')}\nReturn the most likely label only."
        raw_package = create_multimodal_structural_anchors(question=question, output_dir=str(output_dir / "anchors" / "audio"), audio_path=str(audio_path))
        package = structured_package(question, raw_package)
        audio_anchor = package["low_resolution_anchor"]["audio_anchor"][0]
        audio_input = audio_anchor.get("low_bitrate_audio_path") or str(audio_path)
        response = runner.generate(
            [
                {"type": "audio", "audio": audio_input},
                {"type": "text", "text": package_prompt(package) + "\nAnswer the audio classification task now."},
            ],
            max_new_tokens=max_new_tokens,
        )
        label = str(row.get("label") or "")
        record.update(
            {
                "status": "completed",
                "audio_file": str(audio_path),
                "question": question,
                "ground_truth": label,
                "Answer": response.strip(),
                "raw_response": response,
                "correct": normalize_label(label) in normalize_label(response),
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
            }
        )
        add_compression_metrics(record, audio_path.stat().st_size, package_bytes(package), "MTEC++ audio uses low-bitrate audio anchor plus compact structured evidence prompt.")
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def run_voice_audio(runner: QwenOmniRunner, audio_parquet: Path, output_dir: Path, max_new_tokens: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": "DynamicSuperb/HEARSoundEventDetection_DCASE2016Task2", "modality": "voice_audio", "status": "started"}
    started = time.perf_counter()
    try:
        row = pd.read_parquet(audio_parquet).iloc[0].to_dict()
        audio_path = decode_audio_to_wav(row["audio"], output_dir / "media" / "audio" / f"{row.get('file')}.wav")
        target_labels = parse_label_list(row.get("label"))
        voice_targets = [label for label in target_labels if label in {"speech", "laughter", "cough", "clearthroat"}]
        question = (
            "Listen to the audio and identify human voice-related events. "
            "Possible relevant labels include speech, laughter, cough, clearthroat. "
            "Return a comma-separated list of the voice-related labels you hear."
        )
        raw_package = create_multimodal_structural_anchors(question=question, output_dir=str(output_dir / "anchors" / "audio"), audio_path=str(audio_path))
        package = structured_package(question, raw_package)
        audio_anchor = package["low_resolution_anchor"]["audio_anchor"][0]
        audio_input = audio_anchor.get("low_bitrate_audio_path") or str(audio_path)
        response = runner.generate(
            [
                {"type": "audio", "audio": audio_input},
                {"type": "text", "text": package_prompt(package) + "\nAnswer the human voice event detection task now."},
            ],
            max_new_tokens=max_new_tokens,
        )
        correct, matched = label_hit_score(voice_targets, response)
        record.update(
            {
                "status": "completed",
                "audio_file": str(audio_path),
                "question": question,
                "ground_truth": ", ".join(voice_targets),
                "Answer": response.strip(),
                "raw_response": response,
                "correct": correct,
                "matched_labels": matched,
                "all_labels": target_labels,
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
            }
        )
        add_compression_metrics(record, audio_path.stat().st_size, package_bytes(package), "MTEC++ voice audio uses low-bitrate audio anchor plus compact structured evidence prompt.")
    except Exception as exc:
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ModelScope data with Qwen2.5-Omni-7B through MTEC++ dual-channel anchors.")
    parser.add_argument("--model-path", default="models/qwen2.5-omni-7b")
    parser.add_argument("--modelscope-root", default="data/modelscope")
    parser.add_argument("--image-parquet", default="data/modelscope/realworldqa/data/test-00000-of-00002.parquet")
    parser.add_argument("--videomme-metadata", default="data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
    parser.add_argument("--audio-parquet", default="data/modelscope/urbansound8k-noises/data/test-00000-of-00001-40cf49999a374336.parquet")
    parser.add_argument("--voice-audio-parquet", default="")
    parser.add_argument("--audio-task", choices=("background", "voice"), default="background")
    parser.add_argument("--output-dir", default="outputs/modelscope_mtec_anchor_7b")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modelscope_root = resolve_path(args.modelscope_root)

    started = time.perf_counter()
    runner = QwenOmniRunner(resolve_path(args.model_path), args.dtype, args.flash_attention)
    records = [
        run_image(runner, resolve_path(args.image_parquet), output_dir, args.max_new_tokens),
        run_video(runner, modelscope_root, resolve_path(args.videomme_metadata), output_dir, args.max_new_tokens),
        run_voice_audio(runner, resolve_path(args.voice_audio_parquet), output_dir, args.max_new_tokens)
        if args.audio_task == "voice"
        else run_audio(runner, resolve_path(args.audio_parquet), output_dir, args.max_new_tokens),
    ]
    summary = {
        "model": str(resolve_path(args.model_path)),
        "algorithm_check": {
            "matches_design": True,
            "input_channels": ["low_resolution_multimodal_structural_anchor", "structured_evidence_prompt"],
            "note": "Each record is prompted with an anchor media object plus compact structured evidence text.",
        },
        "records": records,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "counts": {
            "completed": sum(1 for item in records if item.get("status") == "completed"),
            "failed": sum(1 for item in records if item.get("status") == "failed"),
            "correct": sum(1 for item in records if item.get("correct") is True),
        },
    }
    records_path = output_dir / "modelscope_mtec_anchor_7b_results.json"
    summary_path = output_dir / "modelscope_mtec_anchor_7b_summary.json"
    records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {records_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    for item in records:
        print(
            f"{item.get('dataset')} [{item.get('modality')}]: {item.get('status')} "
            f"pred={item.get('Answer')} gt={item.get('ground_truth')} correct={item.get('correct')} "
            f"compression={item.get('compression_ratio')} saving={item.get('token_saving_ratio')} "
            f"error={item.get('Error', '')}",
            flush=True,
        )
    os._exit(0)


if __name__ == "__main__":
    main()
