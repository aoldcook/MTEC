import json
import io
import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from zoomrefine.mtec_prompt_plus import (
        DEFAULT_TOTAL_BUDGET,
        build_low_resolution_anchor_package,
        create_image_global_anchor,
        infer_question_profile,
    )
except ImportError:
    from mtec_prompt_plus import (
        DEFAULT_TOTAL_BUDGET,
        build_low_resolution_anchor_package,
        create_image_global_anchor,
        infer_question_profile,
    )


DEFAULT_VIDEO_TARGET_FPS = 3.0
DEFAULT_VIDEO_MAX_FRAMES = 48
DEFAULT_VIDEO_MAX_SIDE = 512
DEFAULT_VIDEO_JPEG_QUALITY = 82
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_BITRATE = "32k"
DEFAULT_AUDIO_WINDOW_SECONDS = 1.0
DEFAULT_IMAGE_ANCHOR_TARGET_RATIO = 0.19
DEFAULT_IMAGE_DETAIL_MAX_CROPS = 3
DEFAULT_IMAGE_DETAIL_MAX_SIDE = 512
DEFAULT_VIDEO_DETAIL_MAX_CROPS = 6
DEFAULT_VIDEO_DETAIL_MAX_SIDE = 960
DEFAULT_VIDEO_TRANSCRIPT_MAX_SEGMENTS = 48
DEFAULT_VIDEO_ASR_MODEL = "base.en"
_FASTER_WHISPER_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def probe_media(path: str) -> Dict[str, Any]:
    media_path = Path(path)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"path": str(media_path), "probe_backend": None, "streams": []}

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {
            "path": str(media_path),
            "probe_backend": "ffprobe",
            "error": result.stderr.strip(),
            "streams": [],
        }

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}
    payload["path"] = str(media_path)
    payload["probe_backend"] = "ffprobe"
    return payload


def create_video_structural_anchor(
    video_path: str,
    output_dir: str,
    target_fps: float = DEFAULT_VIDEO_TARGET_FPS,
    max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
    max_side: int = DEFAULT_VIDEO_MAX_SIDE,
    jpeg_quality: int = DEFAULT_VIDEO_JPEG_QUALITY,
    anchor_id: str = "video_anchor_low_fps",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    frame_dir = output_path / anchor_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / source_fps if source_fps > 0 and frame_count > 0 else 0.0
    step = _frame_step(source_fps, target_fps)

    candidates: List[Dict[str, Any]] = []
    previous_feature: Optional[np.ndarray] = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            time_sec = frame_index / source_fps if source_fps > 0 else float(len(candidates))
            pil_image = _resize_frame(frame, max_side)
            feature = _frame_feature(pil_image)
            change_score = (
                _feature_distance(previous_feature, feature)
                if previous_feature is not None
                else 0.0
            )
            candidates.append(
                {
                    "frame_index": frame_index,
                    "time_sec": round(time_sec, 3),
                    "image": pil_image,
                    "feature": feature,
                    "change_score": round(change_score, 6),
                }
            )
            previous_feature = feature
        frame_index += 1
    capture.release()

    selected = _select_video_candidates(candidates, max_frames)
    frames = _write_video_frames(selected, frame_dir, jpeg_quality, anchor_id)
    clip_path = _write_low_fps_clip(selected, output_path, anchor_id, target_fps)
    event_boundaries = _event_boundaries(frames, max_count=max(1, max_frames // 4))

    selected_bytes = sum(frame.get("bytes", 0) for frame in frames)
    if clip_path and clip_path.exists():
        selected_bytes += clip_path.stat().st_size

    return {
        "anchor_id": anchor_id,
        "anchor_link": anchor_id,
        "type": "video_low_fps_frames",
        "role": "Preserve temporal order, action continuity, scene changes, and event-boundary visual references.",
        "source_path": str(Path(video_path)),
        "source_duration_sec": round(duration, 3),
        "source_fps": round(source_fps, 3),
        "source_frame_count": frame_count,
        "source_resolution": {"width": width, "height": height},
        "target_fps": target_fps,
        "max_side": max_side,
        "frames": frames,
        "event_boundaries": event_boundaries,
        "low_fps_video_path": str(clip_path) if clip_path else None,
        "compression": {
            "selected_frame_count": len(frames),
            "candidate_frame_count": len(candidates),
            "frame_retention_ratio": _safe_ratio(len(frames), frame_count),
            "bytes": selected_bytes,
            "strategy": "Low-FPS coverage plus change-aware and duplication-aware frame selection.",
        },
    }


def create_audio_structural_anchor(
    audio_path: str,
    output_dir: str,
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    bitrate: str = DEFAULT_AUDIO_BITRATE,
    window_seconds: float = DEFAULT_AUDIO_WINDOW_SECONDS,
    max_segments: int = 8,
    anchor_id: str = "audio_anchor_low_bitrate",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    low_audio_path = output_path / f"{anchor_id}.mp3"

    compression_warning = None
    if shutil.which("ffmpeg"):
        compression_warning = _compress_audio_ffmpeg(
            audio_path,
            low_audio_path,
            sample_rate=sample_rate,
            bitrate=bitrate,
        )
    else:
        low_audio_path, compression_warning = _compress_audio_python(
            audio_path,
            output_path,
            sample_rate=sample_rate,
            anchor_id=anchor_id,
        )

    samples, decoded_sample_rate, decode_warning = _load_audio_samples(audio_path, sample_rate)
    if samples.size:
        duration = round(float(samples.shape[0]) / decoded_sample_rate, 3)
        energy_profile = _audio_energy_profile(samples, decoded_sample_rate, window_seconds)
        event_segments = _select_audio_events(
            energy_profile,
            window_seconds=window_seconds,
            max_segments=max_segments,
            anchor_id=anchor_id,
        )
    else:
        duration = _probe_duration(audio_path)
        energy_profile = {
            "window_seconds": window_seconds,
            "rms_mean": None,
            "rms_std": None,
            "rms_max": None,
            "windows": [],
        }
        event_segments = []

    source_bytes = Path(audio_path).stat().st_size if Path(audio_path).exists() else 0
    anchor_bytes = low_audio_path.stat().st_size if low_audio_path.exists() else 0
    warnings = [item for item in (compression_warning, decode_warning) if item]

    return {
        "anchor_id": anchor_id,
        "anchor_link": anchor_id,
        "type": "audio_low_bitrate_events",
        "role": "Preserve low-cost acoustic rhythm and salient event segments.",
        "source_path": str(Path(audio_path)),
        "low_bitrate_audio_path": str(low_audio_path),
        "source_duration_sec": duration,
        "target_sample_rate": sample_rate,
        "target_bitrate": bitrate,
        "audio_event_segments": event_segments,
        "energy_summary": {
            key: value
            for key, value in energy_profile.items()
            if key != "windows"
        },
        "compression": {
            "source_bytes": source_bytes,
            "anchor_bytes": anchor_bytes,
            "byte_ratio": _safe_ratio(anchor_bytes, source_bytes),
            "strategy": "Low-bitrate mono audio plus windowed energy event summary.",
        },
        "warnings": warnings,
    }


def create_multimodal_structural_anchors(
    question: str,
    output_dir: str,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    video_target_fps: float = DEFAULT_VIDEO_TARGET_FPS,
    video_max_frames: int = DEFAULT_VIDEO_MAX_FRAMES,
    video_max_side: int = DEFAULT_VIDEO_MAX_SIDE,
    include_video_audio: bool = True,
    include_video_transcript: bool = True,
    video_transcript_backend: str = "auto",
    video_asr_model: str = DEFAULT_VIDEO_ASR_MODEL,
    video_asr_language: Optional[str] = "en",
    video_transcript_max_segments: int = DEFAULT_VIDEO_TRANSCRIPT_MAX_SEGMENTS,
    video_detail_max_crops: int = DEFAULT_VIDEO_DETAIL_MAX_CROPS,
    video_detail_max_side: int = DEFAULT_VIDEO_DETAIL_MAX_SIDE,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_anchor = None
    if image_path:
        image_anchor = create_image_structural_anchors(
            question=question,
            image_path=image_path,
            output_dir=output_path,
        )

    video_anchor = None
    if video_path:
        video_anchor = create_video_structural_anchor(
            video_path,
            str(output_path),
            target_fps=video_target_fps,
            max_frames=video_max_frames,
            max_side=video_max_side,
        )
        video_detail_anchors = create_video_detail_frame_anchors(
            question=question,
            video_path=video_path,
            output_dir=output_path,
            video_anchor=video_anchor,
            max_crops=video_detail_max_crops,
            max_side=video_detail_max_side,
        )
        if video_detail_anchors:
            image_anchor = _merge_anchor_lists(image_anchor, video_detail_anchors)

    audio_anchor = None
    if audio_path:
        audio_anchor = create_audio_structural_anchor(audio_path, str(output_path))
    elif video_path and include_video_audio:
        audio_anchor = create_audio_structural_anchor(
            video_path,
            str(output_path),
            anchor_id="video_audio_anchor_low_bitrate",
        )

    transcript_anchor = None
    if video_path and include_video_transcript:
        transcript_anchor = create_video_transcript_anchor(
            video_path=video_path,
            output_dir=str(output_path),
            backend=video_transcript_backend,
            model_name=video_asr_model,
            language=video_asr_language,
            max_segments=video_transcript_max_segments,
        )

    package = build_low_resolution_anchor_package(
        question=question,
        global_anchor=image_anchor,
        video_anchor=video_anchor,
        audio_anchor=audio_anchor,
        total_budget=total_budget,
    )
    if transcript_anchor:
        package["low_resolution_anchor"]["transcript_anchor"] = [transcript_anchor]
    package["media_probe"] = {
        "image": {"path": image_path} if image_path else None,
        "video": probe_media(video_path) if video_path else None,
        "audio": probe_media(audio_path) if audio_path else None,
    }
    return package



def create_video_detail_frame_anchors(
    question: str,
    video_path: str,
    output_dir: Path,
    video_anchor: Dict[str, Any],
    max_crops: int = DEFAULT_VIDEO_DETAIL_MAX_CROPS,
    max_side: int = DEFAULT_VIDEO_DETAIL_MAX_SIDE,
    jpeg_quality: int = 86,
    anchor_id: str = "video_keyframe_detail",
) -> List[Dict[str, Any]]:
    if max_crops <= 0 or not _needs_video_detail(question):
        return []

    frames = _select_video_detail_frames(video_anchor, max_count=max(1, math.ceil(max_crops / 2)))
    if not frames:
        return []

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    detail_dir = output_dir / anchor_id
    detail_dir.mkdir(parents=True, exist_ok=True)
    anchors: List[Dict[str, Any]] = []
    source_width = int(video_anchor.get("source_resolution", {}).get("width") or 0)
    source_height = int(video_anchor.get("source_resolution", {}).get("height") or 0)
    crop_specs = _video_detail_crop_specs(source_width, source_height, question)

    try:
        for frame in frames:
            if len(anchors) >= max_crops:
                break
            frame_index = int(frame.get("frame_index") or 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            width, height = image.size
            for spec in crop_specs:
                if len(anchors) >= max_crops:
                    break
                box = _norm_box(width, height, *spec["norm"])
                crop = image.crop(box)
                crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                output_index = len(anchors) + 1
                detail_anchor_id = f"{anchor_id}_{output_index:04d}"
                detail_path = detail_dir / f"{detail_anchor_id}.jpg"
                crop.save(detail_path, format="JPEG", quality=jpeg_quality, optimize=True)
                bbox_norm = _box_to_norm(box, width, height)
                anchors.append(
                    {
                        "anchor_id": detail_anchor_id,
                        "anchor_link": detail_anchor_id,
                        "type": "video_keyframe_detail_crop",
                        "role": "Preserve high-resolution detail from an answer-relevant video keyframe for OCR, screen content, numbers, small objects, and state changes.",
                        "path": str(detail_path),
                        "source_video_path": str(Path(video_path)),
                        "time_sec": frame.get("time_sec"),
                        "frame_index": frame_index,
                        "source_resolution": {"width": width, "height": height},
                        "resolution": {"width": crop.width, "height": crop.height},
                        "bbox_norm": bbox_norm,
                        "expanded_bbox_norm": bbox_norm,
                        "region_hint": spec["label"],
                        "linked_video_anchor": frame.get("anchor_link"),
                        "change_score": frame.get("change_score"),
                        "compression": {
                            "format": "jpeg",
                            "bytes": detail_path.stat().st_size,
                            "max_side": max_side,
                            "quality": jpeg_quality,
                            "strategy": "Question-conditioned high-detail crop from a selected video keyframe.",
                        },
                    }
                )
    finally:
        capture.release()
    return anchors


def _merge_anchor_lists(existing: Optional[Any], extra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    if isinstance(existing, list):
        merged.extend(item for item in existing if item)
    elif existing:
        merged.append(existing)
    merged.extend(extra)
    return merged


def _needs_video_detail(question: str) -> bool:
    text = (question or "").lower()
    detail_keywords = (
        "text", "word", "number", "digit", "price", "$", "label", "sign",
        "screen", "phone", "smart phone", "caption", "ingredient", "color",
        "logo", "painting", "artwork", "not appear", "not used", "which of",
        "read", "shown", "display", "visible", "small", "sequence", "before", "after",
    )
    return any(keyword in text for keyword in detail_keywords)


def _select_video_detail_frames(video_anchor: Dict[str, Any], max_count: int) -> List[Dict[str, Any]]:
    frames = [frame for frame in video_anchor.get("frames", []) if frame.get("frame_index") is not None]
    if not frames:
        return []
    selected: List[Dict[str, Any]] = []
    for boundary in video_anchor.get("event_boundaries", []):
        link = boundary.get("anchor_link")
        match = next((frame for frame in frames if frame.get("anchor_link") == link), None)
        if match and match not in selected:
            selected.append(match)
        if len(selected) >= max_count:
            return selected
    ranked = sorted(frames, key=lambda frame: float(frame.get("change_score") or 0.0), reverse=True)
    for frame in ranked:
        if frame not in selected:
            selected.append(frame)
        if len(selected) >= max_count:
            return sorted(selected, key=lambda item: float(item.get("time_sec") or 0.0))
    return sorted(selected, key=lambda item: float(item.get("time_sec") or 0.0))


def _video_detail_crop_specs(width: int, height: int, question: str) -> List[Dict[str, Any]]:
    text = (question or "").lower()
    specs: List[Dict[str, Any]] = []
    if any(word in text for word in ("screen", "phone", "caption", "subtitle", "text", "read", "price", "$", "number", "digit")):
        specs.extend(
            [
                {"label": "keyframe_full_detail", "norm": (0.0, 0.0, 1.0, 1.0)},
                {"label": "keyframe_center_screen_detail", "norm": (0.12, 0.10, 0.88, 0.90)},
                {"label": "keyframe_lower_text_band", "norm": (0.05, 0.55, 0.95, 1.0)},
                {"label": "keyframe_upper_text_band", "norm": (0.05, 0.0, 0.95, 0.45)},
            ]
        )
    elif any(word in text for word in ("left", "right", "where", "position", "side")):
        specs.extend(
            [
                {"label": "keyframe_full_detail", "norm": (0.0, 0.0, 1.0, 1.0)},
                {"label": "keyframe_left_detail", "norm": (0.0, 0.05, 0.58, 0.95)},
                {"label": "keyframe_right_detail", "norm": (0.42, 0.05, 1.0, 0.95)},
            ]
        )
    else:
        specs.extend(
            [
                {"label": "keyframe_full_detail", "norm": (0.0, 0.0, 1.0, 1.0)},
                {"label": "keyframe_center_detail", "norm": (0.15, 0.12, 0.85, 0.88)},
                {"label": "keyframe_lower_detail", "norm": (0.05, 0.48, 0.95, 1.0)},
            ]
        )
    return _dedupe_video_specs(specs)


def _dedupe_video_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for spec in specs:
        key = spec["norm"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique

def create_video_transcript_anchor(
    video_path: str,
    output_dir: str,
    backend: str = "auto",
    model_name: str = DEFAULT_VIDEO_ASR_MODEL,
    language: Optional[str] = "en",
    max_segments: int = DEFAULT_VIDEO_TRANSCRIPT_MAX_SEGMENTS,
    anchor_id: str = "video_transcript_anchor",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    segments: List[Dict[str, Any]] = []
    source = "none"

    if backend in {"auto", "subtitle"}:
        subtitle_segments, subtitle_warning = _extract_embedded_subtitles(video_path, output_path, anchor_id)
        if subtitle_warning:
            warnings.append(subtitle_warning)
        if subtitle_segments:
            segments = subtitle_segments
            source = "embedded_subtitle"

    if not segments and backend in {"auto", "faster-whisper"}:
        wav_path, audio_warning = _extract_audio_wav_for_asr(video_path, output_path / f"{anchor_id}.wav")
        if audio_warning:
            warnings.append(audio_warning)
        if wav_path:
            asr_segments, asr_warning = _run_faster_whisper_asr(
                wav_path,
                model_name=model_name,
                language=language,
                max_segments=max_segments,
                anchor_id=anchor_id,
            )
            if asr_warning:
                warnings.append(asr_warning)
            if asr_segments:
                segments = asr_segments
                source = "faster_whisper_asr"

    if backend == "none":
        warnings.append("Transcript extraction disabled.")
    elif not segments:
        warnings.append("No transcript segments extracted; install faster-whisper or provide embedded subtitles.")

    text = " ".join(segment.get("text", "") for segment in segments).strip()
    transcript_path = output_path / f"{anchor_id}.json"
    transcript_payload = {
        "anchor_id": anchor_id,
        "anchor_link": anchor_id,
        "type": "video_timestamped_transcript",
        "role": "Preserve speech, subtitles, narration, named entities, event order, and negated/not-mentioned details for long-video reasoning.",
        "source_path": str(Path(video_path)),
        "source": source,
        "backend": backend,
        "model_name": model_name if source == "faster_whisper_asr" else None,
        "language": language,
        "segments": segments[:max_segments],
        "text_preview": text[:1200],
        "warnings": warnings,
    }
    transcript_path.write_text(json.dumps(transcript_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    transcript_payload["path"] = str(transcript_path)
    transcript_payload["compression"] = {
        "segments": len(transcript_payload["segments"]),
        "text_chars": len(text),
        "strategy": "Timestamped subtitle/ASR text is stored as structured evidence rather than raw audio media.",
    }
    return transcript_payload


def _extract_embedded_subtitles(
    video_path: str,
    output_dir: Path,
    anchor_id: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [], "ffmpeg unavailable; skipped embedded subtitle extraction."
    subtitle_path = output_dir / f"{anchor_id}.srt"
    cmd = [ffmpeg, "-y", "-i", str(video_path), "-map", "0:s:0", str(subtitle_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not subtitle_path.exists() or subtitle_path.stat().st_size == 0:
        return [], "No embedded subtitle stream extracted."
    return _parse_srt(subtitle_path.read_text(encoding="utf-8", errors="ignore"), anchor_id), None


def _parse_srt(text: str, anchor_id: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    blocks = [block.strip() for block in re_split_blank_lines(text) if block.strip()]
    for index, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line = next((line for line in lines if "-->" in line), "")
        if not time_line:
            continue
        start_text, end_text = [part.strip() for part in time_line.split("-->", 1)]
        caption = " ".join(line for line in lines if "-->" not in line and not line.isdigit()).strip()
        if not caption:
            continue
        segments.append(
            {
                "anchor_id": f"{anchor_id}_seg_{index:04d}",
                "anchor_link": f"{anchor_id}_seg_{index:04d}",
                "time_range_sec": [_srt_time_to_seconds(start_text), _srt_time_to_seconds(end_text)],
                "text": caption,
                "source": "embedded_subtitle",
            }
        )
    return segments


def re_split_blank_lines(text: str) -> List[str]:
    import re

    return re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))


def _srt_time_to_seconds(value: str) -> float:
    value = value.replace(",", ".").split()[0]
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 3)
        if len(parts) == 2:
            minutes, seconds = parts
            return round(int(minutes) * 60 + float(seconds), 3)
        return round(float(value), 3)
    except ValueError:
        return 0.0


def _extract_audio_wav_for_asr(video_path: str, output_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path, None
    try:
        import av
        import soundfile as sf
    except Exception as err:
        return None, f"PyAV/soundfile unavailable; cannot extract audio for ASR: {err}"

    try:
        container = av.open(str(video_path))
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            return None, "No audio stream found in video."
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            if resampled_frames is None:
                continue
            if not isinstance(resampled_frames, list):
                resampled_frames = [resampled_frames]
            for resampled in resampled_frames:
                array = resampled.to_ndarray()
                if array.ndim > 1:
                    array = array.reshape(-1)
                chunks.append(array.astype(np.int16))
        container.close()
        if not chunks:
            return None, "Audio stream decoded to no samples."
        samples = np.concatenate(chunks).astype(np.float32) / 32768.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), samples, 16000)
        return output_path, None
    except Exception as err:
        return None, f"Failed to extract audio for ASR: {type(err).__name__}: {err}"


def _run_faster_whisper_asr(
    wav_path: Path,
    model_name: str,
    language: Optional[str],
    max_segments: int,
    anchor_id: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        from faster_whisper import WhisperModel
    except Exception as err:
        return [], f"faster-whisper unavailable; ASR skipped: {err}"

    try:
        device = "cpu"
        compute_type = "int8"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception:
            pass
        cache_key = (model_name, device, compute_type)
        model = _FASTER_WHISPER_MODEL_CACHE.get(cache_key)
        if model is None:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _FASTER_WHISPER_MODEL_CACHE[cache_key] = model
        raw_segments, info = model.transcribe(
            str(wav_path),
            language=language,
            beam_size=1,
            vad_filter=True,
            word_timestamps=False,
        )
        segments: List[Dict[str, Any]] = []
        for index, segment in enumerate(raw_segments, start=1):
            text = " ".join(str(segment.text).split())
            if not text:
                continue
            segments.append(
                {
                    "anchor_id": f"{anchor_id}_seg_{index:04d}",
                    "anchor_link": f"{anchor_id}_seg_{index:04d}",
                    "time_range_sec": [round(float(segment.start), 3), round(float(segment.end), 3)],
                    "text": text,
                    "source": "faster_whisper_asr",
                    "avg_logprob": _safe_float(getattr(segment, "avg_logprob", None)),
                    "no_speech_prob": _safe_float(getattr(segment, "no_speech_prob", None)),
                }
            )
            if len(segments) >= max_segments:
                break
        detected = getattr(info, "language", None)
        warning = None if segments else f"faster-whisper produced no segments; detected_language={detected}"
        return segments, warning
    except Exception as err:
        return [], f"faster-whisper ASR failed: {type(err).__name__}: {err}"


def _safe_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def create_image_structural_anchors(
    question: str,
    image_path: str,
    output_dir: Path,
    target_byte_ratio: float = DEFAULT_IMAGE_ANCHOR_TARGET_RATIO,
    max_crops: int = DEFAULT_IMAGE_DETAIL_MAX_CROPS,
) -> List[Dict[str, Any]]:
    """Create a global image anchor plus budgeted detail crops.

    The global anchor keeps spatial layout. Detail crops recover small text,
    colors, counts, and local object attributes when the question asks for
    information that a very compact global image often loses.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = infer_question_profile(question)
    needs = profile.get("needs", {})
    source_path = Path(image_path)
    source_bytes = source_path.stat().st_size if source_path.exists() else 0
    byte_budget = int(source_bytes * target_byte_ratio) if source_bytes else 0

    global_max_side = 640
    global_quality = 78
    image_bytes, _, global_anchor = create_image_global_anchor(
        image_path,
        max_side=global_max_side,
        quality=global_quality,
    )
    global_path = output_dir / "image_anchor_global.jpg"
    global_path.write_bytes(image_bytes)
    global_anchor["path"] = str(global_path)
    global_anchor["compression"]["target_byte_ratio"] = target_byte_ratio

    anchors: List[Dict[str, Any]] = [global_anchor]
    if max_crops <= 0:
        return anchors

    remaining_budget = max(0, byte_budget - len(image_bytes))
    crop_budget = max(remaining_budget, 4096)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        crop_dir = output_dir / "image_anchor_detail"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for index, spec in enumerate(_image_crop_specs(width, height, needs, question), start=1):
            if len(anchors) - 1 >= max_crops:
                break
            crop = image.crop(spec["box"])
            crop_bytes, crop_size = _jpeg_crop_under_budget(
                crop,
                crop_budget,
                DEFAULT_IMAGE_DETAIL_MAX_SIDE,
            )
            if not crop_bytes:
                continue
            anchor_id = f"image_anchor_crop_{index}"
            crop_path = crop_dir / f"{anchor_id}.jpg"
            crop_path.write_bytes(crop_bytes)
            bbox_norm = _box_to_norm(spec["box"], width, height)
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_link": anchor_id,
                    "type": "image_question_detail_crop",
                    "role": "Preserve question-conditioned local detail while staying within the compressed image budget.",
                    "path": str(crop_path),
                    "source_resolution": {"width": width, "height": height},
                    "resolution": {"width": crop_size[0], "height": crop_size[1]},
                    "bbox_norm": bbox_norm,
                    "expanded_bbox_norm": bbox_norm,
                    "region_hint": spec["label"],
                    "compression": {
                        "format": "jpeg",
                        "bytes": len(crop_bytes),
                        "target_byte_ratio": target_byte_ratio,
                        "strategy": "Budgeted question-conditioned detail crop.",
                    },
                }
            )
            remaining_budget -= len(crop_bytes)
            crop_budget = max(remaining_budget, 4096)
    return anchors


def _needs_image_detail(needs: Dict[str, Any]) -> bool:
    return bool(needs.get("ocr_or_detail") or needs.get("spatial") or needs.get("counting"))


def _image_crop_specs(width: int, height: int, needs: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    text = (question or "").lower()
    specs = []
    if any(word in text for word in ("right", "right side")):
        specs.append({"label": "right_side_detail", "box": _norm_box(width, height, 0.45, 0.05, 1.0, 0.95)})
    if any(word in text for word in ("left", "left side")):
        specs.append({"label": "left_side_detail", "box": _norm_box(width, height, 0.0, 0.05, 0.55, 0.95)})
    if any(word in text for word in ("upper", "top", "above")):
        specs.append({"label": "upper_detail", "box": _norm_box(width, height, 0.05, 0.0, 0.95, 0.5)})
    if any(word in text for word in ("lower", "bottom", "below")):
        specs.append({"label": "lower_detail", "box": _norm_box(width, height, 0.05, 0.5, 0.95, 1.0)})
    specs.append({"label": "center_detail", "box": _norm_box(width, height, 0.18, 0.18, 0.82, 0.82)})
    if needs.get("spatial") or needs.get("counting"):
        specs.extend(
            [
                {"label": "upper_left_context", "box": _norm_box(width, height, 0.0, 0.0, 0.58, 0.58)},
                {"label": "upper_right_context", "box": _norm_box(width, height, 0.42, 0.0, 1.0, 0.58)},
                {"label": "lower_left_context", "box": _norm_box(width, height, 0.0, 0.42, 0.58, 1.0)},
                {"label": "lower_right_context", "box": _norm_box(width, height, 0.42, 0.42, 1.0, 1.0)},
            ]
        )
    else:
        specs.extend(
            [
                {"label": "upper_detail_band", "box": _norm_box(width, height, 0.05, 0.0, 0.95, 0.42)},
                {"label": "middle_detail_band", "box": _norm_box(width, height, 0.05, 0.29, 0.95, 0.71)},
                {"label": "lower_detail_band", "box": _norm_box(width, height, 0.05, 0.58, 0.95, 1.0)},
            ]
        )
    return _dedupe_boxes(specs)


def _norm_box(width: int, height: int, x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(round(width * x1))))
    upper = max(0, min(height - 1, int(round(height * y1))))
    right = max(left + 1, min(width, int(round(width * x2))))
    lower = max(upper + 1, min(height, int(round(height * y2))))
    return left, upper, right, lower


def _box_to_norm(box: Tuple[int, int, int, int], width: int, height: int) -> List[float]:
    left, upper, right, lower = box
    return [
        round(left / width, 4),
        round(upper / height, 4),
        round(right / width, 4),
        round(lower / height, 4),
    ]


def _dedupe_boxes(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for spec in specs:
        box = spec["box"]
        if box in seen:
            continue
        seen.add(box)
        unique.append(spec)
    return unique


def _jpeg_crop_under_budget(
    image: Image.Image,
    byte_budget: int,
    max_side: int,
) -> Tuple[Optional[bytes], Tuple[int, int]]:
    smallest: Tuple[Optional[bytes], Tuple[int, int]] = (None, (0, 0))
    for side, quality in (
        (max_side, 84),
        (576, 82),
        (512, 80),
        (448, 78),
        (384, 76),
        (320, 74),
        (256, 72),
        (224, 70),
        (192, 68),
        (160, 66),
    ):
        candidate = image.copy()
        candidate.thumbnail((side, side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        candidate.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        smallest = (data, candidate.size)
        if len(data) <= byte_budget:
            return data, candidate.size
    return smallest


def _frame_step(source_fps: float, target_fps: float) -> int:
    if source_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def _resize_frame(frame_bgr: np.ndarray, max_side: int) -> Image.Image:
    cv2 = _require_cv2()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _frame_feature(image: Image.Image) -> np.ndarray:
    gray = image.convert("L").resize((24, 24), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(values)
    if norm == 0:
        return values
    return values / norm


def _feature_distance(a: Optional[np.ndarray], b: np.ndarray) -> float:
    if a is None:
        return 0.0
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


def _select_video_candidates(
    candidates: Sequence[Dict[str, Any]],
    max_frames: int,
) -> List[Dict[str, Any]]:
    if not candidates or max_frames <= 0:
        return []
    if len(candidates) <= max_frames:
        return list(candidates)

    selected_indexes = {0, len(candidates) - 1}
    boundary_budget = max(1, max_frames // 3)
    top_changes = sorted(
        range(1, len(candidates) - 1),
        key=lambda idx: candidates[idx]["change_score"],
        reverse=True,
    )[:boundary_budget]
    selected_indexes.update(top_changes)

    remaining_slots = max_frames - len(selected_indexes)
    if remaining_slots > 0:
        for idx in np.linspace(0, len(candidates) - 1, remaining_slots + 2)[1:-1]:
            selected_indexes.add(int(round(idx)))

    selected = [candidates[idx] for idx in sorted(selected_indexes)]
    while len(selected) > max_frames:
        selected = _drop_most_duplicate_frame(selected)
    return selected


def _drop_most_duplicate_frame(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(frames) <= 2:
        return frames[:-1]
    removable_scores = []
    for idx in range(1, len(frames) - 1):
        prev_distance = _feature_distance(frames[idx - 1]["feature"], frames[idx]["feature"])
        next_distance = _feature_distance(frames[idx]["feature"], frames[idx + 1]["feature"])
        removable_scores.append((prev_distance + next_distance, idx))
    _, remove_idx = min(removable_scores, key=lambda item: item[0])
    return [frame for idx, frame in enumerate(frames) if idx != remove_idx]


def _write_video_frames(
    selected: Sequence[Dict[str, Any]],
    frame_dir: Path,
    jpeg_quality: int,
    anchor_id: str,
) -> List[Dict[str, Any]]:
    frames = []
    for output_index, item in enumerate(selected, start=1):
        anchor_link = f"{anchor_id}_frame_{output_index:04d}"
        frame_path = frame_dir / f"{anchor_link}.jpg"
        image = item["image"]
        image.save(frame_path, format="JPEG", quality=jpeg_quality, optimize=True)
        frames.append(
            {
                "anchor_id": anchor_link,
                "anchor_link": anchor_link,
                "path": str(frame_path),
                "time_sec": item["time_sec"],
                "frame_index": item["frame_index"],
                "resolution": {"width": image.width, "height": image.height},
                "bytes": frame_path.stat().st_size,
                "change_score": item["change_score"],
                "summary": "Representative low-FPS frame selected for temporal coverage or visual change.",
            }
        )
    return frames


def _write_low_fps_clip(
    selected: Sequence[Dict[str, Any]],
    output_dir: Path,
    anchor_id: str,
    target_fps: float,
) -> Optional[Path]:
    if not selected:
        return None
    cv2 = _require_cv2()
    first = selected[0]["image"]
    clip_path = output_dir / f"{anchor_id}.mp4"
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, target_fps),
        (first.width, first.height),
    )
    if not writer.isOpened():
        return None
    for item in selected:
        image = item["image"]
        if image.size != first.size:
            image = image.resize(first.size, Image.Resampling.LANCZOS)
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    return clip_path if clip_path.exists() and clip_path.stat().st_size > 0 else None


def _require_cv2():
    try:
        import cv2
    except ImportError as err:
        raise RuntimeError(
            "OpenCV is required for video structural anchors. Install opencv-python or use an environment that provides cv2."
        ) from err
    return cv2


def _event_boundaries(frames: Sequence[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    boundaries = sorted(
        [frame for frame in frames if frame.get("change_score", 0.0) > 0.0],
        key=lambda item: item["change_score"],
        reverse=True,
    )[:max_count]
    return [
        {
            "time_sec": frame["time_sec"],
            "anchor_link": frame["anchor_link"],
            "change_score": frame["change_score"],
        }
        for frame in sorted(boundaries, key=lambda item: item["time_sec"])
    ]


def _compress_audio_ffmpeg(
    audio_path: str,
    output_path: Path,
    sample_rate: int,
    bitrate: str,
) -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-b:a",
        bitrate,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return result.stderr.strip()
    return None


def _compress_audio_python(
    audio_path: str,
    output_dir: Path,
    sample_rate: int,
    anchor_id: str,
) -> Tuple[Path, Optional[str]]:
    """Create a real low-cost audio anchor when ffmpeg is unavailable.

    soundfile/libsndfile can write MP3/OGG on the AutoDL image. If those
    codecs are unavailable, fall back to low-sample-rate mono PCM WAV; that is
    still a genuine compressed structural anchor compared with the source WAV.
    """
    try:
        import soundfile as sf
    except Exception as err:
        return Path(audio_path), f"ffmpeg and soundfile are unavailable; using original audio path as anchor: {err}"

    try:
        read_path = Path(audio_path)
        if read_path.suffix.lower() not in {".wav", ".flac", ".ogg", ".mp3"}:
            decoded_path, decode_warning = _extract_audio_wav_for_asr(str(read_path), output_dir / f"{anchor_id}_decoded.wav")
            if decoded_path:
                read_path = decoded_path
            else:
                return Path(audio_path), f"ffmpeg not found and PyAV audio extraction failed; using original audio path as anchor: {decode_warning}"
        samples, source_rate = sf.read(str(read_path), dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = _resample_audio(samples.astype(np.float32), source_rate, sample_rate)
        samples = np.clip(samples, -1.0, 1.0)

        attempts = [
            (output_dir / f"{anchor_id}.mp3", {"format": "MP3", "subtype": "MPEG_LAYER_III"}),
            (output_dir / f"{anchor_id}.ogg", {"format": "OGG", "subtype": "OPUS"}),
            (output_dir / f"{anchor_id}.wav", {"format": "WAV", "subtype": "PCM_16"}),
        ]
        errors = []
        for candidate_path, kwargs in attempts:
            try:
                sf.write(str(candidate_path), samples, sample_rate, **kwargs)
                if candidate_path.exists() and candidate_path.stat().st_size > 0:
                    return candidate_path, "ffmpeg not found; generated low-sample-rate mono audio anchor with soundfile."
            except Exception as err:
                errors.append(f"{candidate_path.suffix}: {err}")
        return Path(audio_path), "ffmpeg not found and soundfile compression failed; using original audio path as anchor. " + " | ".join(errors)
    except Exception as err:
        return Path(audio_path), f"ffmpeg not found and Python audio compression failed; using original audio path as anchor: {err}"


def _resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if samples.size == 0 or source_rate <= 0 or source_rate == target_rate:
        return samples
    duration = samples.shape[0] / float(source_rate)
    target_length = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _load_audio_samples(audio_path: str, sample_rate: int) -> Tuple[np.ndarray, int, Optional[str]]:
    if shutil.which("ffmpeg"):
        ffmpeg = shutil.which("ffmpeg")
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode == 0 and result.stdout:
            samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, sample_rate, None
        return np.asarray([], dtype=np.float32), sample_rate, result.stderr.decode("utf-8", errors="ignore").strip()

    path = Path(audio_path)
    if path.suffix.lower() != ".wav":
        decoded_path, decode_warning = _extract_audio_wav_for_asr(str(path), path.with_suffix(".mtec_audio.wav"))
        if decoded_path:
            path = decoded_path
        else:
            return np.asarray([], dtype=np.float32), sample_rate, f"ffmpeg not found and PyAV audio extraction failed: {decode_warning}"
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            source_rate = audio.getframerate()
            frames = audio.readframes(audio.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            return samples, source_rate, None
    except wave.Error as err:
        return np.asarray([], dtype=np.float32), sample_rate, str(err)


def _audio_energy_profile(
    samples: np.ndarray,
    sample_rate: int,
    window_seconds: float,
) -> Dict[str, Any]:
    window_size = max(1, int(round(sample_rate * window_seconds)))
    windows = []
    for start in range(0, samples.shape[0], window_size):
        stop = min(samples.shape[0], start + window_size)
        chunk = samples[start:stop]
        if chunk.size == 0:
            continue
        rms = float(math.sqrt(float(np.mean(chunk ** 2))))
        windows.append(
            {
                "start_sec": round(start / sample_rate, 3),
                "end_sec": round(stop / sample_rate, 3),
                "rms": round(rms, 6),
            }
        )
    rms_values = np.asarray([item["rms"] for item in windows], dtype=np.float32)
    return {
        "window_seconds": window_seconds,
        "rms_mean": round(float(rms_values.mean()), 6) if rms_values.size else None,
        "rms_std": round(float(rms_values.std()), 6) if rms_values.size else None,
        "rms_max": round(float(rms_values.max()), 6) if rms_values.size else None,
        "windows": windows,
    }


def _select_audio_events(
    energy_profile: Dict[str, Any],
    window_seconds: float,
    max_segments: int,
    anchor_id: str,
) -> List[Dict[str, Any]]:
    windows = energy_profile.get("windows", [])
    if not windows:
        return []

    mean = energy_profile.get("rms_mean") or 0.0
    std = energy_profile.get("rms_std") or 0.0
    high_threshold = mean + std
    silence_threshold = mean * 0.2

    scored = []
    for window in windows:
        rms = window["rms"]
        if rms >= high_threshold and rms > 0:
            scored.append(("high_energy", rms - high_threshold, window))
        elif rms <= silence_threshold:
            scored.append(("silence_or_pause", silence_threshold - rms, window))

    if not scored:
        scored = [
            ("representative_audio", window["rms"], window)
            for window in sorted(windows, key=lambda item: item["rms"], reverse=True)[:max_segments]
        ]

    scored = sorted(scored, key=lambda item: item[1], reverse=True)[:max_segments]
    events = []
    for index, (event_type, score, window) in enumerate(sorted(scored, key=lambda item: item[2]["start_sec"]), start=1):
        anchor_link = f"{anchor_id}_event_{index:04d}"
        events.append(
            {
                "anchor_id": anchor_link,
                "anchor_link": anchor_link,
                "event_type": event_type,
                "time": f"{window['start_sec']:.2f}-{window['end_sec']:.2f}s",
                "start_sec": window["start_sec"],
                "end_sec": window["end_sec"],
                "score": round(float(score), 6),
                "rms": window["rms"],
                "content": _audio_event_content(event_type, window_seconds),
            }
        )
    return events


def _audio_event_content(event_type: str, window_seconds: float) -> str:
    if event_type == "high_energy":
        return f"High-energy acoustic segment over about {window_seconds:.1f}s; inspect for alarm, impact, emphasis, or sudden speech."
    if event_type == "silence_or_pause":
        return f"Low-energy or pause segment over about {window_seconds:.1f}s; useful for rhythm, hesitation, or event boundary cues."
    return f"Representative acoustic segment over about {window_seconds:.1f}s."


def _probe_duration(path: str) -> Optional[float]:
    payload = probe_media(path)
    try:
        return round(float(payload.get("format", {}).get("duration")), 3)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 6)
