import json
import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from zoomrefine.mtec_prompt_plus import (
        DEFAULT_TOTAL_BUDGET,
        build_low_resolution_anchor_package,
        create_image_global_anchor,
    )
except ImportError:
    from mtec_prompt_plus import (
        DEFAULT_TOTAL_BUDGET,
        build_low_resolution_anchor_package,
        create_image_global_anchor,
    )


DEFAULT_VIDEO_TARGET_FPS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 16
DEFAULT_VIDEO_MAX_SIDE = 512
DEFAULT_VIDEO_JPEG_QUALITY = 82
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_BITRATE = "32k"
DEFAULT_AUDIO_WINDOW_SECONDS = 1.0


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
        low_audio_path = Path(audio_path)
        compression_warning = "ffmpeg not found; using original audio path as anchor."

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
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_anchor = None
    if image_path:
        image_bytes, _, image_anchor = create_image_global_anchor(image_path)
        image_anchor_path = output_path / "image_anchor_global.jpg"
        image_anchor_path.write_bytes(image_bytes)
        image_anchor["path"] = str(image_anchor_path)

    video_anchor = None
    if video_path:
        video_anchor = create_video_structural_anchor(video_path, str(output_path))

    audio_anchor = None
    if audio_path:
        audio_anchor = create_audio_structural_anchor(audio_path, str(output_path))

    package = build_low_resolution_anchor_package(
        question=question,
        global_anchor=image_anchor,
        video_anchor=video_anchor,
        audio_anchor=audio_anchor,
        total_budget=total_budget,
    )
    package["media_probe"] = {
        "image": {"path": image_path} if image_path else None,
        "video": probe_media(video_path) if video_path else None,
        "audio": probe_media(audio_path) if audio_path else None,
    }
    return package


def _frame_step(source_fps: float, target_fps: float) -> int:
    if source_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def _resize_frame(frame_bgr: np.ndarray, max_side: int) -> Image.Image:
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
        return np.asarray([], dtype=np.float32), sample_rate, "ffmpeg not found and audio is not WAV."
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
