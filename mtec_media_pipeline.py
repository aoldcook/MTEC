import html
import json
import io
import math
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

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
DEFAULT_VIDEO_TUBELET_MAX_STORYBOARDS = 4
DEFAULT_VIDEO_TUBELET_MAX_SIDE = 768
DEFAULT_VIDEO_OCR_MAX_REGIONS = 8
DEFAULT_VIDEO_MOTION_MAX_REGIONS = 8
DEFAULT_VIDEO_OBJECT_MAX_DETECTIONS = 12
DEFAULT_VIDEO_SCENE_MAX_SEGMENTS = 12
DEFAULT_VIDEO_TRANSCRIPT_MAX_SEGMENTS = 48
DEFAULT_VIDEO_QUERY_MAX_SEGMENTS = 12
DEFAULT_VIDEO_QUERY_WINDOW_PADDING_SEC = 8.0
DEFAULT_VIDEO_ASR_MODEL = "base.en"
_FASTER_WHISPER_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_PADDLE_OCR_MODEL: Any = None
_YOLO_MODEL_CACHE: Dict[str, Any] = {}


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
    video_subtitle_path: Optional[str] = None,
    video_query_retrieval: bool = True,
    video_query_max_segments: int = DEFAULT_VIDEO_QUERY_MAX_SEGMENTS,
    video_query_window_padding_sec: float = DEFAULT_VIDEO_QUERY_WINDOW_PADDING_SEC,
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

    transcript_anchor = None
    question_time_windows: List[Dict[str, Any]] = []
    if video_path and include_video_transcript:
        transcript_anchor = create_video_transcript_anchor(
            video_path=video_path,
            output_dir=str(output_path),
            backend=video_transcript_backend,
            model_name=video_asr_model,
            language=video_asr_language,
            max_segments=video_transcript_max_segments,
            anchor_id="video_transcript_anchor",
            external_subtitle_path=video_subtitle_path,
            question=question,
            query_retrieval=video_query_retrieval,
            query_max_segments=video_query_max_segments,
            query_window_padding_sec=video_query_window_padding_sec,
        )
        question_time_windows = transcript_anchor.get("question_relevant_time_windows") or []

    video_evidence_anchor = None
    video_ocr_image_anchors: List[Dict[str, Any]] = []
    if video_path and video_anchor:
        video_evidence_anchor, video_ocr_image_anchors = create_video_evidence_extraction_anchor(
            question=question,
            video_path=video_path,
            output_dir=output_path,
            video_anchor=video_anchor,
            question_time_windows=question_time_windows,
            transcript_anchor=transcript_anchor,
        )
        if video_ocr_image_anchors:
            image_anchor = _merge_anchor_lists(image_anchor, video_ocr_image_anchors)

    if video_path and video_anchor:
        video_detail_anchors = create_video_detail_frame_anchors(
            question=question,
            video_path=video_path,
            output_dir=output_path,
            video_anchor=video_anchor,
            max_crops=video_detail_max_crops,
            max_side=video_detail_max_side,
            question_time_windows=question_time_windows,
        )
        if video_detail_anchors:
            image_anchor = _merge_anchor_lists(image_anchor, video_detail_anchors)
        video_tubelet_anchors = create_video_tubelet_storyboard_anchors(
            question=question,
            video_path=video_path,
            output_dir=output_path,
            video_anchor=video_anchor,
            max_storyboards=max(
                1,
                min(
                    DEFAULT_VIDEO_TUBELET_MAX_STORYBOARDS,
                    math.ceil(max(1, video_detail_max_crops) / 2),
                ),
            ),
            max_side=min(
                max(video_max_side, DEFAULT_VIDEO_TUBELET_MAX_SIDE),
                max(video_detail_max_side, DEFAULT_VIDEO_TUBELET_MAX_SIDE),
            ),
            question_time_windows=question_time_windows,
        )
        if video_tubelet_anchors:
            image_anchor = _merge_anchor_lists(image_anchor, video_tubelet_anchors)

    audio_anchor = None
    if audio_path:
        audio_anchor = create_audio_structural_anchor(audio_path, str(output_path))
    elif video_path and include_video_audio:
        audio_anchor = create_audio_structural_anchor(
            video_path,
            str(output_path),
            anchor_id="video_audio_anchor_low_bitrate",
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
    if video_evidence_anchor:
        package["low_resolution_anchor"]["video_evidence_anchor"] = [video_evidence_anchor]
    package["media_probe"] = {
        "image": {"path": image_path} if image_path else None,
        "video": probe_media(video_path) if video_path else None,
        "audio": probe_media(audio_path) if audio_path else None,
    }
    return package


def create_video_evidence_extraction_anchor(
    question: str,
    video_path: str,
    output_dir: Path,
    video_anchor: Dict[str, Any],
    question_time_windows: Optional[List[Dict[str, Any]]] = None,
    transcript_anchor: Optional[Dict[str, Any]] = None,
    anchor_id: str = "video_evidence_extractor",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    evidence_dir = output_dir / anchor_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    scene_segments, scene_warning = _detect_scene_segments(video_path, video_anchor)
    if scene_warning:
        warnings.append(scene_warning)
    motion_regions, motion_warning = _detect_motion_regions(
        video_path,
        video_anchor,
        max_regions=DEFAULT_VIDEO_MOTION_MAX_REGIONS,
    )
    if motion_warning:
        warnings.append(motion_warning)
    ocr_regions, ocr_image_anchors, ocr_warning = _extract_video_ocr_regions(
        question=question,
        video_path=video_path,
        output_dir=evidence_dir,
        video_anchor=video_anchor,
        max_regions=DEFAULT_VIDEO_OCR_MAX_REGIONS,
    )
    if ocr_warning:
        warnings.append(ocr_warning)
    object_detections, object_image_anchors, object_warning = _extract_video_object_detections(
        question=question,
        video_path=video_path,
        output_dir=evidence_dir,
        video_anchor=video_anchor,
        max_detections=DEFAULT_VIDEO_OBJECT_MAX_DETECTIONS,
    )
    if object_warning:
        warnings.append(object_warning)

    query_profile = _video_query_extraction_profile(question)
    temporal_scope = _resolve_video_temporal_scope(question, video_anchor, scene_segments)
    priority_windows = _question_priority_time_windows(question, video_anchor, temporal_scope, scene_segments)
    deterministic_evidence = _build_deterministic_evidence_engine(
        question=question,
        query_profile=query_profile,
        temporal_scope=temporal_scope,
        scene_segments=scene_segments,
        motion_regions=motion_regions,
        ocr_regions=ocr_regions,
        object_detections=object_detections,
        transcript_anchor=transcript_anchor,
    )
    evidence_units = _build_video_evidence_units(
        query_profile=query_profile,
        temporal_scope=temporal_scope,
        scene_segments=scene_segments,
        motion_regions=motion_regions,
        ocr_regions=ocr_regions,
        object_detections=object_detections,
        question_time_windows=(priority_windows + (question_time_windows or [])),
    )

    payload = {
        "anchor_id": anchor_id,
        "anchor_link": anchor_id,
        "type": "video_task_aware_evidence_extraction",
        "role": "Pre-LLM task-aware evidence extractor using scene cuts, motion saliency, OCR regions, ASR windows, scoring, and diversity hints.",
        "source_video_path": str(Path(video_path)),
        "query_profile": query_profile,
        "temporal_scope": temporal_scope,
        "scene_segments": scene_segments,
        "motion_regions": motion_regions,
        "ocr_regions": ocr_regions,
        "object_detections": object_detections,
        "deterministic_evidence": deterministic_evidence,
        "question_time_windows": question_time_windows or [],
        "priority_time_windows": priority_windows,
        "evidence_units": evidence_units,
        "scoring_policy": {
            "query_relevance": "question type and keyword overlap",
            "temporal_importance": "scene boundary, motion peak, start/current/count windows, ASR hit windows",
            "motion_saliency": "OpenCV frame-difference bbox intensity",
            "ocr_confidence": "Tesseract text confidence and readable crop",
            "diversity": "prefer different time windows and evidence types before repeating nearby evidence",
        },
        "warnings": warnings,
    }
    path = evidence_dir / f"{anchor_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["path"] = str(path)
    payload["compression"] = {
        "scene_segments": len(scene_segments),
        "motion_regions": len(motion_regions),
        "ocr_regions": len(ocr_regions),
        "evidence_units": len(evidence_units),
        "strategy": "Structured task-aware evidence replaces part of the burden previously placed on one generic VLM pass.",
    }
    return payload, ocr_image_anchors + object_image_anchors


def _video_query_extraction_profile(question: str) -> Dict[str, Any]:
    text = (question or "").lower()
    question_types: List[str] = []
    target_terms = _query_terms_for_video_evidence(question)
    if any(term in text for term in ("how many", "number of", "count", "many")):
        question_types.append("count")
    if any(term in text for term in ("before", "after", "first", "then", "next", "finally", "order", "sequence")):
        question_types.append("temporal_order")
    if any(term in text for term in ("where", "left", "right", "above", "below", "near", "between", "position")):
        question_types.append("spatial_relation")
    if any(term in text for term in ("text", "word", "sign", "label", "title", "screen", "score", "number", "read")):
        question_types.append("ocr")
    if any(term in text for term in ("sound", "audio", "hear", "heard", "say", "says", "spoken", "voice", "music")):
        question_types.append("audio_visual")
    if any(term in text for term in ("move", "moving", "enter", "leave", "pick", "put", "open", "close", "turn", "action", "doing")):
        question_types.append("action_motion")
    if any(term in text for term in ("absent", "missing", "not appear", "not shown", "which color")):
        question_types.append("absence_verification")
    if not question_types:
        question_types.append("semantic_video")
    return {
        "question_types": question_types,
        "target_terms": target_terms[:32],
        "required_evidence": _required_video_evidence(question_types),
    }


def _required_video_evidence(question_types: List[str]) -> List[str]:
    required = ["global_timeline", "low_fps_context"]
    mapping = {
        "count": ["multi_time_instance_coverage", "duplicate_guard"],
        "temporal_order": ["event_boundaries", "before_during_after"],
        "spatial_relation": ["shared_context_frame", "relative_position"],
        "ocr": ["ocr_text_regions", "high_resolution_text_crop"],
        "audio_visual": ["speech_or_audio_timestamp", "nearby_visual_frame"],
        "action_motion": ["motion_region", "state_change_tubelet"],
        "absence_verification": ["option_presence_absence_check", "wide_context"],
    }
    for question_type in question_types:
        for item in mapping.get(question_type, []):
            if item not in required:
                required.append(item)
    return required


def _parse_mcq_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for match in re.finditer(r"(?m)([A-F])\.\s*([^A-F\n][^\n]*)", str(question or "")):
        options[match.group(1).upper()] = " ".join(match.group(2).strip().split())
    return options


def _resolve_video_temporal_scope(
    question: str,
    video_anchor: Dict[str, Any],
    scene_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    text = (question or "").lower()
    duration = float(video_anchor.get("source_duration_sec") or 0.0)
    if duration <= 0:
        return {
            "type": "full_video",
            "time_range_sec": None,
            "confidence": 0.5,
            "source": "missing_duration",
            "instructions": ["No reliable duration; evidence scope defaults to all available anchors."],
        }

    def scoped(scope_type: str, start: float, end: float, confidence: float, source: str, **extra: Any) -> Dict[str, Any]:
        start = _clamp_float(start, 0.0, duration)
        end = _clamp_float(max(end, start + 0.5), 0.0, duration)
        payload = {
            "type": scope_type,
            "time_range_sec": [round(start, 3), round(end, 3)],
            "confidence": round(confidence, 3),
            "source": source,
            "instructions": [
                "Use evidence inside temporal_scope first.",
                "Mark evidence outside temporal_scope as scope_match=false and do not use it for final decisions unless no scoped evidence exists.",
            ],
        }
        payload.update(extra)
        return payload

    if any(term in text for term in ("beginning", "at the start", "start of", "opening", "initially", "displayed at the beginning")):
        end = min(duration, max(6.0, min(15.0, duration * 0.15)))
        return scoped("beginning", 0.0, end, 0.9, "keyword_beginning")
    if any(term in text for term in ("at the end", "ending", "finally", "final scene")):
        window = min(duration, max(6.0, min(15.0, duration * 0.15)))
        return scoped("end", max(0.0, duration - window), duration, 0.86, "keyword_end")
    clip_match = re.search(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+(?:clip|segment|scene)\b", text)
    if clip_match:
        order_map = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
        clip_index = order_map.get(clip_match.group(1), 1)
        usable_segments = _merge_short_scene_segments(scene_segments, duration, min_duration=1.2)
        if len(usable_segments) < clip_index:
            usable_segments = _uniform_segments(duration, max(clip_index, 3))
        segment = usable_segments[min(clip_index - 1, len(usable_segments) - 1)]
        start, end = _coerce_time_window(segment)
        pad = min(1.5, max(0.0, (end - start) * 0.25))
        return scoped(
            "clip_index",
            start - pad,
            end + pad,
            0.76 if segment.get("source") != "uniform_scope_fallback" else 0.62,
            "scene_segment_clip_index",
            clip_index=clip_index,
            segment_id=segment.get("id"),
        )
    scene_terms = {
        "train": ("on the train", "in the train", "train"),
        "stage": ("on the stage", "stage"),
        "table": ("on the table", "table"),
        "screen": ("screen", "laptop"),
        "court": ("court", "basketball"),
    }
    for scene_name, terms in scene_terms.items():
        if any(term in text for term in terms):
            return scoped(
                "scene_specific",
                0.0,
                duration,
                0.55,
                "scene_keyword_no_semantic_locator",
                target_scene=scene_name,
                instructions=[
                    f"Locate the {scene_name} scene before using appearance evidence.",
                    "Evidence from other scenes should be treated as scope_match=false.",
                ],
            )
    if any(term in text for term in ("current score", "ongoing game", "current state", "currently", "now")):
        return scoped(
            "current_state",
            0.0,
            duration,
            0.58,
            "keyword_current_requires_visual_state",
            instructions=[
                "Current-state questions require the frame where the queried state is visible; do not prefer late transcript or unrelated later frames.",
                "For score questions, use scoreboard OCR/crops with direct time support and avoid guessing from later score changes.",
            ],
        )
    return {
        "type": "full_video",
        "time_range_sec": [0.0, round(duration, 3)],
        "confidence": 0.7,
        "source": "default_full_video",
        "instructions": ["No narrow temporal constraint detected; aggregate evidence across the full video."],
    }


def _merge_short_scene_segments(scene_segments: List[Dict[str, Any]], duration: float, min_duration: float) -> List[Dict[str, Any]]:
    if not scene_segments:
        return _uniform_segments(duration, 3)
    merged: List[Dict[str, Any]] = []
    for segment in scene_segments:
        start, end = _coerce_time_window(segment)
        if merged and end - start < min_duration:
            merged[-1]["time_range_sec"][1] = round(max(float(merged[-1]["time_range_sec"][1]), end), 3)
            merged[-1]["duration_sec"] = round(float(merged[-1]["time_range_sec"][1]) - float(merged[-1]["time_range_sec"][0]), 3)
            merged[-1]["source"] = f"{merged[-1].get('source')}_merged_short"
        else:
            merged.append(dict(segment))
    return merged


def _uniform_segments(duration: float, count: int) -> List[Dict[str, Any]]:
    count = max(1, count)
    return [
        {
            "id": f"scope_uniform_{index + 1:04d}",
            "time_range_sec": [round(duration * index / count, 3), round(duration * (index + 1) / count, 3)],
            "duration_sec": round(duration / count, 3),
            "source": "uniform_scope_fallback",
        }
        for index in range(count)
    ]


def _scope_match(time_value: Any, temporal_scope: Dict[str, Any]) -> bool:
    scope_type = temporal_scope.get("type")
    if scope_type in {None, "full_video", "current_state", "scene_specific"}:
        return True
    scope_range = temporal_scope.get("time_range_sec")
    if not isinstance(scope_range, (list, tuple)) or len(scope_range) < 2:
        return True
    start, end = _safe_float_value(scope_range[0]), _safe_float_value(scope_range[1])
    if isinstance(time_value, (list, tuple)) and len(time_value) >= 2:
        item_start, item_end = _safe_float_value(time_value[0]), _safe_float_value(time_value[1])
        return item_end >= start and item_start <= end
    time_sec = _safe_float_value(time_value)
    return start <= time_sec <= end


def _add_scope_flags(items: List[Dict[str, Any]], temporal_scope: Dict[str, Any]) -> List[Dict[str, Any]]:
    flagged = []
    for item in items:
        copy = dict(item)
        copy["scope_match"] = _scope_match(copy.get("time_range_sec") or copy.get("time_sec"), temporal_scope)
        flagged.append(copy)
    return flagged


def _build_deterministic_evidence_engine(
    question: str,
    query_profile: Dict[str, Any],
    temporal_scope: Dict[str, Any],
    scene_segments: List[Dict[str, Any]],
    motion_regions: List[Dict[str, Any]],
    ocr_regions: List[Dict[str, Any]],
    object_detections: List[Dict[str, Any]],
    transcript_anchor: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    question_types = set(query_profile.get("question_types") or [])
    options = _parse_mcq_options(question)
    scoped_objects = _add_scope_flags(object_detections, temporal_scope)
    scoped_ocr = _add_scope_flags(ocr_regions, temporal_scope)
    scoped_motion = _add_scope_flags(motion_regions, temporal_scope)
    transcript_scope = _transcript_scope_evidence(transcript_anchor, temporal_scope, question_types)
    constraints = _llm_constraints_for_query(question, question_types, temporal_scope)
    hard_evidence: Dict[str, Any] = {
        "scoped_motion": scoped_motion[:DEFAULT_VIDEO_MOTION_MAX_REGIONS],
        "scoped_ocr": scoped_ocr[:DEFAULT_VIDEO_OCR_MAX_REGIONS],
        "scoped_objects": scoped_objects[:DEFAULT_VIDEO_OBJECT_MAX_DETECTIONS],
        "transcript_scope": transcript_scope,
    }
    quality_missing: List[str] = []
    if "count" in question_types:
        tracks = _track_object_detections_iou(scoped_objects, target_labels=_count_target_labels(question))
        hard_evidence["count_tracks"] = tracks
        if not tracks.get("confirmed_tracks"):
            quality_missing.append("No confirmed multi-frame object tracks for count question.")
    if "absence_verification" in question_types:
        visible_set = _build_visible_set_evidence(options, scoped_objects, scoped_ocr, question)
        hard_evidence["visible_set"] = visible_set
        if not visible_set.get("visible_options") and options:
            quality_missing.append("No option-level visible set evidence confirmed.")
    if "ocr" in question_types:
        ocr_votes = _build_ocr_vote_evidence(scoped_ocr)
        hard_evidence["ocr_votes"] = ocr_votes
        if not ocr_votes.get("voted_text"):
            quality_missing.append("OCR text is uncertain or unreadable; do not infer text/model from appearance.")
    return {
        "version": "deterministic_evidence_engine_v1",
        "question_analysis": {
            "question_types": list(question_types),
            "options": options,
            "target_terms": query_profile.get("target_terms") or [],
        },
        "temporal_scope": temporal_scope,
        "hard_evidence": hard_evidence,
        "evidence_quality": {
            "missing": quality_missing,
            "conflicts": [],
            "needs_reextract": bool(quality_missing),
        },
        "constraints_for_llm": constraints,
    }


def _llm_constraints_for_query(question: str, question_types: set, temporal_scope: Dict[str, Any]) -> List[str]:
    text = (question or "").lower()
    constraints = [
        "Evidence extractor must not output candidate_answer, preliminary_answer, best_option, or final answer.",
        "Every observation must be timestamped and grounded to an anchor_link when available.",
        "Final verification must evaluate each option independently as supported, contradicted, or unknown.",
    ]
    if temporal_scope.get("type") not in {None, "full_video"}:
        constraints.append("Do not use evidence outside temporal_scope for the answer unless all scoped evidence is missing.")
    if "count" in question_types:
        constraints.append("For count questions, use count_tracks/instances instead of guessing from a single crop.")
    if "absence_verification" in question_types:
        constraints.append("For missing/absent questions, use visible_set aggregation over the valid scope, not one frame where something is unseen.")
    if "ocr" in question_types or any(term in text for term in ("model", "screen", "score", "text", "advertised")):
        constraints.append("For model/text/score questions, do not infer from appearance if OCR is unreadable or outside temporal scope.")
    if any(term in text for term in ("beginning", "at the start", "opening")):
        constraints.append("For beginning questions, later transcript or late visual evidence is scope_mismatch and cannot override opening visual evidence.")
    return constraints


def _transcript_scope_evidence(
    transcript_anchor: Optional[Dict[str, Any]],
    temporal_scope: Dict[str, Any],
    question_types: set,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not transcript_anchor:
        return rows
    visual_count_question = "count" in question_types and temporal_scope.get("type") in {"beginning", "clip_index", "scene_specific"}
    for segment in (transcript_anchor.get("question_relevant_segments") or transcript_anchor.get("segments") or [])[:24]:
        time_range = segment.get("time_range_sec")
        scope_match = _scope_match(time_range, temporal_scope)
        use_for_answer = scope_match and not visual_count_question
        reason = "Transcript is inside temporal scope."
        if not scope_match:
            reason = "Transcript is outside temporal scope."
        elif visual_count_question:
            reason = "Question asks for visual counting in a scoped segment; transcript is auxiliary only."
        rows.append(
            {
                "anchor_link": segment.get("anchor_link"),
                "time_range_sec": time_range,
                "text": segment.get("text"),
                "scope_match": scope_match,
                "task_match": "weak" if visual_count_question else "normal",
                "use_for_answer": use_for_answer,
                "reason": reason,
            }
        )
    return rows


def _count_target_labels(question: str) -> List[str]:
    text = (question or "").lower()
    if any(term in text for term in ("person", "people", "men", "women", "challenger", "player", "singer", "dancer")):
        return ["person"]
    if "ball" in text:
        return ["sports ball"]
    return []


def _track_object_detections_iou(detections: List[Dict[str, Any]], target_labels: List[str]) -> Dict[str, Any]:
    relevant = [
        dict(item)
        for item in detections
        if item.get("scope_match", True)
        and (not target_labels or str(item.get("label") or "").lower() in target_labels)
    ]
    tracks: List[Dict[str, Any]] = []
    for detection in sorted(relevant, key=lambda item: float(item.get("time_sec") or 0.0)):
        label = str(detection.get("label") or "").lower()
        best_track = None
        best_iou = 0.0
        for track in tracks:
            if track.get("label") != label:
                continue
            iou = _bbox_iou(track.get("last_bbox_norm") or [], detection.get("bbox_norm") or [])
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is not None and best_iou >= 0.30:
            best_track["detections"].append(detection.get("anchor_link"))
            best_track["frames_seen"] += 1
            best_track["last_bbox_norm"] = detection.get("bbox_norm")
            best_track["time_range_sec"][1] = detection.get("time_sec")
            best_track["representative_times"].append(detection.get("time_sec"))
            best_track["confidence"] = round(max(float(best_track.get("confidence") or 0.0), float(detection.get("confidence") or 0.0)), 4)
        else:
            tracks.append(
                {
                    "track_id": f"T{len(tracks) + 1:02d}",
                    "label": label or detection.get("label"),
                    "detections": [detection.get("anchor_link")],
                    "frames_seen": 1,
                    "time_range_sec": [detection.get("time_sec"), detection.get("time_sec")],
                    "representative_times": [detection.get("time_sec")],
                    "last_bbox_norm": detection.get("bbox_norm"),
                    "confidence": detection.get("confidence"),
                }
            )
    confirmed = []
    uncertain = []
    for track in tracks:
        item = {key: value for key, value in track.items() if key != "last_bbox_norm"}
        item["representative_times"] = item["representative_times"][:5]
        if int(track.get("frames_seen") or 0) >= 2:
            confirmed.append(item)
        else:
            uncertain.append(item)
    return {
        "method": "yolo_iou_tracking_v1",
        "target_labels": target_labels or ["question_relevant_objects"],
        "confirmed_tracks": confirmed,
        "uncertain_tracks": uncertain,
        "count_value": len(confirmed),
        "count_confidence": round(min(0.95, 0.35 + 0.12 * len(confirmed)), 3) if confirmed else 0.0,
        "warning": "IoU tracking is a deterministic de-dup aid, not identity ReID; use uncertain_tracks conservatively.",
    }


def _bbox_iou(box_a: Sequence[Any], box_b: Sequence[Any]) -> float:
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _build_ocr_vote_evidence(ocr_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = []
    votes: Dict[str, Dict[str, Any]] = {}
    for region in ocr_regions:
        text = " ".join(str(region.get("text") or "").split())
        if not text or not region.get("scope_match", True):
            continue
        norm = re.sub(r"[^a-z0-9]+", "", text.lower())
        if not norm:
            continue
        conf = float(region.get("confidence") or 0.0)
        candidates.append(
            {
                "engine": region.get("source"),
                "text": text,
                "confidence": conf,
                "time_sec": region.get("time_sec"),
                "anchor_link": region.get("anchor_link"),
                "scope_match": region.get("scope_match", True),
            }
        )
        item = votes.setdefault(norm, {"text": text, "score": 0.0, "count": 0})
        item["score"] += conf
        item["count"] += 1
    if not votes:
        return {"ocr_candidates": candidates, "voted_text": None, "ocr_status": "unreadable_or_out_of_scope"}
    winner = max(votes.values(), key=lambda item: (item["count"], item["score"]))
    status = "confident" if winner["count"] >= 2 and winner["score"] / max(1, winner["count"]) >= 0.6 else "uncertain"
    return {"ocr_candidates": candidates[:12], "voted_text": winner["text"] if status == "confident" else None, "ocr_status": status}


def _build_visible_set_evidence(
    options: Dict[str, str],
    object_detections: List[Dict[str, Any]],
    ocr_regions: List[Dict[str, Any]],
    question: str,
) -> Dict[str, Any]:
    option_terms = {letter: _option_terms(text) for letter, text in options.items()}
    visible: Dict[str, Dict[str, Any]] = {
        letter: {"option": options.get(letter), "seen": False, "times": [], "evidence": [], "confidence": 0.0}
        for letter in options
    }
    for letter, terms in option_terms.items():
        for detection in object_detections:
            label = str(detection.get("label") or "").lower()
            if detection.get("scope_match", True) and any(term in label for term in terms):
                visible[letter]["seen"] = True
                visible[letter]["times"].append(detection.get("time_sec"))
                visible[letter]["evidence"].append(detection.get("anchor_link"))
                visible[letter]["confidence"] = max(float(visible[letter]["confidence"]), float(detection.get("confidence") or 0.0))
        for region in ocr_regions:
            text = str(region.get("text") or "").lower()
            if region.get("scope_match", True) and any(term in text for term in terms):
                visible[letter]["seen"] = True
                visible[letter]["times"].append(region.get("time_sec"))
                visible[letter]["evidence"].append(region.get("anchor_link"))
                visible[letter]["confidence"] = max(float(visible[letter]["confidence"]), float(region.get("confidence") or 0.0))
    return {
        "target_category": "option_level_visible_set",
        "visible_options": [letter for letter, item in visible.items() if item["seen"]],
        "missing_candidates": [letter for letter, item in visible.items() if not item["seen"]],
        "options": visible,
        "rule": "Absence is determined by valid-scope aggregation; one frame where an option is unseen is not enough.",
    }


def _option_terms(text: str) -> List[str]:
    stopwords = {"the", "and", "with", "without", "style", "open", "relaxed", "fit", "shirt", "coat", "jacket"}
    terms = []
    for token in re.sub(r"[^a-zA-Z0-9]+", " ", str(text or "").lower()).split():
        if len(token) < 3 or token in stopwords:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _query_terms_for_video_evidence(question: str) -> List[str]:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "only", "option",
        "respond", "video", "which", "what", "when", "where", "does", "did",
        "doing", "show", "shown", "following", "answer", "letter", "above",
        "below", "based", "select", "best", "are", "is", "was", "were",
    }
    terms: List[str] = []
    for token in re.sub(r"[^a-zA-Z0-9]+", " ", str(question or "").lower()).split():
        if len(token) < 3 and not token.isdigit():
            continue
        if token in stopwords:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _detect_scene_segments(video_path: str, video_anchor: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        from scenedetect import AdaptiveDetector, SceneManager, VideoManager

        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector())
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        raw_scenes = scene_manager.get_scene_list()
        video_manager.release()
        segments = []
        for index, (start, end) in enumerate(raw_scenes[:DEFAULT_VIDEO_SCENE_MAX_SEGMENTS], start=1):
            start_sec = round(start.get_seconds(), 3)
            end_sec = round(end.get_seconds(), 3)
            segments.append(
                {
                    "id": f"scene_{index:04d}",
                    "time_range_sec": [start_sec, end_sec],
                    "duration_sec": round(max(0.0, end_sec - start_sec), 3),
                    "source": "pyscenedetect_adaptive",
                }
            )
        if segments:
            return segments, None
    except Exception as err:
        return _fallback_scene_segments(video_anchor), f"PySceneDetect unavailable or failed; used fallback scene segments: {type(err).__name__}: {err}"
    return _fallback_scene_segments(video_anchor), "PySceneDetect found no scene cuts; used timeline fallback."


def _fallback_scene_segments(video_anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
    duration = float(video_anchor.get("source_duration_sec") or 0.0)
    if duration <= 0:
        return []
    count = min(DEFAULT_VIDEO_SCENE_MAX_SEGMENTS, max(1, int(math.ceil(duration / 20.0))))
    segments = []
    for index in range(count):
        start = duration * index / count
        end = duration * (index + 1) / count
        segments.append(
            {
                "id": f"scene_fallback_{index + 1:04d}",
                "time_range_sec": [round(start, 3), round(end, 3)],
                "duration_sec": round(end - start, 3),
                "source": "uniform_fallback",
            }
        )
    return segments


def _detect_motion_regions(
    video_path: str,
    video_anchor: Dict[str, Any],
    max_regions: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], f"Could not open video for motion extraction: {video_path}"
    source_fps = capture.get(cv2.CAP_PROP_FPS) or float(video_anchor.get("source_fps") or 0.0) or 25.0
    step = _frame_step(source_fps, 2.0)
    previous_gray: Optional[np.ndarray] = None
    frame_index = 0
    candidates: List[Dict[str, Any]] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            small = _resize_frame(frame, 384)
            gray = np.asarray(small.convert("L"), dtype=np.uint8)
            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask, kernel, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    area = float(cv2.contourArea(contour))
                    image_area = float(mask.shape[0] * mask.shape[1])
                    if area / max(1.0, image_area) >= 0.002:
                        x, y, w, h = cv2.boundingRect(contour)
                        intensity = float(diff[mask > 0].mean()) if np.any(mask > 0) else 0.0
                        width, height = small.size
                        time_sec = frame_index / source_fps if source_fps > 0 else float(frame_index)
                        candidates.append(
                            {
                                "id": f"motion_candidate_{len(candidates) + 1:04d}",
                                "time_sec": round(time_sec, 3),
                                "frame_index": frame_index,
                                "bbox_norm": _box_to_norm((x, y, x + w, y + h), width, height),
                                "motion_area_ratio": round(area / max(1.0, image_area), 5),
                                "motion_intensity": round(intensity / 255.0, 5),
                                "score": round((area / max(1.0, image_area)) * 0.5 + (intensity / 255.0) * 0.5, 5),
                                "source": "opencv_frame_difference",
                            }
                        )
            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()
    selected = _select_diverse_time_items(candidates, max_regions, min_gap_sec=1.5)
    return selected, None


def _extract_video_ocr_regions(
    question: str,
    video_path: str,
    output_dir: Path,
    video_anchor: Dict[str, Any],
    max_regions: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    if not _question_needs_ocr_evidence(question):
        return [], [], None
    paddle_regions, paddle_anchors, paddle_warning = _extract_paddle_ocr_regions(
        question=question,
        video_path=video_path,
        output_dir=output_dir,
        video_anchor=video_anchor,
        max_regions=max_regions,
    )
    if paddle_regions:
        return paddle_regions, paddle_anchors, paddle_warning

    try:
        import pytesseract
        from pytesseract import Output
    except Exception as err:
        warnings = [item for item in (paddle_warning, f"pytesseract unavailable; skipped OCR evidence: {err}") if item]
        return [], [], "; ".join(warnings) if warnings else None

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], [], f"Could not open video for OCR extraction: {video_path}"

    ocr_dir = output_dir / "ocr_regions"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    frames = _ocr_candidate_frames(video_anchor, question, max_frames=8)
    regions: List[Dict[str, Any]] = []
    image_anchors: List[Dict[str, Any]] = []
    seen_text = set()
    try:
        for frame in frames:
            if len(regions) >= max_regions:
                break
            frame_index = int(frame.get("frame_index") or 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(image, lang="eng", output_type=Output.DICT, config="--psm 11")
            width, height = image.size
            for index, text in enumerate(data.get("text", [])):
                if len(regions) >= max_regions:
                    break
                clean = " ".join(str(text or "").split())
                if len(clean) < 2:
                    continue
                try:
                    conf = float(data.get("conf", [0])[index])
                except (TypeError, ValueError):
                    conf = 0.0
                if conf < 35:
                    continue
                key = clean.lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                left = int(data["left"][index])
                top = int(data["top"][index])
                box_width = int(data["width"][index])
                box_height = int(data["height"][index])
                box = _ensure_min_box_size(
                    _expand_box((left, top, left + box_width, top + box_height), width, height, scale=1.8),
                    width,
                    height,
                )
                crop = image.crop(box)
                crop.thumbnail((960, 960), Image.Resampling.LANCZOS)
                anchor_link = f"video_ocr_region_{len(regions) + 1:04d}"
                crop_path = ocr_dir / f"{anchor_link}.jpg"
                crop.save(crop_path, format="JPEG", quality=88, optimize=True)
                time_sec = frame.get("time_sec")
                region = {
                    "id": anchor_link,
                    "anchor_link": anchor_link,
                    "time_sec": time_sec,
                    "frame_index": frame_index,
                    "text": clean,
                    "confidence": round(conf / 100.0, 4),
                    "bbox_norm": _box_to_norm(box, width, height),
                    "crop_path": str(crop_path),
                    "source": "tesseract_ocr",
                    "score": round(conf / 100.0, 4),
                }
                regions.append(region)
                image_anchors.append(
                    {
                        "anchor_id": anchor_link,
                        "anchor_link": anchor_link,
                        "type": "video_ocr_region_crop",
                        "role": "Preserve high-resolution OCR region detected before the VLM evidence pass.",
                        "path": str(crop_path),
                        "source_video_path": str(Path(video_path)),
                        "time_sec": time_sec,
                        "frame_index": frame_index,
                        "source_resolution": {"width": width, "height": height},
                        "resolution": {"width": crop.width, "height": crop.height},
                        "bbox_norm": region["bbox_norm"],
                        "region_hint": "ocr_text_region",
                        "recognized_text": clean,
                        "ocr_confidence": region["confidence"],
                        "compression": {
                            "format": "jpeg",
                            "bytes": crop_path.stat().st_size,
                            "quality": 88,
                            "strategy": "Tesseract OCR text region crop with source timestamp.",
                        },
                    }
                )
    finally:
        capture.release()
    warnings = [item for item in (paddle_warning, None if regions else "Tesseract OCR found no confident text regions.") if item]
    return regions, image_anchors, "; ".join(warnings) if warnings else None


def _extract_paddle_ocr_regions(
    question: str,
    video_path: str,
    output_dir: Path,
    video_anchor: Dict[str, Any],
    max_regions: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    try:
        ocr = _get_paddle_ocr_model()
    except Exception as err:
        return [], [], f"PaddleOCR unavailable; will use fallback OCR: {type(err).__name__}: {err}"
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], [], f"Could not open video for PaddleOCR extraction: {video_path}"
    ocr_dir = output_dir / "paddle_ocr_regions"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    frames = _ocr_candidate_frames(video_anchor, question, max_frames=8)
    regions: List[Dict[str, Any]] = []
    image_anchors: List[Dict[str, Any]] = []
    seen_text = set()
    try:
        for frame in frames:
            if len(regions) >= max_regions:
                break
            frame_index = int(frame.get("frame_index") or 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            width, height = image.size
            raw_items = _paddle_ocr_items(ocr, np.asarray(image))
            for item in raw_items:
                if len(regions) >= max_regions:
                    break
                text = " ".join(str(item.get("text") or "").split())
                conf = float(item.get("confidence") or 0.0)
                points = item.get("points") or []
                if len(text) < 2 or conf < 0.35 or not points:
                    continue
                key = text.lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                box = _ensure_min_box_size(
                    _expand_box((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), width, height, scale=1.8),
                    width,
                    height,
                )
                crop = image.crop(box)
                crop.thumbnail((960, 960), Image.Resampling.LANCZOS)
                anchor_link = f"video_ocr_region_{len(regions) + 1:04d}"
                crop_path = ocr_dir / f"{anchor_link}.jpg"
                crop.save(crop_path, format="JPEG", quality=88, optimize=True)
                time_sec = frame.get("time_sec")
                region = {
                    "id": anchor_link,
                    "anchor_link": anchor_link,
                    "time_sec": time_sec,
                    "frame_index": frame_index,
                    "text": text,
                    "confidence": round(conf, 4),
                    "bbox_norm": _box_to_norm(box, width, height),
                    "crop_path": str(crop_path),
                    "source": "paddleocr",
                    "score": round(conf, 4),
                }
                regions.append(region)
                image_anchors.append(_ocr_region_image_anchor(anchor_link, crop_path, video_path, time_sec, frame_index, width, height, crop, region))
    finally:
        capture.release()
    return regions, image_anchors, None if regions else "PaddleOCR found no confident text regions; will use fallback OCR."


def _get_paddle_ocr_model() -> Any:
    global _PADDLE_OCR_MODEL
    if _PADDLE_OCR_MODEL is not None:
        return _PADDLE_OCR_MODEL
    from paddleocr import PaddleOCR

    try:
        _PADDLE_OCR_MODEL = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        _PADDLE_OCR_MODEL = PaddleOCR(lang="en", use_angle_cls=True)
    return _PADDLE_OCR_MODEL


def _paddle_ocr_items(ocr: Any, image_array: np.ndarray) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        result = ocr.ocr(image_array, cls=True)
    except Exception:
        try:
            result = ocr.predict(image_array)
        except Exception:
            return []
    for page in result or []:
        if isinstance(page, dict):
            texts = page.get("rec_texts") or page.get("texts") or []
            scores = page.get("rec_scores") or page.get("scores") or []
            boxes = page.get("rec_boxes") or page.get("dt_polys") or page.get("boxes") or []
            for index, text in enumerate(texts):
                score = scores[index] if index < len(scores) else 0.0
                box = boxes[index] if index < len(boxes) else []
                points = _coerce_ocr_points(box)
                items.append({"text": text, "confidence": score, "points": points})
            continue
        if isinstance(page, list):
            for entry in page:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                points = _coerce_ocr_points(entry[0])
                text = ""
                score = 0.0
                if isinstance(entry[1], (list, tuple)) and len(entry[1]) >= 2:
                    text = entry[1][0]
                    score = entry[1][1]
                items.append({"text": text, "confidence": score, "points": points})
    return items


def _coerce_ocr_points(value: Any) -> List[Tuple[float, float]]:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 4 and array.ndim == 1:
        x1, y1, x2, y2 = array.tolist()
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if array.ndim >= 2 and array.shape[-1] >= 2:
        flat = array.reshape(-1, array.shape[-1])
        return [(float(point[0]), float(point[1])) for point in flat[:4]]
    return []


def _ocr_region_image_anchor(
    anchor_link: str,
    crop_path: Path,
    video_path: str,
    time_sec: Any,
    frame_index: int,
    width: int,
    height: int,
    crop: Image.Image,
    region: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "anchor_id": anchor_link,
        "anchor_link": anchor_link,
        "type": "video_ocr_region_crop",
        "role": "Preserve high-resolution OCR region detected before the VLM evidence pass.",
        "path": str(crop_path),
        "source_video_path": str(Path(video_path)),
        "time_sec": time_sec,
        "frame_index": frame_index,
        "source_resolution": {"width": width, "height": height},
        "resolution": {"width": crop.width, "height": crop.height},
        "bbox_norm": region["bbox_norm"],
        "region_hint": "ocr_text_region",
        "recognized_text": region.get("text"),
        "ocr_confidence": region.get("confidence"),
        "compression": {
            "format": "jpeg",
            "bytes": crop_path.stat().st_size,
            "quality": 88,
            "strategy": "OCR text region crop with source timestamp.",
        },
    }


def _question_needs_ocr_evidence(question: str) -> bool:
    text = (question or "").lower()
    keywords = (
        "text", "word", "read", "sign", "label", "title", "screen", "score",
        "number", "digit", "written", "display", "shown on", "board", "牌子",
        "标题", "写", "比分",
    )
    return any(keyword in text for keyword in keywords)


def _extract_video_object_detections(
    question: str,
    video_path: str,
    output_dir: Path,
    video_anchor: Dict[str, Any],
    max_detections: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    if not _question_needs_object_detection(question):
        return [], [], None
    try:
        model = _get_yolo_model()
    except Exception as err:
        return [], [], f"Ultralytics YOLO unavailable; skipped object evidence: {type(err).__name__}: {err}"
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], [], f"Could not open video for YOLO object detection: {video_path}"

    object_dir = output_dir / "object_regions"
    object_dir.mkdir(parents=True, exist_ok=True)
    frames = _object_candidate_frames(video_anchor, question, max_frames=10)
    detections: List[Dict[str, Any]] = []
    image_anchors: List[Dict[str, Any]] = []
    try:
        for frame in frames:
            if len(detections) >= max_detections:
                break
            frame_index = int(frame.get("frame_index") or 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            width, height = image.size
            results = model.predict(source=np.asarray(image), verbose=False, imgsz=960, conf=0.20)
            for result in results:
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
                if boxes is None:
                    continue
                for box_index in range(len(boxes)):
                    if len(detections) >= max_detections:
                        break
                    xyxy = boxes.xyxy[box_index].detach().cpu().numpy().tolist()
                    conf = float(boxes.conf[box_index].detach().cpu().item())
                    cls_id = int(boxes.cls[box_index].detach().cpu().item())
                    label = str(names.get(cls_id, cls_id))
                    if not _object_label_relevant(label, question) and conf < 0.45:
                        continue
                    raw_box = tuple(int(round(value)) for value in xyxy)
                    expanded = _ensure_min_box_size(
                        _expand_box(raw_box, width, height, scale=1.35),
                        width,
                        height,
                    )
                    crop = image.crop(expanded)
                    crop.thumbnail((960, 960), Image.Resampling.LANCZOS)
                    anchor_link = f"video_object_region_{len(detections) + 1:04d}"
                    crop_path = object_dir / f"{anchor_link}.jpg"
                    crop.save(crop_path, format="JPEG", quality=86, optimize=True)
                    detection = {
                        "id": anchor_link,
                        "anchor_link": anchor_link,
                        "time_sec": frame.get("time_sec"),
                        "frame_index": frame_index,
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox_norm": _box_to_norm(expanded, width, height),
                        "source": "ultralytics_yolo",
                        "score": round(conf + (0.15 if _object_label_relevant(label, question) else 0.0), 4),
                    }
                    detections.append(detection)
                    image_anchors.append(
                        {
                            "anchor_id": anchor_link,
                            "anchor_link": anchor_link,
                            "type": "video_object_region_crop",
                            "role": "Preserve object detection crop from the task-aware evidence extractor.",
                            "path": str(crop_path),
                            "source_video_path": str(Path(video_path)),
                            "time_sec": frame.get("time_sec"),
                            "frame_index": frame_index,
                            "source_resolution": {"width": width, "height": height},
                            "resolution": {"width": crop.width, "height": crop.height},
                            "bbox_norm": detection["bbox_norm"],
                            "region_hint": f"object_{label}",
                            "detected_label": label,
                            "detection_confidence": detection["confidence"],
                            "compression": {
                                "format": "jpeg",
                                "bytes": crop_path.stat().st_size,
                                "quality": 86,
                                "strategy": "YOLO object region crop with timestamp and label.",
                            },
                        }
                    )
    finally:
        capture.release()
    return _select_diverse_object_detections(detections, max_detections), image_anchors[:max_detections], None if detections else "YOLO found no relevant object detections."


def _get_yolo_model() -> Any:
    from ultralytics import YOLO

    last_error: Optional[Exception] = None
    for model_name in ("yolo11s.pt", "yolo11n.pt", "yolo11x.pt"):
        if model_name in _YOLO_MODEL_CACHE:
            return _YOLO_MODEL_CACHE[model_name]
        try:
            model = YOLO(model_name)
            _YOLO_MODEL_CACHE[model_name] = model
            return model
        except Exception as err:
            last_error = err
    raise RuntimeError(f"Could not load any YOLO model for object evidence: {last_error}")


def _question_needs_object_detection(question: str) -> bool:
    text = (question or "").lower()
    keywords = (
        "how many", "number of", "count", "many", "men", "women", "person",
        "people", "player", "ball", "car", "vehicle", "dog", "cat", "cup",
        "bottle", "box", "table", "chair", "left", "right", "where", "visible",
        "absent", "missing", "challenger", "stage",
    )
    return any(keyword in text for keyword in keywords)


def _object_label_relevant(label: str, question: str) -> bool:
    text = (question or "").lower()
    label_text = str(label or "").lower()
    aliases = {
        "person": ("person", "people", "men", "women", "player", "challenger", "dancer", "singer", "protagonist"),
        "sports ball": ("ball", "score", "game"),
        "bottle": ("bottle",),
        "cup": ("cup",),
        "car": ("car", "vehicle"),
        "dog": ("dog",),
        "cat": ("cat",),
        "chair": ("chair",),
        "dining table": ("table",),
        "tv": ("screen", "display"),
    }
    terms = aliases.get(label_text, (label_text,))
    return any(term in text for term in terms)


def _object_candidate_frames(video_anchor: Dict[str, Any], question: str, max_frames: int) -> List[Dict[str, Any]]:
    frames = [dict(frame) for frame in video_anchor.get("frames", []) if frame.get("frame_index") is not None]
    if not frames:
        return []
    selected: List[Dict[str, Any]] = []
    for window in _question_priority_time_windows(question, video_anchor):
        start, end = _coerce_time_window(window)
        for point in (start, (start + end) / 2.0, end):
            nearest = _nearest_video_anchor_frame(frames, point)
            if nearest:
                _append_unique_frame(selected, nearest, max_frames)
    if len(selected) < max_frames:
        if max_frames == 1:
            indices = [0]
        else:
            indices = sorted({round(i * (len(frames) - 1) / max(1, max_frames - 1)) for i in range(max_frames)})
        for index in indices:
            _append_unique_frame(selected, frames[int(index)], max_frames)
            if len(selected) >= max_frames:
                break
    return sorted(selected, key=lambda item: float(item.get("time_sec") or 0.0))


def _select_diverse_object_detections(detections: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in sorted(detections, key=lambda value: float(value.get("score") or 0.0), reverse=True):
        if any(
            existing.get("label") == item.get("label")
            and abs(float(existing.get("time_sec") or 0.0) - float(item.get("time_sec") or 0.0)) < 1.0
            for existing in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= max_count:
            break
    return sorted(selected, key=lambda value: float(value.get("time_sec") or 0.0))


def _ocr_candidate_frames(video_anchor: Dict[str, Any], question: str, max_frames: int) -> List[Dict[str, Any]]:
    frames = [dict(frame) for frame in video_anchor.get("frames", []) if frame.get("frame_index") is not None]
    if not frames:
        return []
    priority = _question_priority_time_windows(question, video_anchor)
    selected: List[Dict[str, Any]] = []
    for window in priority:
        start, end = _coerce_time_window(window)
        midpoint = (start + end) / 2.0
        nearest = _nearest_video_anchor_frame(frames, midpoint)
        if nearest:
            _append_unique_frame(selected, nearest, max_frames)
    for frame in sorted(frames, key=lambda item: float(item.get("change_score") or 0.0), reverse=True):
        _append_unique_frame(selected, frame, max_frames)
        if len(selected) >= max_frames:
            break
    return sorted(selected, key=lambda item: float(item.get("time_sec") or 0.0))[:max_frames]


def _expand_box(box: Tuple[int, int, int, int], width: int, height: int, scale: float) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    half_w = max(1.0, (right - left) * scale / 2.0)
    half_h = max(1.0, (bottom - top) * scale / 2.0)
    return (
        max(0, int(round(center_x - half_w))),
        max(0, int(round(center_y - half_h))),
        min(width, int(round(center_x + half_w))),
        min(height, int(round(center_y + half_h))),
    )


def _ensure_min_box_size(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    min_size: int = 32,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    target_w = min(max(min_size, right - left), width)
    target_h = min(max(min_size, bottom - top), height)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    left = int(round(center_x - target_w / 2.0))
    top = int(round(center_y - target_h / 2.0))
    left = max(0, min(max(0, width - target_w), left))
    top = max(0, min(max(0, height - target_h), top))
    right = min(width, left + target_w)
    bottom = min(height, top + target_h)
    return left, top, right, bottom


def _select_diverse_time_items(items: List[Dict[str, Any]], max_count: int, min_gap_sec: float) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda value: float(value.get("score") or 0.0), reverse=True):
        time_sec = float(item.get("time_sec") or 0.0)
        if any(abs(float(existing.get("time_sec") or 0.0) - time_sec) < min_gap_sec for existing in selected):
            continue
        selected.append(item)
        if len(selected) >= max_count:
            break
    return sorted(selected, key=lambda value: float(value.get("time_sec") or 0.0))


def _build_video_evidence_units(
    query_profile: Dict[str, Any],
    temporal_scope: Dict[str, Any],
    scene_segments: List[Dict[str, Any]],
    motion_regions: List[Dict[str, Any]],
    ocr_regions: List[Dict[str, Any]],
    object_detections: List[Dict[str, Any]],
    question_time_windows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    question_types = set(query_profile.get("question_types") or [])
    for index, scene in enumerate(scene_segments[:6], start=1):
        units.append(
            {
                "id": f"E_scene_{index:04d}",
                "evidence_type": "global_scene_segment",
                "time_range_sec": scene.get("time_range_sec"),
                "event": "Scene segment for global temporal structure and context.",
                "confidence": 0.65,
                "source": scene.get("source"),
                "scope_match": _scope_match(scene.get("time_range_sec"), temporal_scope),
            }
        )
    for index, region in enumerate(motion_regions[:DEFAULT_VIDEO_MOTION_MAX_REGIONS], start=1):
        confidence = min(0.95, 0.45 + float(region.get("score") or 0.0) * 4.0)
        units.append(
            {
                "id": f"E_motion_{index:04d}",
                "evidence_type": "motion_saliency",
                "time_sec": region.get("time_sec"),
                "region": {"bbox_norm": region.get("bbox_norm")},
                "event": "Frame-difference motion region; inspect nearby before/during/after frames for actions, entries, exits, and state changes.",
                "confidence": round(confidence, 4),
                "source": region.get("source"),
                "scope_match": _scope_match(region.get("time_sec"), temporal_scope),
                "score_terms": {
                    "motion_intensity": region.get("motion_intensity"),
                    "motion_area_ratio": region.get("motion_area_ratio"),
                },
            }
        )
    for index, region in enumerate(ocr_regions[:DEFAULT_VIDEO_OCR_MAX_REGIONS], start=1):
        units.append(
            {
                "id": f"E_ocr_{index:04d}",
                "evidence_type": "ocr_text_region",
                "time_sec": region.get("time_sec"),
                "region": {"bbox_norm": region.get("bbox_norm")},
                "event": f"OCR text candidate: {region.get('text')}",
                "anchor_link": region.get("anchor_link"),
                "confidence": region.get("confidence"),
                "source": region.get("source"),
                "scope_match": _scope_match(region.get("time_sec"), temporal_scope),
            }
        )
    for index, detection in enumerate(object_detections[:DEFAULT_VIDEO_OBJECT_MAX_DETECTIONS], start=1):
        units.append(
            {
                "id": f"E_object_{index:04d}",
                "evidence_type": "object_detection",
                "time_sec": detection.get("time_sec"),
                "region": {"bbox_norm": detection.get("bbox_norm")},
                "event": f"Detected object candidate: {detection.get('label')}",
                "anchor_link": detection.get("anchor_link"),
                "confidence": detection.get("confidence"),
                "source": detection.get("source"),
                "scope_match": _scope_match(detection.get("time_sec"), temporal_scope),
            }
        )
    for index, window in enumerate(question_time_windows[:8], start=1):
        units.append(
            {
                "id": f"E_query_window_{index:04d}",
                "evidence_type": "query_relevant_time_window",
                "time_range_sec": window.get("time_range_sec"),
                "event": "Question-prioritized or ASR-retrieved time window; keep nearby visual anchors.",
                "confidence": 0.8 if window.get("source") == "question_relevant_transcript" else 0.7,
                "source": window.get("source"),
                "scope_match": _scope_match(window.get("time_range_sec"), temporal_scope),
            }
        )
    if "count" in question_types:
        units.append(
            {
                "id": "E_count_guard",
                "evidence_type": "counting_guard",
                "event": "For count questions, compare early/middle/late evidence and avoid counting one frame as the total video count.",
                "confidence": 0.75,
                "source": "query_policy",
            }
        )
    return units[:36]


def create_video_tubelet_storyboard_anchors(
    question: str,
    video_path: str,
    output_dir: Path,
    video_anchor: Dict[str, Any],
    max_storyboards: int = DEFAULT_VIDEO_TUBELET_MAX_STORYBOARDS,
    max_side: int = DEFAULT_VIDEO_TUBELET_MAX_SIDE,
    jpeg_quality: int = 84,
    anchor_id: str = "video_tubelet_storyboard",
    question_time_windows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if max_storyboards <= 0:
        return []

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    output_path = output_dir / anchor_id
    output_path.mkdir(parents=True, exist_ok=True)
    source_fps = float(video_anchor.get("source_fps") or 0.0)
    source_frame_count = int(video_anchor.get("source_frame_count") or 0)
    source_duration = float(video_anchor.get("source_duration_sec") or 0.0)
    source_width = int(video_anchor.get("source_resolution", {}).get("width") or 0)
    source_height = int(video_anchor.get("source_resolution", {}).get("height") or 0)
    crop_specs = _video_tubelet_crop_specs(source_width, source_height, question)
    priority_windows = _question_priority_time_windows(question, video_anchor)
    windows = _select_tubelet_windows(
        video_anchor,
        question_time_windows=priority_windows + (question_time_windows or []),
        max_count=max_storyboards,
    )
    if not windows:
        capture.release()
        return []

    anchors: List[Dict[str, Any]] = []
    frame_cache: Dict[int, Image.Image] = {}
    try:
        for window_index, window in enumerate(windows):
            if len(anchors) >= max_storyboards:
                break
            spec = crop_specs[min(window_index, len(crop_specs) - 1)]
            start, end = _coerce_time_window(window)
            if end < start:
                start, end = end, start
            points = _tubelet_time_points(start, end, source_duration)
            storyboard_frames: List[Dict[str, Any]] = []
            for role, time_sec in points:
                frame_index = _time_to_frame_index(time_sec, source_fps, source_frame_count)
                if frame_index not in frame_cache:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame_bgr = capture.read()
                    if not ok:
                        continue
                    frame_cache[frame_index] = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                image = frame_cache[frame_index]
                width, height = image.size
                box = _norm_box(width, height, *spec["norm"])
                crop = image.crop(box)
                crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                storyboard_frames.append(
                    {
                        "role": role,
                        "time_sec": round(time_sec, 3),
                        "frame_index": frame_index,
                        "image": crop.copy(),
                        "bbox_norm": _box_to_norm(box, width, height),
                    }
                )
            if len(storyboard_frames) < 2:
                continue
            output_index = len(anchors) + 1
            storyboard_anchor_id = f"{anchor_id}_{output_index:04d}"
            storyboard_path = output_path / f"{storyboard_anchor_id}.jpg"
            board = _compose_tubelet_storyboard(storyboard_frames, storyboard_anchor_id)
            board.save(storyboard_path, format="JPEG", quality=jpeg_quality, optimize=True)
            anchors.append(
                {
                    "anchor_id": storyboard_anchor_id,
                    "anchor_link": storyboard_anchor_id,
                    "type": "video_tubelet_storyboard",
                    "role": "Preserve a continuous before/during/after visual tubelet so the model can reason about action order and state change rather than isolated keyframes.",
                    "path": str(storyboard_path),
                    "source_video_path": str(Path(video_path)),
                    "time_range_sec": [round(start, 3), round(end, 3)],
                    "source_resolution": {"width": source_width, "height": source_height},
                    "resolution": {"width": board.width, "height": board.height},
                    "region_hint": spec["label"],
                    "bbox_norm": storyboard_frames[0].get("bbox_norm"),
                    "frames": [
                        {
                            "role": frame["role"],
                            "time_sec": frame["time_sec"],
                            "frame_index": frame["frame_index"],
                            "bbox_norm": frame["bbox_norm"],
                        }
                        for frame in storyboard_frames
                    ],
                    "linked_video_anchor": window.get("anchor_link"),
                    "question_relevant": window.get("source") == "question_relevant_transcript",
                    "selection_reason": window.get("source") or "event_or_temporal_coverage",
                    "compression": {
                        "format": "jpeg",
                        "bytes": storyboard_path.stat().st_size,
                        "max_side": max_side,
                        "quality": jpeg_quality,
                        "strategy": "Question-conditioned continuous tubelet storyboard with before/during/after frames.",
                    },
                }
            )
    finally:
        capture.release()
    return anchors


def create_video_detail_frame_anchors(
    question: str,
    video_path: str,
    output_dir: Path,
    video_anchor: Dict[str, Any],
    max_crops: int = DEFAULT_VIDEO_DETAIL_MAX_CROPS,
    max_side: int = DEFAULT_VIDEO_DETAIL_MAX_SIDE,
    jpeg_quality: int = 86,
    anchor_id: str = "video_keyframe_detail",
    question_time_windows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    query_windows = _question_priority_time_windows(question, video_anchor) + (question_time_windows or [])
    if max_crops <= 0 or (not _needs_video_detail(question) and not query_windows):
        return []

    max_detail_frames = max(1, max_crops if query_windows else math.ceil(max_crops / 2))
    frames = _select_video_detail_frames(
        video_anchor,
        max_count=max_detail_frames,
        question_time_windows=query_windows,
    )
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

    frame_cache: Dict[int, Tuple[Image.Image, int, int]] = {}
    try:
        for spec in crop_specs:
            if len(anchors) >= max_crops:
                break
            for frame in frames:
                if len(anchors) >= max_crops:
                    break
                frame_index = int(frame.get("frame_index") or 0)
                if frame_index not in frame_cache:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame_bgr = capture.read()
                    if not ok:
                        continue
                    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    width, height = image.size
                    frame_cache[frame_index] = (image, width, height)
                image, width, height = frame_cache[frame_index]
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
                        "question_relevant": bool(frame.get("question_relevant")),
                        "query_window": frame.get("query_window"),
                        "selection_reason": frame.get("selection_reason"),
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
        "how many", "number of", "count", "many", "score", "scoreboard", "points",
        "challenger", "challengers", "men", "women", "absent", "missing", "stored",
        "box", "beginning", "start", "current", "ongoing game",
    )
    return any(keyword in text for keyword in detail_keywords)


def _select_video_detail_frames(
    video_anchor: Dict[str, Any],
    max_count: int,
    question_time_windows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    frames = [frame for frame in video_anchor.get("frames", []) if frame.get("frame_index") is not None]
    selected: List[Dict[str, Any]] = []
    source_fps = float(video_anchor.get("source_fps") or 0.0)
    source_frame_count = int(video_anchor.get("source_frame_count") or 0)
    source_duration = float(video_anchor.get("source_duration_sec") or 0.0)

    for window in question_time_windows or []:
        if len(selected) >= max_count:
            break
        start, end = _coerce_time_window(window)
        if end < start:
            start, end = end, start
        midpoint = (start + end) / 2.0
        points = [midpoint]
        if max_count - len(selected) >= 3 and end - start > 6.0:
            points = [start, midpoint, end]
        for point in points:
            if len(selected) >= max_count:
                break
            time_sec = _clamp_float(point, 0.0, source_duration) if source_duration > 0 else max(0.0, point)
            if source_fps > 0:
                frame_index = int(round(time_sec * source_fps))
                if source_frame_count > 0:
                    frame_index = max(0, min(source_frame_count - 1, frame_index))
            else:
                frame_index = int(round(time_sec))
            nearest = _nearest_video_anchor_frame(frames, time_sec)
            candidate = {
                "frame_index": frame_index,
                "time_sec": round(time_sec, 3),
                "anchor_link": nearest.get("anchor_link") if nearest else None,
                "change_score": nearest.get("change_score") if nearest else None,
                "question_relevant": True,
                "query_window": window,
                "selection_reason": "question_relevant_transcript_window",
            }
            _append_unique_frame(selected, candidate, max_count)

    for boundary in video_anchor.get("event_boundaries", []):
        link = boundary.get("anchor_link")
        match = next((frame for frame in frames if frame.get("anchor_link") == link), None)
        if match:
            candidate = dict(match)
            candidate.setdefault("selection_reason", "event_boundary")
            _append_unique_frame(selected, candidate, max_count)
        if len(selected) >= max_count:
            return sorted(selected, key=lambda item: float(item.get("time_sec") or 0.0))

    ranked = sorted(frames, key=lambda frame: float(frame.get("change_score") or 0.0), reverse=True)
    for frame in ranked:
        candidate = dict(frame)
        candidate.setdefault("selection_reason", "high_change_frame")
        _append_unique_frame(selected, candidate, max_count)
        if len(selected) >= max_count:
            return sorted(selected, key=lambda item: float(item.get("time_sec") or 0.0))
    return sorted(selected, key=lambda item: float(item.get("time_sec") or 0.0))


def _coerce_time_window(window: Any) -> Tuple[float, float]:
    if isinstance(window, dict):
        value = window.get("time_range_sec") or window.get("window_sec") or window.get("time")
    else:
        value = window
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _safe_float_value(value[0]), _safe_float_value(value[1])
    value_float = _safe_float_value(value)
    return value_float, value_float


def _question_priority_time_windows(
    question: str,
    video_anchor: Dict[str, Any],
    temporal_scope: Optional[Dict[str, Any]] = None,
    scene_segments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    text = (question or "").lower()
    duration = float(video_anchor.get("source_duration_sec") or 0.0)
    if duration <= 0:
        return []
    windows: List[Dict[str, Any]] = []

    def add(start: float, end: float, source: str) -> None:
        start = _clamp_float(start, 0.0, duration)
        end = _clamp_float(max(end, start + 0.5), 0.0, duration)
        if end <= start:
            return
        candidate = {
            "time_range_sec": [round(start, 3), round(end, 3)],
            "source": source,
        }
        if candidate["time_range_sec"] not in [item.get("time_range_sec") for item in windows]:
            windows.append(candidate)

    if temporal_scope and temporal_scope.get("type") not in {None, "full_video"}:
        start, end = _coerce_time_window(temporal_scope)
        add(start, end, f"temporal_scope_{temporal_scope.get('type')}")
    if any(term in text for term in ("beginning", "at the start", "start of", "opening", "initially", "displayed at the beginning")):
        add(0.0, min(10.0, duration), "question_start_focus")
    if any(term in text for term in ("current score", "ongoing game", "scoreboard", "score of", "final score")) and not temporal_scope:
        add(0.0, duration, "question_score_full_timeline_focus")
    if any(term in text for term in ("how many", "number of", "count", "many", "challenger", "challengers", "men", "women")):
        add(0.0, min(duration, max(4.0, duration / 3.0)), "question_counting_early_coverage")
        add(max(0.0, duration / 3.0 - 2.0), min(duration, 2.0 * duration / 3.0 + 2.0), "question_counting_middle_coverage")
        add(max(0.0, 2.0 * duration / 3.0 - 2.0), duration, "question_counting_late_coverage")
    return windows[:6]


def _nearest_video_anchor_frame(frames: List[Dict[str, Any]], time_sec: float) -> Optional[Dict[str, Any]]:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(float(frame.get("time_sec") or 0.0) - time_sec))


def _append_unique_frame(selected: List[Dict[str, Any]], candidate: Dict[str, Any], max_count: int) -> None:
    frame_index = candidate.get("frame_index")
    if frame_index is None:
        return
    if any(item.get("frame_index") == frame_index for item in selected):
        return
    if len(selected) < max_count:
        selected.append(candidate)


def _clamp_float(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return max(lower, value)
    return max(lower, min(upper, value))


def _safe_float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def _video_tubelet_crop_specs(width: int, height: int, question: str) -> List[Dict[str, Any]]:
    specs = [{"label": "tubelet_full_context", "norm": (0.0, 0.0, 1.0, 1.0)}]
    text = (question or "").lower()
    if any(word in text for word in ("screen", "text", "read", "number", "digit", "caption", "sign")):
        specs.extend(
            [
                {"label": "tubelet_center_screen_detail", "norm": (0.10, 0.08, 0.90, 0.92)},
                {"label": "tubelet_lower_text_band", "norm": (0.05, 0.52, 0.95, 1.0)},
            ]
        )
    elif any(word in text for word in ("left", "right", "where", "position", "side")):
        specs.extend(
            [
                {"label": "tubelet_left_context", "norm": (0.0, 0.05, 0.60, 0.95)},
                {"label": "tubelet_right_context", "norm": (0.40, 0.05, 1.0, 0.95)},
            ]
        )
    else:
        specs.extend(
            [
                {"label": "tubelet_center_action", "norm": (0.12, 0.10, 0.88, 0.90)},
                {"label": "tubelet_lower_action", "norm": (0.05, 0.45, 0.95, 1.0)},
            ]
        )
    return _dedupe_video_specs(specs)


def _select_tubelet_windows(
    video_anchor: Dict[str, Any],
    question_time_windows: Optional[List[Dict[str, Any]]],
    max_count: int,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    duration = float(video_anchor.get("source_duration_sec") or 0.0)
    for window in question_time_windows or []:
        start, end = _coerce_time_window(window)
        if end <= start:
            end = start + 2.0
        item = dict(window)
        item["time_range_sec"] = [max(0.0, start), min(duration, end) if duration > 0 else end]
        item.setdefault("source", "question_relevant_transcript")
        windows.append(item)
        if len(windows) >= max_count:
            return windows

    for boundary in video_anchor.get("event_boundaries", []):
        time_sec = float(boundary.get("time_sec") or 0.0)
        item = dict(boundary)
        item["time_range_sec"] = [
            max(0.0, time_sec - 1.5),
            min(duration, time_sec + 1.5) if duration > 0 else time_sec + 1.5,
        ]
        item["source"] = "visual_change_boundary"
        windows.append(item)
        if len(windows) >= max_count:
            return windows

    frames = video_anchor.get("frames") or []
    if frames:
        if len(frames) <= max_count:
            selected_frames = frames
        else:
            selected_indices = sorted({round(i * (len(frames) - 1) / max(1, max_count - 1)) for i in range(max_count)})
            selected_frames = [frames[int(index)] for index in selected_indices]
        for frame in selected_frames:
            time_sec = float(frame.get("time_sec") or 0.0)
            windows.append(
                {
                    "time_range_sec": [
                        max(0.0, time_sec - 1.5),
                        min(duration, time_sec + 1.5) if duration > 0 else time_sec + 1.5,
                    ],
                    "anchor_link": frame.get("anchor_link"),
                    "source": "temporal_coverage",
                }
            )
            if len(windows) >= max_count:
                break
    return windows[:max_count]


def _tubelet_time_points(start: float, end: float, duration: float) -> List[Tuple[str, float]]:
    if duration > 0:
        start = _clamp_float(start, 0.0, duration)
        end = _clamp_float(end, 0.0, duration)
    if end <= start:
        end = start + 2.0
    midpoint = (start + end) / 2.0
    return [
        ("before", start),
        ("during", midpoint),
        ("after", end),
    ]


def _time_to_frame_index(time_sec: float, source_fps: float, source_frame_count: int) -> int:
    if source_fps <= 0:
        frame_index = int(round(max(0.0, time_sec)))
    else:
        frame_index = int(round(max(0.0, time_sec) * source_fps))
    if source_frame_count > 0:
        frame_index = max(0, min(source_frame_count - 1, frame_index))
    return frame_index


def _compose_tubelet_storyboard(frames: List[Dict[str, Any]], anchor_id: str) -> Image.Image:
    images = [frame["image"].convert("RGB") for frame in frames]
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    label_height = 42
    gap = 8
    board_width = len(images) * max_width + (len(images) - 1) * gap
    board_height = max_height + label_height
    board = Image.new("RGB", (board_width, board_height), "white")
    draw = ImageDraw.Draw(board)
    for index, (image, frame) in enumerate(zip(images, frames)):
        x = index * (max_width + gap) + (max_width - image.width) // 2
        y = label_height
        board.paste(image, (x, y))
        label = f"{frame.get('role')} t={frame.get('time_sec')}s"
        draw.text((index * (max_width + gap) + 6, 8), label, fill=(0, 0, 0))
    draw.text((6, board_height - 16), anchor_id, fill=(70, 70, 70))
    return board

def create_video_transcript_anchor(
    video_path: str,
    output_dir: str,
    backend: str = "auto",
    model_name: str = DEFAULT_VIDEO_ASR_MODEL,
    language: Optional[str] = "en",
    max_segments: int = DEFAULT_VIDEO_TRANSCRIPT_MAX_SEGMENTS,
    anchor_id: str = "video_transcript_anchor",
    external_subtitle_path: Optional[str] = None,
    question: str = "",
    query_retrieval: bool = True,
    query_max_segments: int = DEFAULT_VIDEO_QUERY_MAX_SEGMENTS,
    query_window_padding_sec: float = DEFAULT_VIDEO_QUERY_WINDOW_PADDING_SEC,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    segments: List[Dict[str, Any]] = []
    source = "none"

    if backend in {"auto", "subtitle"} and external_subtitle_path:
        external_path = Path(external_subtitle_path)
        if external_path.exists() and external_path.stat().st_size > 0:
            subtitle_segments = _parse_srt(external_path.read_text(encoding="utf-8", errors="ignore"), anchor_id, source="external_subtitle")
            if subtitle_segments:
                segments = subtitle_segments
                source = "external_subtitle"
            else:
                warnings.append(f"External subtitle file had no parseable segments: {external_path}")
        else:
            warnings.append(f"External subtitle file missing or empty: {external_path}")

    if not segments and backend in {"auto", "subtitle"}:
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

    all_segments = segments
    question_relevant_segments: List[Dict[str, Any]] = []
    question_relevant_time_windows: List[Dict[str, Any]] = []
    query_terms: List[str] = []
    if query_retrieval and all_segments:
        question_relevant_segments, question_relevant_time_windows, query_terms = _select_question_relevant_transcript_segments(
            all_segments,
            question,
            max_segments=max(0, query_max_segments),
            padding_sec=query_window_padding_sec,
        )
    if query_retrieval and all_segments and not question_relevant_segments:
        warnings.append("Question-driven transcript retrieval found no direct lexical hits; using temporal coverage segments.")
    stored_segments = _build_transcript_segments_for_prompt(
        all_segments,
        max_segments=max_segments,
        question_relevant_segments=question_relevant_segments,
    )
    text = " ".join(segment.get("text", "") for segment in all_segments).strip()
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
        "external_subtitle_path": str(external_subtitle_path) if external_subtitle_path else None,
        "language": language,
        "segments": stored_segments,
        "all_segment_count": len(all_segments),
        "query_retrieval": bool(query_retrieval),
        "query_terms": query_terms,
        "question_relevant_segments": question_relevant_segments,
        "question_relevant_time_windows": question_relevant_time_windows,
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



def _clean_caption_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\\[^}]+\}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _parse_srt(text: str, anchor_id: str, source: str = "embedded_subtitle") -> List[Dict[str, Any]]:
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
        caption = _clean_caption_text(" ".join(line for line in lines if "-->" not in line and not line.isdigit()))
        if not caption:
            continue
        segments.append(
            {
                "anchor_id": f"{anchor_id}_seg_{index:04d}",
                "anchor_link": f"{anchor_id}_seg_{index:04d}",
                "sequence_index": index,
                "time_range_sec": [_srt_time_to_seconds(start_text), _srt_time_to_seconds(end_text)],
                "text": caption,
                "source": source,
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


_QUERY_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "only", "option", "letter",
    "respond", "video", "which", "what", "when", "where", "why", "how", "does", "did",
    "doing", "show", "shows", "shown", "following", "about", "into", "onto", "they", "them",
    "their", "while", "there", "were", "was", "are", "is", "not", "none", "more", "most",
    "than", "previously", "believed", "specific", "evidence", "claim", "answer", "choose",
}


def _query_terms(question: str) -> List[str]:
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", str(question or "").lower())
    terms: List[str] = []
    for token in normalized.split():
        if len(token) < 3 and not token.isdigit():
            continue
        if token in _QUERY_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:80]


def _select_question_relevant_transcript_segments(
    segments: List[Dict[str, Any]],
    question: str,
    max_segments: int,
    padding_sec: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    if max_segments <= 0:
        return [], [], []
    terms = _query_terms(question)
    if not terms:
        return [], [], []
    scored: List[Tuple[float, int, List[str]]] = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "").lower()
        if not text:
            continue
        hits = [term for term in terms if term in text]
        if not hits:
            continue
        score = float(len(hits))
        score += sum(1.0 for term in hits if len(term) >= 6)
        score += min(2.0, len(set(hits)) / 3.0)
        scored.append((score, index, hits[:12]))
    if not scored:
        return [], [], terms
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected_indices: List[int] = []
    score_by_index = {index: score for score, index, _ in scored}
    hits_by_index = {index: hits for _, index, hits in scored}
    for _, index, _ in scored:
        for candidate_index in (index - 1, index, index + 1):
            if candidate_index < 0 or candidate_index >= len(segments):
                continue
            if candidate_index in selected_indices:
                continue
            selected_indices.append(candidate_index)
            if len(selected_indices) >= max_segments:
                break
        if len(selected_indices) >= max_segments:
            break
    selected: List[Dict[str, Any]] = []
    for rank, index in enumerate(selected_indices, start=1):
        item = dict(segments[index])
        item["selection_role"] = "question_relevant_transcript"
        item["relevance_rank"] = rank
        item["relevance_score"] = round(score_by_index.get(index, 0.0), 3)
        item["relevance_terms"] = hits_by_index.get(index, [])
        selected.append(item)
    windows = _merge_time_windows(selected, padding_sec=padding_sec)
    return selected, windows, terms


def _merge_time_windows(segments: List[Dict[str, Any]], padding_sec: float) -> List[Dict[str, Any]]:
    ranges: List[Tuple[float, float, List[str]]] = []
    for segment in segments:
        value = segment.get("time_range_sec") or []
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        start = max(0.0, _safe_float_value(value[0]) - padding_sec)
        end = max(start, _safe_float_value(value[1]) + padding_sec)
        ranges.append((start, end, [str(segment.get("anchor_link") or "")]))
    if not ranges:
        return []
    ranges.sort(key=lambda item: item[0])
    merged: List[Tuple[float, float, List[str]]] = []
    for start, end, links in ranges:
        if not merged or start > merged[-1][1] + 1.0:
            merged.append((start, end, links))
        else:
            old_start, old_end, old_links = merged[-1]
            merged[-1] = (old_start, max(old_end, end), old_links + links)
    return [
        {
            "time_range_sec": [round(start, 3), round(end, 3)],
            "anchor_links": [link for link in links if link],
            "source": "question_relevant_transcript",
        }
        for start, end, links in merged[:12]
    ]


def _build_transcript_segments_for_prompt(
    segments: List[Dict[str, Any]],
    max_segments: int,
    question_relevant_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if max_segments <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    seen = set()
    for segment in question_relevant_segments:
        key = segment.get("anchor_link") or segment.get("anchor_id")
        if key in seen:
            continue
        seen.add(key)
        selected.append(segment)
        if len(selected) >= max_segments:
            return selected
    coverage = _select_transcript_coverage_segments(segments, max_segments - len(selected), seen)
    selected.extend(coverage)
    return selected[:max_segments]


def _select_transcript_coverage_segments(
    segments: List[Dict[str, Any]],
    max_count: int,
    seen: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if max_count <= 0 or not segments:
        return []
    seen = seen or set()
    available = [segment for segment in segments if (segment.get("anchor_link") or segment.get("anchor_id")) not in seen]
    if len(available) <= max_count:
        return [dict(segment) for segment in available]
    if max_count == 1:
        indices = [0]
    else:
        indices = sorted({round(i * (len(available) - 1) / (max_count - 1)) for i in range(max_count)})
    coverage: List[Dict[str, Any]] = []
    for index in indices:
        item = dict(available[int(index)])
        item.setdefault("selection_role", "temporal_coverage_transcript")
        coverage.append(item)
    return coverage[:max_count]


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
        segments: List[Dict[str, Any]] = []
        detected = None
        attempts = (
            {"vad_filter": True, "beam_size": 1, "source": "faster_whisper_asr"},
            {"vad_filter": False, "beam_size": 5, "source": "faster_whisper_asr_no_vad_retry"},
        )
        for attempt in attempts:
            raw_segments, info = model.transcribe(
                str(wav_path),
                language=language,
                beam_size=attempt["beam_size"],
                vad_filter=attempt["vad_filter"],
                word_timestamps=False,
            )
            detected = getattr(info, "language", None)
            segments = []
            for index, segment in enumerate(raw_segments, start=1):
                text = " ".join(str(segment.text).split())
                if not text:
                    continue
                segments.append(
                    {
                        "anchor_id": f"{anchor_id}_seg_{index:04d}",
                        "anchor_link": f"{anchor_id}_seg_{index:04d}",
                        "sequence_index": index,
                        "time_range_sec": [round(float(segment.start), 3), round(float(segment.end), 3)],
                        "text": text,
                        "source": attempt["source"],
                        "avg_logprob": _safe_float(getattr(segment, "avg_logprob", None)),
                        "no_speech_prob": _safe_float(getattr(segment, "no_speech_prob", None)),
                    }
                )
                if len(segments) >= max_segments:
                    break
            if segments:
                break
        warning = None if segments else f"faster-whisper produced no segments after VAD/no-VAD retries; detected_language={detected}"
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
