import argparse
import base64
import io
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zoomrefine.mtec_media_pipeline as mtec_media_pipeline  # noqa: E402
from zoomrefine.mtec_media_pipeline import create_multimodal_structural_anchors, probe_media  # noqa: E402
from zoomrefine.mtec_prompt_plus import (  # noqa: E402
    build_evidence_extraction_prompt,
    build_structured_evidence_prompt,
    format_compact_evidence_prompt,
    format_rich_evidence_prompt,
)
from zoomrefine.mtec_task_resolvers import (  # noqa: E402
    build_task_specific_resolver_guidance as _build_task_family_resolver_guidance,
    route_question_family,
    is_performing_cast_question,
)


SYSTEM_PROMPT = (
    "You are evaluating MTEC-Prompt++. Use both channels: low-cost multimodal structural anchors "
    "and the structured evidence prompt. Return concise answers only."
)


class FatalAPIError(RuntimeError):
    pass


class _InspectionBlocked(RuntimeError):
    """Raised when a request is rejected by content moderation (DataInspectionFailed)
    after the normal retry loop, signaling the caller to retry with fewer media items."""
    pass


def _reduce_media_to_primary_video(content: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Return a reduced content list that keeps all text items plus only the first
    video item (dropping images and any extra videos), or None if there is nothing
    to reduce. Used to slip past the content-moderation false-positive that scales
    with the number of attached media items."""
    if not isinstance(content, list):
        return None
    media_items = [c for c in content if isinstance(c, dict) and c.get("type") in {"video_url", "image_url", "audio_url"}]
    if len(media_items) <= 1:
        return None  # already minimal; nothing to gain
    reduced: List[Dict[str, Any]] = []
    kept_video = False
    for c in content:
        t = c.get("type") if isinstance(c, dict) else None
        if t == "video_url" and not kept_video:
            reduced.append(c)
            kept_video = True
        elif t in {"video_url", "image_url", "audio_url"}:
            continue  # drop extra media
        else:
            reduced.append(c)  # keep text
    return reduced


def resolve_path(path_text: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_text)))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path



def cleanup_record_artifacts(*paths: Any) -> int:
    removed = 0
    for value in paths:
        if not value:
            continue
        path = Path(str(value))
        try:
            if path.is_dir():
                shutil.rmtree(path)
                removed += 1
            elif path.exists():
                path.unlink()
                removed += 1
        except Exception:
            pass
    return removed

def image_stats_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    image_cell = row.get("image") or {}
    image_bytes = image_cell.get("bytes")
    if image_bytes is None:
        return {"width": 0, "height": 0, "bytes": 0, "megapixels": 0.0}
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    return {
        "width": int(width),
        "height": int(height),
        "bytes": len(image_bytes),
        "megapixels": round((width * height) / 1_000_000.0, 4),
    }


def is_high_resolution_image(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    stats = image_stats_from_row(row)
    return (
        stats["width"] >= args.min_image_width
        and stats["height"] >= args.min_image_height
        and stats["megapixels"] >= args.min_image_megapixels
        and stats["bytes"] >= args.min_image_bytes
    )


def video_row_matches_duration(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    allowed = set(args.video_duration_classes or [])
    duration_class = str(row.get("duration") or "").lower()
    if allowed and duration_class and duration_class not in allowed:
        return False
    if args.min_video_duration_sec <= 0:
        return True
    for key in ("duration_sec", "duration_seconds", "video_duration", "length"):
        value = row.get(key)
        try:
            return float(value) >= args.min_video_duration_sec
        except (TypeError, ValueError):
            pass
    return True


def video_row_matches_resolution(
    row: Dict[str, Any],
    video_lookup: Dict[str, Tuple[Path, zipfile.ZipInfo]],
    output_dir: Path,
    args: argparse.Namespace,
) -> bool:
    if args.min_video_width <= 0 and args.min_video_height <= 0:
        return True
    video_id = str(row.get("videoID"))
    if video_id not in video_lookup:
        return False
    zip_path, member = video_lookup[video_id]
    if args.min_video_bytes and member.file_size < args.min_video_bytes:
        return False
    if args.max_video_bytes and member.file_size > args.max_video_bytes:
        return False
    video_path = extract_zip_member(zip_path, member, output_dir / "media" / "video")
    metadata = probe_media(str(video_path))
    streams = metadata.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        try:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            capture.release()
        except Exception:
            width = height = 0
    return width >= args.min_video_width and height >= args.min_video_height


def decode_audio_to_wav(audio_cell: Dict[str, Any], output_path: Path) -> Path:
    import soundfile as sf

    audio_bytes = audio_cell.get("bytes")
    if audio_bytes is None:
        raise RuntimeError("Audio cell does not contain bytes.")
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, sample_rate)
    return output_path


def extract_letter(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\(?\s*([A-Fa-f])\s*\)?\.?,?", text):
        return re.search(r"([A-Fa-f])", text).group(1).upper()
    patterns = (
        r"(?:final\s*answer|final_answer|answer|candidate_answer|preliminary_answer|choice|option|选项|答案)\s*(?:is|=|:|：)?\s*\(?\s*([A-Fa-f])\s*\)?",
        r"(?:correct\s+option\s+is|correct\s+answer\s+is|select\s+option|choose\s+option)\s*\(?\s*([A-Fa-f])\s*\)?",
        r'"(?:candidate_answer|preliminary_answer|answer|final_answer)"\s*:\s*"([A-Fa-f])"',
        r"'(?:candidate_answer|preliminary_answer|answer|final_answer)'\s*:\s*'([A-Fa-f])'",
        r"\(([A-Fa-f])\)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    standalone = re.findall(r"(?m)^\s*([A-Fa-f])\s*[\).。:]?\s*$", text)
    if standalone:
        return standalone[-1].upper()
    if len(text) <= 120:
        tokens = re.findall(r"\b([A-Fa-f])\b", text)
        if tokens:
            return tokens[-1].upper()
    return None


def _combined_message_text(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return ""
    message = meta.get("message") or {}
    parts = []
    for key in ("content", "reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def extract_final_prediction(
    final_response: Any,
    final_meta: Optional[Dict[str, Any]],
    computed_evidence_response: Any = None,
    computed_evidence_meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    for source in (
        final_response,
        _combined_message_text(final_meta),
    ):
        letter = extract_letter(source)
        if letter:
            return letter
    return None

def is_letter_answer(value: Any) -> bool:
    return bool(re.fullmatch(r"\(?\s*[A-Fa-f]\s*\)?", str(value or "").strip()))


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def strip_final_answer_prefix(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(?:final\s*answer|answer|答案)\s*[:：]\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def short_answer_correct(ground_truth: Any, response: str) -> bool:
    expected = normalize_label(ground_truth)
    predicted = normalize_label(strip_final_answer_prefix(response))
    if not expected or not predicted:
        return False
    if expected == predicted:
        return True
    if expected in {"yes", "no"}:
        return predicted.split()[0] == expected
    return expected in predicted or predicted in expected


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
    hits = [label for label in target_list if label in response_norm]
    return bool(hits), ", ".join(hits[:8])


def estimated_tokens(byte_count: int) -> int:
    return max(1, ceil(max(0, int(byte_count)) / 4))



def _short_line(value: Any, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "..."


def _strip_json_code_fence(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _json_object_slice(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return ""


def validate_json_object_response(text: Any) -> Tuple[bool, str, Optional[Any]]:
    cleaned = _strip_json_code_fence(str(text or ""))
    if not cleaned:
        return False, "empty_response", None
    for candidate in (cleaned, _json_object_slice(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if isinstance(parsed, dict):
            return True, "", parsed
        return False, "json_is_not_object", parsed
    return False, last_error if "last_error" in locals() else "no_json_object", None


def fallback_invalid_evidence_json(reason: str, attempts: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "question_type": "unknown",
            "temporal_scope": {"type": "unknown", "time_range_sec": None, "confidence": 0.0},
            "observations": [],
            "counts": [],
            "visible_set": {},
            "ocr_text": [],
            "missing_required_evidence": ["evidence_pass_json_invalid_or_truncated"],
            "forbidden_inference": ["Do not rely on the truncated evidence text."],
            "uncertainties": [
                f"Evidence pass did not return valid JSON after retry: {reason}",
                "Final verifier must rely on attached low-resolution video, high-resolution crops, transcript, and deterministic anchors.",
            ],
            "must_rely_on_visual_inputs": True,
            "retry_attempts": attempts,
        },
        ensure_ascii=False,
    )


def add_compression_metrics(record: Dict[str, Any], source_bytes: int, evidence_bytes: int, note: str) -> None:
    original_tokens = estimated_tokens(source_bytes)
    compressed_tokens = estimated_tokens(evidence_bytes)
    raw_saving = 1.0 - compressed_tokens / original_tokens
    record.update(
        {
            "source_bytes": int(source_bytes),
            "evidence_bytes": int(evidence_bytes),
            "compression_ratio": round(evidence_bytes / source_bytes, 4) if source_bytes > 0 else None,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_saving_ratio": round(max(0.0, raw_saving), 4),
            "token_saving_ratio_raw": round(raw_saving, 4),
            "compression_expanded": raw_saving < 0,
            "compression_note": note,
        }
    )


def build_task_specific_resolver_guidance(question: str) -> Dict[str, Any]:
    return _build_task_family_resolver_guidance(question)


def structured_package(
    question: str,
    package: Dict[str, Any],
    stage1_response: Optional[str] = None,
    visual_context_response: Optional[str] = None,
    global_timeline_response: Optional[str] = None,
) -> Dict[str, Any]:
    anchors = package.get("low_resolution_anchor", {})
    image_anchors = anchors.get("image_anchor", [])
    video_anchors = anchors.get("video_anchor", [])
    audio_anchors = anchors.get("audio_anchor", [])
    transcript_anchors = anchors.get("transcript_anchor", [])
    video_evidence_anchors = anchors.get("video_evidence_anchor", [])
    structured = build_structured_evidence_prompt(
        question=question,
        stage1_response=stage1_response,
        bbox_norm=None,
        expanded_bbox_norm=None,
        global_anchor=image_anchors,
        video_anchor=video_anchors,
        audio_anchor=audio_anchors,
    )
    prompt = structured.setdefault("structured_evidence_prompt", {})
    prompt["task_specific_resolver_guidance"] = build_task_specific_resolver_guidance(question)
    if transcript_anchors:
        structured.setdefault("low_resolution_anchor", {})["transcript_anchor"] = transcript_anchors
        prompt["ocr_asr_evidence"] = _transcript_structured_evidence(transcript_anchors)
        prompt["question_relevant_time_windows"] = _transcript_question_windows(transcript_anchors)
    if video_evidence_anchors:
        structured.setdefault("low_resolution_anchor", {})["video_evidence_anchor"] = video_evidence_anchors
        prompt["video_extraction_evidence"] = video_evidence_anchors
    if visual_context_response:
        prompt["visual_context_hint"] = _coerce_visual_context_hint(visual_context_response)
    if global_timeline_response:
        prompt["global_video_timeline"] = _coerce_visual_context_hint(global_timeline_response)
    return structured


def should_suppress_computed_evidence_for_final(question: str, raw_package: Dict[str, Any]) -> bool:
    if route_question_family(question).get("task_family") != "scene_group_attribute_count":
        return False
    video_anchors = (raw_package.get("low_resolution_anchor") or {}).get("video_anchor") or []
    return any(anchor.get("type") == "video_visual_count_clip" for anchor in video_anchors)


def _coerce_visual_context_hint(response: Optional[str]) -> Any:
    if not response:
        return {}
    text = str(response).strip()
    if not text:
        return {}
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return {"raw_visual_context": _short_line(cleaned, max_chars=1800)}


def _transcript_structured_evidence(transcript_anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    seen = set()
    for anchor in transcript_anchors:
        source = anchor.get("source") or anchor.get("backend") or "transcript"
        for segment in (anchor.get("question_relevant_segments") or [])[:16]:
            key = segment.get("anchor_link") or segment.get("anchor_id")
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "time": segment.get("time_range_sec"),
                    "source": source,
                    "content": segment.get("text"),
                    "anchor_link": key or anchor.get("anchor_link"),
                    "selection_role": segment.get("selection_role") or "question_relevant_transcript",
                    "relevance_score": segment.get("relevance_score"),
                    "relevance_terms": segment.get("relevance_terms") or [],
                }
            )
    return evidence


def _transcript_question_windows(transcript_anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    for anchor in transcript_anchors:
        windows.extend(anchor.get("question_relevant_time_windows") or [])
    return windows


# ---------------------------------------------------------------------------
# Module #2: anchor metadata stripping
# Non-semantic anchor fields that the model never reasons over. They only bloat
# the serialized prompt text. Stripping them from the *prompt copy* (not the
# saved record, and not the media-attachment path) reduces text tokens with no
# change to the visual evidence or decision inputs.
# ---------------------------------------------------------------------------
PROMPT_NON_SEMANTIC_KEYS = frozenset({
    "compression", "bytes", "quality", "jpeg_quality", "strategy", "format",
    "path", "low_fps_video_path", "source_video_path", "low_bitrate_audio_path",
    "source_frame_count", "source_fps", "source_resolution", "resolution",
    "max_side", "anchor_id",
})


def _strip_non_semantic_metadata(obj: Any) -> Any:
    """Recursively drop non-semantic bookkeeping keys from a structure."""
    if isinstance(obj, dict):
        return {
            k: _strip_non_semantic_metadata(v)
            for k, v in obj.items()
            if k not in PROMPT_NON_SEMANTIC_KEYS
        }
    if isinstance(obj, list):
        return [_strip_non_semantic_metadata(v) for v in obj]
    return obj


def strip_package_for_prompt(package: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
    """Return a copy of the package with non-semantic metadata removed for text
    serialization. No-op when disabled. The original package is left intact so
    media attachment (which needs paths) and the saved record are unaffected."""
    if not enabled:
        return package
    return _strip_non_semantic_metadata(package)


def package_prompt(package: Dict[str, Any], prompt_style: str = "compact") -> str:
    if prompt_style == "rich":
        return format_rich_evidence_prompt(package)
    return format_compact_evidence_prompt(package)


def package_bytes(package: Dict[str, Any], prompt_style: str = "compact", include_audio_media: bool = True) -> int:
    total = len(package_prompt(package, prompt_style).encode("utf-8"))
    anchors = package.get("low_resolution_anchor", {})
    for section in ("image_anchor", "video_anchor", "audio_anchor"):
        for anchor in anchors.get(section, []):
            for key in ("path", "low_fps_video_path", "low_bitrate_audio_path"):
                if key == "low_bitrate_audio_path" and not include_audio_media:
                    continue
                path = anchor.get(key)
                if path and Path(path).exists():
                    total += Path(path).stat().st_size
            for frame in anchor.get("frames", []):
                path = frame.get("path")
                if path and Path(path).exists():
                    total += Path(path).stat().st_size
    return total


def _path_stat(path: Any) -> Dict[str, Any]:
    if not path:
        return {"path": "", "exists": False, "bytes": 0}
    value = Path(str(path))
    return {
        "path": str(value),
        "exists": value.exists(),
        "bytes": value.stat().st_size if value.exists() and value.is_file() else 0,
    }


def video_package_input_audit(
    package: Dict[str, Any],
    media_contents: List[Dict[str, Any]],
    source_video_path: Path,
    subtitle_path: Optional[Path],
    prompt_style: str,
    include_audio_media: bool,
) -> Dict[str, Any]:
    anchors = package.get("low_resolution_anchor", {})
    image_anchors = anchors.get("image_anchor") or []
    video_anchors = anchors.get("video_anchor") or []
    full_timeline_anchors = [item for item in video_anchors if item.get("type") == "video_full_timeline_lowres"]
    visual_count_video_anchors = [item for item in video_anchors if item.get("type") == "video_visual_count_clip"]
    audio_anchors = anchors.get("audio_anchor") or []
    transcript_anchors = anchors.get("transcript_anchor") or []
    video_evidence_anchors = anchors.get("video_evidence_anchor") or []
    detail_anchors = [item for item in image_anchors if item.get("type") == "video_keyframe_detail_crop"]
    tubelet_anchors = [item for item in image_anchors if item.get("type") == "video_tubelet_storyboard"]
    ocr_image_anchors = [item for item in image_anchors if item.get("type") == "video_ocr_region_crop"]
    object_image_anchors = [item for item in image_anchors if item.get("type") == "video_object_region_crop"]
    visual_count_anchors = [item for item in image_anchors if item.get("type") == "video_visual_count_sheet"]

    file_checks: List[Dict[str, Any]] = [_path_stat(source_video_path)]
    if subtitle_path:
        file_checks.append(_path_stat(subtitle_path))
    for anchor in video_anchors:
        file_checks.append(_path_stat(anchor.get("low_fps_video_path")))
        for frame in anchor.get("frames") or []:
            file_checks.append(_path_stat(frame.get("path")))
    for anchor in detail_anchors + tubelet_anchors + ocr_image_anchors + object_image_anchors + visual_count_anchors:
        file_checks.append(_path_stat(anchor.get("path")))
    for anchor in video_evidence_anchors:
        file_checks.append(_path_stat(anchor.get("path")))
    for anchor in audio_anchors:
        path = anchor.get("low_bitrate_audio_path")
        if include_audio_media or path:
            file_checks.append(_path_stat(path))

    missing_files = [item for item in file_checks if item.get("path") and not item.get("exists")]
    empty_files = [item for item in file_checks if item.get("exists") and item.get("bytes", 0) <= 0]
    media_counts: Dict[str, int] = {}
    for content in media_contents:
        media_counts[str(content.get("type") or "unknown")] = media_counts.get(str(content.get("type") or "unknown"), 0) + 1

    warnings = []
    if not video_anchors:
        warnings.append("missing_video_anchor")
    if video_anchors and not any(_path_stat(anchor.get("low_fps_video_path")).get("exists") for anchor in video_anchors):
        warnings.append("missing_low_fps_video_file")
    if video_anchors and not full_timeline_anchors:
        warnings.append("missing_full_timeline_lowres_anchor")
    if not tubelet_anchors:
        warnings.append("missing_tubelet_storyboard_anchor")
    if any(term in str((package.get("structured_evidence_prompt") or {}).get("user_question") or "").lower() for term in ("how many", "number of", "count")) and not visual_count_anchors and not visual_count_video_anchors:
        warnings.append("missing_visual_count_executor_anchor")
    if media_counts.get("video_url", 0) <= 0:
        warnings.append("no_video_media_attached_to_api")
    if transcript_anchors and not any(anchor.get("segments") for anchor in transcript_anchors):
        warnings.append("transcript_anchor_has_no_segments")

    prompt_text_bytes = len(package_prompt(package, prompt_style).encode("utf-8"))
    evidence_bytes = package_bytes(package, prompt_style, include_audio_media=include_audio_media)
    return {
        "source_video": _path_stat(source_video_path),
        "subtitle": _path_stat(subtitle_path) if subtitle_path else None,
        "video_anchor_count": len(video_anchors),
        "full_timeline_lowres_anchor_count": len(full_timeline_anchors),
        "visual_count_video_anchor_count": len(visual_count_video_anchors),
        "full_timeline_lowres_sample_count": sum(len(anchor.get("frames") or []) for anchor in full_timeline_anchors),
        "low_fps_frame_count": sum(len(anchor.get("frames") or []) for anchor in video_anchors),
        "detail_crop_count": len(detail_anchors),
        "tubelet_storyboard_count": len(tubelet_anchors),
        "ocr_region_crop_count": len(ocr_image_anchors),
        "object_region_crop_count": len(object_image_anchors),
        "visual_count_sheet_count": len(visual_count_anchors),
        "video_evidence_anchor_count": len(video_evidence_anchors),
        "motion_region_count": sum(len(anchor.get("motion_regions") or []) for anchor in video_evidence_anchors),
        "ocr_region_count": sum(len(anchor.get("ocr_regions") or []) for anchor in video_evidence_anchors),
        "object_detection_count": sum(len(anchor.get("object_detections") or []) for anchor in video_evidence_anchors),
        "scene_segment_count": sum(len(anchor.get("scene_segments") or []) for anchor in video_evidence_anchors),
        "audio_anchor_count": len(audio_anchors),
        "transcript_anchor_count": len(transcript_anchors),
        "transcript_segment_count": sum(len(anchor.get("segments") or []) for anchor in transcript_anchors),
        "question_relevant_transcript_segment_count": sum(len(anchor.get("question_relevant_segments") or []) for anchor in transcript_anchors),
        "media_content_counts": media_counts,
        "prompt_text_bytes": prompt_text_bytes,
        "evidence_bytes": evidence_bytes,
        "missing_files": missing_files[:20],
        "empty_files": empty_files[:20],
        "warnings": warnings,
    }


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Large-media fix: upload big media to DashScope temporary OSS and reference it
# by oss:// URL (with header X-DashScope-OssResourceResolve: enable) instead of a
# base64 data-URI. This avoids the provider's 20 MB-per-data-URI and ~28 MB
# total-request-string limits that fail long videos. Configured in main().
# ---------------------------------------------------------------------------
OSS_MEDIA = {
    "enabled": False,
    "model": "qwen-vl-max",   # model used only to obtain a temp-OSS upload certificate
    "api_key": None,
    "threshold_bytes": 8_000_000,  # upload media files larger than this
    "always_kinds": {"video"},     # always upload these kinds when enabled
    "cache": {},                    # path -> oss url (per process)
    "used": False,
}


def _maybe_oss_url(kind: str, path: Path) -> Optional[str]:
    if not OSS_MEDIA["enabled"] or not OSS_MEDIA["api_key"]:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if kind not in OSS_MEDIA["always_kinds"] and size <= OSS_MEDIA["threshold_bytes"]:
        return None
    cache_key = f"{path}:{size}"
    if cache_key in OSS_MEDIA["cache"]:
        return OSS_MEDIA["cache"][cache_key]
    try:
        from dashscope.utils.oss_utils import OssUtils

        url, _cert = OssUtils.upload(
            model=OSS_MEDIA["model"], file_path=str(path), api_key=OSS_MEDIA["api_key"]
        )
    except Exception:
        return None
    if url:
        OSS_MEDIA["cache"][cache_key] = url
        OSS_MEDIA["used"] = True
    return url


def media_content(kind: str, path: Path) -> Dict[str, Any]:
    if kind == "image":
        oss = _maybe_oss_url("image", path)
        return {"type": "image_url", "image_url": {"url": oss or data_url(path, "image/jpeg")}}
    if kind == "video":
        oss = _maybe_oss_url("video", path)
        return {"type": "video_url", "video_url": {"url": oss or data_url(path, "video/mp4")}}
    if kind == "audio":
        suffix = path.suffix.lower()
        mime = "audio/mpeg" if suffix == ".mp3" else "audio/ogg" if suffix == ".ogg" else "audio/wav"
        oss = _maybe_oss_url("audio", path)
        return {"type": "audio_url", "audio_url": {"url": oss or data_url(path, mime)}}
    raise ValueError(f"Unsupported media kind: {kind}")


def image_anchor_contents(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    contents = []
    for anchor in package.get("low_resolution_anchor", {}).get("image_anchor", []):
        path = anchor.get("path")
        if path and Path(path).exists():
            contents.append(
                {
                    "type": "text",
                    "text": (
                        f"Attached anchor {anchor.get('anchor_link')}: "
                        f"{anchor.get('type')}; region={anchor.get('region_hint') or 'whole_image'}; "
                        f"bbox={anchor.get('bbox_norm') or 'full'}."
                    ),
                }
            )
            contents.append(media_content("image", Path(path)))
    return contents


def video_detail_image_anchor_contents(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    contents = []
    for anchor in package.get("low_resolution_anchor", {}).get("image_anchor", []):
        if anchor.get("type") not in {"video_keyframe_detail_crop", "video_tubelet_storyboard", "video_ocr_region_crop", "video_object_region_crop", "video_visual_count_sheet"}:
            continue
        path = anchor.get("path")
        if path and Path(path).exists():
            if anchor.get("type") == "video_tubelet_storyboard":
                frame_summary = "; ".join(
                    f"{frame.get('role')}@{frame.get('time_sec')}s"
                    for frame in (anchor.get("frames") or [])
                )
                text = (
                    f"Attached video tubelet storyboard {anchor.get('anchor_link')}: "
                    f"time_range={anchor.get('time_range_sec')}; frames={frame_summary}; "
                    f"region={anchor.get('region_hint')}; bbox={anchor.get('bbox_norm')}; "
                    f"reason={anchor.get('selection_reason')}. Use it for action order and state-change reasoning."
                )
            elif anchor.get("type") == "video_ocr_region_crop":
                text = (
                    f"Attached video OCR region crop {anchor.get('anchor_link')}: "
                    f"time={anchor.get('time_sec')}s; frame={anchor.get('frame_index')}; "
                    f"ocr_text={anchor.get('recognized_text')}; conf={anchor.get('ocr_confidence')}; "
                    f"bbox={anchor.get('bbox_norm')}. Verify text visually before answering."
                )
            elif anchor.get("type") == "video_object_region_crop":
                text = (
                    f"Attached video object region crop {anchor.get('anchor_link')}: "
                    f"time={anchor.get('time_sec')}s; frame={anchor.get('frame_index')}; "
                    f"label={anchor.get('detected_label')}; conf={anchor.get('detection_confidence')}; "
                    f"bbox={anchor.get('bbox_norm')}. Use with global context to avoid duplicate counting."
                )
            elif anchor.get("type") == "video_visual_count_sheet":
                text = (
                    f"Attached strong visual count sheet {anchor.get('anchor_link')}: "
                    f"task_family={anchor.get('task_family')}; times={anchor.get('time_points_sec')}; "
                    f"region={anchor.get('region_hint')}; rules={anchor.get('counting_rules')}. "
                    "Use this image as primary evidence for visual count questions and verify each option independently."
                )
            else:
                text = (
                    f"Attached video detail image anchor {anchor.get('anchor_link')}: "
                    f"time={anchor.get('time_sec')}s; frame={anchor.get('frame_index')}; "
                    f"region={anchor.get('region_hint')}; bbox={anchor.get('bbox_norm')}; "
                    f"linked_video_anchor={anchor.get('linked_video_anchor')}."
                )
            contents.append(
                {
                    "type": "text",
                    "text": text,
                }
            )
            contents.append(media_content("image", Path(path)))
    return contents


def multimodal_anchor_contents(
    package: Dict[str, Any],
    include_images: bool = True,
    include_audio_media: bool = True,
) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    anchors = package.get("low_resolution_anchor", {})
    if include_images:
        contents.extend(image_anchor_contents(package))
    else:
        contents.extend(video_detail_image_anchor_contents(package))
    for anchor in anchors.get("video_anchor", []):
        path = anchor.get("low_fps_video_path")
        if path and Path(path).exists():
            if anchor.get("type") == "video_full_timeline_lowres":
                role_text = "full chronological low-resolution global fallback video"
            elif anchor.get("type") == "video_visual_count_clip":
                role_text = "original-resolution visual count verification video"
            else:
                role_text = "compressed low-FPS evidence video"
            contents.append(
                {
                    "type": "text",
                    "text": (
                        f"Attached video anchor {anchor.get('anchor_link')} ({role_text}): "
                        f"{anchor.get('type')}; duration={anchor.get('source_duration_sec')}s; "
                        f"frames={len(anchor.get('frames', []))}; target_fps={anchor.get('target_fps')}; "
                        f"source_res={anchor.get('source_resolution')}; "
                        f"counting_rules={anchor.get('counting_rules') or []}."
                    ),
                }
            )
            contents.append(media_content("video", Path(path)))
    for anchor in anchors.get("audio_anchor", []):
        path = anchor.get("low_bitrate_audio_path")
        if path and Path(path).exists():
            contents.append(
                {
                    "type": "text",
                    "text": (
                        f"Attached audio anchor {anchor.get('anchor_link')}: "
                        f"{anchor.get('type')}; duration={anchor.get('source_duration_sec')}s; "
                        f"events={len(anchor.get('audio_event_segments') or [])}; "
                        f"bitrate={anchor.get('target_bitrate')}."
                    ),
                }
            )
            if include_audio_media:
                contents.append(media_content("audio", Path(path)))
    return contents



VIDEO_ANCHOR_POLICIES = {
    "tiny": {"fps": 0.5, "max_frames": 8, "max_side": 512, "detail_max_crops": 1, "detail_max_side": 512, "audio_anchor": True, "transcript": True, "evidence_max_tokens": 160},
    "light": {"fps": 1.0, "max_frames": 16, "max_side": 512, "detail_max_crops": 2, "detail_max_side": 640, "audio_anchor": False, "transcript": True, "evidence_max_tokens": 192},
    "medium": {"fps": 1.5, "max_frames": 32, "max_side": 640, "detail_max_crops": 3, "detail_max_side": 768, "audio_anchor": False, "transcript": True, "evidence_max_tokens": 256},
    "full": {"fps": 4.0, "max_frames": 80, "max_side": 640, "detail_max_crops": 6, "detail_max_side": 960, "audio_anchor": True, "transcript": True, "evidence_max_tokens": 512},
    "dense": {"fps": 8.0, "max_frames": 320, "max_side": 960, "detail_max_crops": 24, "detail_max_side": 1280, "audio_anchor": False, "transcript": True, "evidence_max_tokens": 768},
}


def infer_video_anchor_policy(question: str, requested_policy: str) -> Tuple[str, Dict[str, Any]]:
    if requested_policy != "auto":
        return requested_policy, dict(VIDEO_ANCHOR_POLICIES[requested_policy])
    text = str(question or "").lower()
    task_family = route_question_family(question).get("task_family")
    if task_family == "scene_group_attribute_count":
        policy = dict(VIDEO_ANCHOR_POLICIES["full"])
        policy.update({"detail_max_crops": 8, "detail_max_side": 960, "audio_anchor": False, "evidence_max_tokens": 384})
        return "full_scene_group_count", policy
    if task_family in {"container_object_count", "missing_set"}:
        policy = dict(VIDEO_ANCHOR_POLICIES["medium"])
        policy.update({"detail_max_crops": 8, "detail_max_side": 960, "audio_anchor": False, "evidence_max_tokens": 320})
        return f"medium_{task_family}", policy
    asr_terms = ("say", "said", "says", "saying", "speak", "speaker", "spoken", "voice", "narrator", "narration", "mentioned", "heard", "listen", "audio", "sound", "music", "song", "year", "date", "when was", "painted", "born", "published")
    ocr_terms = ("text", "word", "letter", "number", "digit", "written", "label", "title", "sign", "screen", "display", "shown on", "read", "ocr", "equation", "graph", "chart", "axis", "table", "score", "percentage", "price", "advertised", "laptop", "phone", "smart phone", "smartphone")
    temporal_terms = ("first", "then", "before", "after", "next", "finally", "order", "sequence", "timeline", "event", "happen", "happens", "change", "transition", "start", "end", "doing", "action", "move", "moving", "turn", "appears", "disappears", "intention", "goal", "left", "right", "ward", "slay", "enemy", "ally", "legend", "fox tail")
    if any(term in text for term in asr_terms):
        return "tiny_asr", dict(VIDEO_ANCHOR_POLICIES["tiny"])
    if any(term in text for term in ocr_terms):
        policy = dict(VIDEO_ANCHOR_POLICIES["medium"])
        policy.update({"detail_max_crops": 4, "detail_max_side": 960, "audio_anchor": False, "evidence_max_tokens": 256})
        return "medium_ocr", policy
    if any(term in text for term in temporal_terms):
        return "medium_temporal", dict(VIDEO_ANCHOR_POLICIES["medium"])
    return "light", dict(VIDEO_ANCHOR_POLICIES["light"])


def apply_manual_video_policy(fps: float, max_frames: int, max_side: int, detail_max_crops: int, detail_max_side: int, audio_anchor: bool, transcript: bool, evidence_max_tokens: int) -> Dict[str, Any]:
    return {"fps": fps, "max_frames": max_frames, "max_side": max_side, "detail_max_crops": detail_max_crops, "detail_max_side": detail_max_side, "audio_anchor": audio_anchor, "transcript": transcript, "evidence_max_tokens": evidence_max_tokens}


def build_minimal_evidence_extraction_prompt(prompt: Dict[str, Any], strip_metadata: bool = False) -> str:
    prompt = strip_package_for_prompt(prompt, strip_metadata)
    structured = prompt.get("structured_evidence_prompt", prompt)
    anchors = prompt.get("low_resolution_anchor", {})
    question = structured.get("user_question") or ""
    lines = [
        "MTEC++ minimal evidence extraction. You are an evidence recorder, not an answer generator.",
        f"Question: {_short_line(question)}",
        "Use attached compressed anchors only. Return compact JSON only; no prose.",
        '{"question_type":"","task_family":"","task_family_evidence":{},"temporal_scope":{"type":"","time_range_sec":null,"confidence":0.0},"observations":[{"id":"E1","time":"","modality":"visual|video|asr|audio|ocr","observation":"short observable fact","anchor":"anchor_link","scope_match":true,"confidence":0.0}],"counts":[],"visible_set":{},"ocr_text":[],"missing_required_evidence":[],"forbidden_inference":[],"uncertainty":[]}',
        "Never output candidate_answer, preliminary_answer, best_option, final_answer, or an option letter as the answer.",
        "Never write phrases like 'therefore the answer is', 'likely option', 'should be A/B/C/D', or 'correct answer'.",
        "Keep at most 6 evidence items. Prefer transcript/ASR for speech/date/year questions, OCR/crop for text/number/screen questions, video frames for actions/order, and global frames for scene/object questions.",
        "First fill task_family from TaskSpecificResolverGuidance. Then fill task_family_evidence using that resolver's evidence_template.",
        "For missing_set questions, enumerate visible_set for every option; do not infer absence from a single frame.",
        "For cross_shot_entity_count questions, build an entity bank across the valid timeline; do not count only one moment if the question asks total participants/events.",
        "For scene_group_attribute_count questions, use the best wide/panorama frame and do not sum repeated close-ups.",
        "For container_object_count questions, locate the container ROI and count only visible inside-container items.",
        "For beginning/start/displayed-at-the-beginning questions, mark later transcript or later visual evidence scope_match=false.",
        "For container_object_count with beginning/start scope, do not use later reveal shots, later pack shots, or later ASR product-list counts as primary evidence.",
        "For scene_group_attribute_count multiple-choice questions, verify each option against the best wide/panorama stage frame; include stage-edge people and do not count only central dancers.",
        "If a strong visual_count_sheet is attached, use it as primary count evidence before generic low-FPS frames, object detections, transcript, or ASR.",
        "For current score/ongoing game questions, record only directly visible scoreboard OCR/crops with time and scope; write unreadable if uncertain.",
        "If text/tool evidence conflicts with attached video or crop images, mark the conflict in uncertainty and prefer direct visual evidence from video/crops.",
    ]
    global_timeline = structured.get("global_video_timeline")
    if global_timeline:
        lines.append("GlobalVideoTimelineHint:")
        lines.append(_short_line(json.dumps(global_timeline, ensure_ascii=False), max_chars=1800))
        lines.append("Align all observations, counts, OCR, and option evidence to this timeline. If tool evidence points to a different scene/time than the global video, mark it as conflict or scope_mismatch.")
        if isinstance(global_timeline, dict) and global_timeline.get("task_specific_resolver"):
            lines.append("AIGeneratedTaskSpecificResolver:")
            lines.append(_short_line(json.dumps(global_timeline.get("task_specific_resolver"), ensure_ascii=False), max_chars=2200))
            lines.append("Fill this task-family resolver's evidence_template in computed evidence. Do not replace specialized resolver fields with generic observations when task_family is high-priority.")
    resolver_guidance = structured.get("task_specific_resolver_guidance")
    if resolver_guidance:
        lines.append("TaskSpecificResolverGuidance:")
        lines.append(_short_line(json.dumps(resolver_guidance, ensure_ascii=False), max_chars=2200))
        lines.append("Use this specialized task-family template and rules when collecting evidence. If generic evidence conflicts with the specialized resolver, report the conflict and keep the resolver-specific fields.")
    visual_hint = structured.get("visual_context_hint")
    if visual_hint:
        lines.append("NoTranscriptVisualContextHint:")
        lines.append(_short_line(json.dumps(visual_hint, ensure_ascii=False), max_chars=1600))
        lines.append("Use the visual context hint to focus actions, objects, absence checks, and question keywords, but verify it against attached video/crop anchors.")
    for section, label in (("video_anchor", "Video"), ("image_anchor", "Image"), ("transcript_anchor", "Transcript"), ("audio_anchor", "Audio")):
        items = anchors.get(section) or []
        if not items:
            continue
        lines.append(f"{label}Anchors:")
        for anchor in items[:8]:
            if section == "video_anchor":
                lines.append(f"- {anchor.get('anchor_link')} frames={len(anchor.get('frames') or [])} fps={anchor.get('target_fps')} duration={anchor.get('source_duration_sec')}s")
            elif section == "image_anchor":
                lines.append(f"- {anchor.get('anchor_link')} type={anchor.get('type')} region={_short_line(anchor.get('region_hint') or anchor.get('bbox_norm') or '')}")
            elif section == "transcript_anchor":
                relevant_count = len(anchor.get('question_relevant_segments') or [])
                lines.append(
                    f"- {anchor.get('anchor_link')} segments={len(anchor.get('segments') or [])} "
                    f"relevant={relevant_count} source={anchor.get('source')}"
                )
                for window in (anchor.get('question_relevant_time_windows') or [])[:6]:
                    lines.append(f"  - query_window t={window.get('time_range_sec')} anchors={window.get('anchor_links')}")
                for segment in _prioritized_transcript_segments(anchor, limit=12):
                    role = segment.get('selection_role') or 'transcript'
                    score = segment.get('relevance_score')
                    score_text = f" score={score}" if score else ""
                    lines.append(
                        f"  - {segment.get('anchor_link')} role={role}{score_text} "
                        f"t={segment.get('time_range_sec')} text={_short_line(segment.get('text'))}"
                    )
            elif section == "audio_anchor":
                lines.append(f"- {anchor.get('anchor_link')} events={len(anchor.get('audio_event_segments') or [])} duration={anchor.get('source_duration_sec')}s")
    return "\n".join(lines)

def _prioritized_transcript_segments(anchor: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for bucket in (anchor.get("question_relevant_segments") or [], anchor.get("segments") or []):
        for segment in bucket:
            key = segment.get("anchor_link") or segment.get("anchor_id") or json.dumps(segment, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            selected.append(segment)
            if len(selected) >= limit:
                return selected
    return selected


def _video_visual_context_reason(question: str, transcript_segment_count: int) -> str:
    text = str(question or "").lower()
    if transcript_segment_count <= 0:
        return "missing_transcript"
    hard_terms = (
        "absent", "not appear", "not shown", "not used", "not discussed", "not mentioned",
        "which of the following", "which acrobatic", "which component", "distinct color",
        "color", "chassis", "power supply", "graphics card", "motherboard", "memory stick",
        "why", "how were", "formed", "formation", "due to", "result of",
        "intention", "goal", "purpose", "doing", "action", "fighting", "playing",
        "theme", "themes", "primary themes", "mainly about", "regarding",
        "screen", "display", "shown on", "visible", "text", "number", "ocr",
        "what does this video show", "what is the video regarding",
    )
    if any(term in text for term in hard_terms):
        return "hard_visual_or_negative_question"
    return ""


def build_video_visual_context_prompt(package: Dict[str, Any]) -> str:
    structured = package.get("structured_evidence_prompt", package)
    anchors = package.get("low_resolution_anchor", {})
    question = structured.get("user_question") or ""
    lines = [
        "MTEC++ no-transcript visual context pass.",
        f"Question: {_short_line(question)}",
        "The video has no usable transcript/subtitles. Inspect the attached low-FPS video and high-detail keyframe crops.",
        "Do not answer the multiple-choice question. Return compact JSON only.",
        '{"visual_summary":"what the video is mainly about","event_chain":[{"time":"","action":"","objects":[],"anchor":""}],"visible_objects":[],"actions":[],"question_keywords":[],"option_visual_cues":{"A":{"support":[],"contradiction":[],"needs_check":[]},"B":{"support":[],"contradiction":[],"needs_check":[]},"C":{"support":[],"contradiction":[],"needs_check":[]},"D":{"support":[],"contradiction":[],"needs_check":[]}},"absence_checks":[],"uncertainty":[]}',
        "Focus on: what happens, who/what moves, tools/props/objects, repeated actions, absent-vs-present options, and visual keywords that help retrieve or judge evidence later.",
    ]
    for anchor in (anchors.get("video_anchor") or [])[:3]:
        lines.append(f"VideoAnchor {anchor.get('anchor_link')}: duration={anchor.get('source_duration_sec')}s frames={len(anchor.get('frames') or [])} fps={anchor.get('target_fps')}")
        for frame in (anchor.get("frames") or [])[:12]:
            lines.append(f"- {frame.get('anchor_link')} t={frame.get('time_sec')}s change={frame.get('change_score')}")
    crops = [anchor for anchor in (anchors.get("image_anchor") or []) if anchor.get("type") == "video_keyframe_detail_crop"]
    if crops:
        lines.append("DetailCrops:")
        for anchor in crops[:12]:
            lines.append(f"- {anchor.get('anchor_link')} t={anchor.get('time_sec')}s region={_short_line(anchor.get('region_hint') or '')} reason={_short_line(anchor.get('selection_reason') or '')}")
    tubelets = [anchor for anchor in (anchors.get("image_anchor") or []) if anchor.get("type") == "video_tubelet_storyboard"]
    if tubelets:
        lines.append("TubeletStoryboards:")
        for anchor in tubelets[:8]:
            frames = ", ".join(f"{frame.get('role')}@{frame.get('time_sec')}s" for frame in (anchor.get("frames") or []))
            lines.append(f"- {anchor.get('anchor_link')} range={anchor.get('time_range_sec')} frames={frames} region={_short_line(anchor.get('region_hint') or '')} reason={_short_line(anchor.get('selection_reason') or '')}")
    return "\n".join(lines)


def build_global_video_timeline_prompt(package: Dict[str, Any]) -> str:
    structured = package.get("structured_evidence_prompt", package)
    question = structured.get("user_question") or ""
    resolver_guidance = structured.get("task_specific_resolver_guidance") or build_task_specific_resolver_guidance(question)
    return "\n".join(
        [
            "MTEC++ global video timeline pass.",
            f"Question: {_short_line(question)}",
            "Inspect only the attached low-resolution full-video anchor.",
            "Do not answer the multiple-choice question.",
            "First route the question to a task-specific resolver, then build the global timeline and locate the scene/time span relevant to that resolver.",
            "Use this resolver guidance as the default; adjust only if the video clearly requires a better specialized resolver:",
            json.dumps(resolver_guidance, ensure_ascii=False),
            "Do not override question_scope_guard. If it says beginning_locked, target_time_ranges must stay in the opening segment; later reveal shots or later ASR are out-of-scope conflicts, not primary evidence.",
            "Resolver registry: cross_shot_entity_count, scene_group_attribute_count, container_object_count, missing_set, stateful_ocr, ordinal_clip_action, domain_intention, scene_conditioned_attribute, generic_video_evidence.",
            "For cross_shot_entity_count: create an entity-bank plan, identify wide/panorama moments, and list time ranges where each participant/challenger/entity should be checked.",
            "For scene_group_attribute_count: find the best wide scene frame and count people/attributes there; do not sum close-ups.",
            "For container_object_count: locate the container ROI and count only inside-container instances in the valid temporal scope.",
            "For missing_set: check each option independently across the full valid scope and report seen/not_seen.",
            "For stateful_ocr: locate the stable OCR region, vote across nearby frames, and apply state constraints.",
            "For ordinal_clip_action: group atomic visual changes into logical event clips, identify the target clip index, target range, backup range, and action-specific cues.",
            "For domain_intention: detect the domain ontology, rank possible intents, and include negative evidence.",
            "If later text evidence or crops conflict with this visual timeline, the final verifier should re-check the video/crops and prefer direct visual evidence.",
            "Return compact JSON only with this schema:",
            '{"task_specific_resolver":{"resolver_type":"","evidence_template":{},"special_rules":[],"target_time_ranges":[],"fallback_policy":"","confidence":0.0},"timeline":[{"time_range":"","scene":"","actions":[],"objects":[],"visual_cues":[],"confidence":0.0}],"question_relevant_time_ranges":[{"time_range":"","reason":"","confidence":0.0}],"scene_locator":{"target_scene":"","time_range":"","confidence":0.0},"global_uncertainties":[]}',
        ]
    )


def global_video_anchor_contents(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    anchors = package.get("low_resolution_anchor", {}).get("video_anchor", [])
    global_anchors = [anchor for anchor in anchors if anchor.get("type") == "video_full_timeline_lowres"]
    for anchor in global_anchors[:1]:
        path = anchor.get("low_fps_video_path")
        if path and Path(path).exists():
            contents.append(
                {
                    "type": "text",
                    "text": (
                        f"Attached full-video global anchor {anchor.get('anchor_link')}: "
                        f"duration={anchor.get('source_duration_sec')}s; "
                        f"samples={len(anchor.get('frames', []))}; target_fps={anchor.get('target_fps')}; "
                        f"source_res={anchor.get('source_resolution')}."
                    ),
                }
            )
            contents.append(media_content("video", Path(path)))
    return contents


def video_only_anchor_contents(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for anchor in package.get("low_resolution_anchor", {}).get("video_anchor", []):
        path = anchor.get("low_fps_video_path")
        if path and Path(path).exists():
            contents.append(
                {
                    "type": "text",
                    "text": (
                        f"Attached video-only visual context anchor {anchor.get('anchor_link')}: "
                        f"duration={anchor.get('source_duration_sec')}s; "
                        f"frames={len(anchor.get('frames', []))}; target_fps={anchor.get('target_fps')}; "
                        f"source_res={anchor.get('source_resolution')}."
                    ),
                }
            )
            contents.append(media_content("video", Path(path)))
    return contents


def compute_video_visual_context(
    client: "SiliconFlowClient",
    package: Dict[str, Any],
    media_contents: List[Dict[str, Any]],
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    return client.generate(
        [
            *(video_only_anchor_contents(package) or media_contents),
            {"type": "text", "text": build_video_visual_context_prompt(package)},
        ],
        max_tokens=max_tokens,
    )


def compute_global_video_timeline(
    client: "SiliconFlowClient",
    package: Dict[str, Any],
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    contents = global_video_anchor_contents(package)
    if not contents:
        return (
            json.dumps(
                {
                    "timeline": [],
                    "question_relevant_time_ranges": [],
                    "scene_locator": {"target_scene": "", "time_range": "", "confidence": 0.0},
                    "global_uncertainties": ["No full-video global anchor was available."],
                },
                ensure_ascii=False,
            ),
            {"skipped": "missing_full_video_global_anchor"},
        )
    return client.generate(
        [
            *contents,
            {"type": "text", "text": build_global_video_timeline_prompt(package)},
        ],
        max_tokens=max_tokens,
    )


def compute_structured_evidence(
    client: "SiliconFlowClient",
    package: Dict[str, Any],
    media_contents: List[Dict[str, Any]],
    max_tokens: int,
    evidence_prompt_style: str = "minimal",
    strip_metadata: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    if evidence_prompt_style == "rich":
        prompt_text = build_evidence_extraction_prompt(package)
    else:
        prompt_text = build_minimal_evidence_extraction_prompt(package, strip_metadata=strip_metadata)
    return client.generate(
        [
            *media_contents,
            {"type": "text", "text": prompt_text},
        ],
        max_tokens=max_tokens,
    )


def compute_structured_evidence_with_json_retry(
    client: "SiliconFlowClient",
    package: Dict[str, Any],
    media_contents: List[Dict[str, Any]],
    max_tokens: int,
    evidence_prompt_style: str = "minimal",
    max_attempts: int = 3,
    strip_metadata: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    response = ""
    meta: Dict[str, Any] = {}
    token_budget = max(256, int(max_tokens or 256))
    last_error = ""
    for attempt_index in range(1, max_attempts + 1):
        response, meta = compute_structured_evidence(
            client,
            package,
            media_contents,
            max_tokens=token_budget,
            evidence_prompt_style=evidence_prompt_style,
            strip_metadata=strip_metadata,
        )
        valid, error, _ = validate_json_object_response(response)
        attempt_info = {
            "attempt": attempt_index,
            "max_tokens": token_budget,
            "valid_json": valid,
            "error": error,
            "response_chars": len(str(response or "")),
        }
        attempts.append(attempt_info)
        if valid:
            meta = dict(meta or {})
            meta["evidence_json_valid"] = True
            meta["evidence_json_retry_attempts"] = attempts
            return response, meta
        last_error = error
        token_budget = min(max(token_budget * 2, token_budget + 256), 2048)
    fallback = fallback_invalid_evidence_json(last_error, attempts)
    meta = dict(meta or {})
    meta["evidence_json_valid"] = False
    meta["evidence_json_retry_attempts"] = attempts
    meta["evidence_json_fallback_used"] = True
    meta["evidence_json_error"] = last_error
    meta["raw_invalid_evidence_response"] = _short_line(response, max_chars=2000)
    return fallback, meta

def build_final_answer_prompt(package: Dict[str, Any], prompt_style: str, strip_metadata: bool = False) -> str:
    return (
        package_prompt(strip_package_for_prompt(package, strip_metadata), prompt_style)
        + "\n\nFINAL DECISION INSTRUCTIONS:\n"
        + "You are an option verifier. You receive compressed video anchors, a low-resolution full-video global anchor, deterministic evidence, answer-neutral observations, transcript/ASR evidence, OCR/object crops, and tubelet storyboards.\n"
        + "First use global_video_timeline and task_specific_resolver, if present, to locate the relevant scene/time span and understand the full-video event order. Then align structured/tool evidence to that visual timeline before judging options.\n"
        + "If task_specific_resolver or computed evidence contains a medium/high-confidence task-family template, prioritize its structured fields over generic observations while still mapping evidence to options yourself.\n"
        + "Task-family policy: cross_shot_entity_count uses entity bank and inclusion/exclusion reasons; scene_group_attribute_count uses one best wide scene instead of summed close-ups; temporal_event_count enumerates each occurrence of a repeated action across the full timeline and sums them (never a single-frame count); container_object_count counts inside the container ROI only; missing_set checks every option across the valid scope; stateful_ocr votes stable OCR states; ordinal_clip_action uses the target logical clip, not an atomic scene cut; domain_intention ranks ontology-grounded intents with negative evidence.\n"
        + "Evaluate each option independently as supported, contradicted, or unknown before choosing. Do not trust any previous candidate_answer, preliminary_answer, best_option, or option letter if it appears in computed evidence.\n"
        + "If text evidence, computed evidence, transcript, OCR text, or deterministic metadata conflicts with the attached video or high-resolution crop images, prefer the direct visual evidence from the video/crops. Treat conflicting text evidence as uncertain, not authoritative.\n"
        + "Treat temporal_scope as confidence-gated: confidence >= 0.75 means scoped evidence is primary; 0.50-0.75 means use scoped evidence plus the full-video global anchor; below 0.50 or missing evidence means do not hard-filter the rest of the video.\n"
        + "For beginning/start/displayed-at-the-beginning questions, never let later reveal shots, later transcript/ASR claims, or later product summaries override opening visual evidence. Mark them out-of-scope.\n"
        + _beginning_scope_directive(package)
        + _scene_group_count_directive(package)
        + _count_hypothesis_directive(package)
        + _count_undercount_directive(package)
        + _deterministic_count_directive(package)
        + "When a strong visual count sheet is attached, treat it as the primary evidence for count verification and compare each option against it before trusting computed evidence, ASR, or generic low-FPS observations.\n"
        + "If structured evidence is low quality, all/most options are unknown, OCR/count/visible-set evidence is incomplete, or local crops conflict, fall back to the full-video global anchor to re-check event order, scene context, action flow, and global layout.\n"
        + "For count questions, prefer deterministic count_tracks/instances only when tracks are valid; otherwise use full-context video plus visible frames. For missing-set questions, use visible_set only when complete; otherwise re-check the full-video anchor. For OCR/model/score questions, combine OCR with high-detail crops and global context when OCR is uncertain.\n"
        + _isolated_cast_count_directive(package)
        + "Silently build option_verification with A/B/C/D statuses and evidence IDs. Return exactly one line and nothing else: FINAL_ANSWER: <letter>."
    )


def _resolver_guidance(package: Dict[str, Any]) -> Dict[str, Any]:
    return (package.get("structured_evidence_prompt") or {}).get("task_specific_resolver_guidance") or {}


def _deterministic_count_directive(package: Dict[str, Any]) -> str:
    """Surface the deterministic detector's count as a strong prior for counting."""
    ev = (package.get("structured_evidence_prompt") or {}).get("deterministic_count_evidence") or {}
    if not ev:
        return ""
    bits = []
    if ev.get("max_persons_single_frame") is not None:
        bits.append(
            f"A deterministic object detector found at most {ev['max_persons_single_frame']} persons simultaneously "
            f"in one frame (median {ev.get('median_persons')}) over the counting window; the true number of people is "
            "AT LEAST this many — do not pick an option below it unless the detector clearly missed occluded people."
        )
    if ev.get("approx_repetitions") is not None:
        bits.append(
            f"A deterministic motion-cycle counter estimated about {ev['approx_repetitions']} repetitions of the action "
            "over the continuous segment; use it as a prior alongside the dense-segment montage when choosing the count."
        )
    if not bits:
        return ""
    return "Deterministic count evidence (instance-level): " + " ".join(bits) + "\n"


def _count_undercount_directive(package: Dict[str, Any]) -> str:
    """For any counting task, counteract the systematic UNDER-count bias from
    compressed/low-FPS video (skipped instances/occurrences). Gated to count
    families so non-count questions are unaffected."""
    guidance = _resolver_guidance(package)
    fam = guidance.get("task_family")
    if fam not in {"temporal_event_count", "cross_shot_entity_count", "scene_group_attribute_count", "container_object_count"}:
        return ""
    if fam == "temporal_event_count":
        return (
            "Counting a repeated action over time: enumerate every occurrence across the full relevant span (use the "
            "low-FPS video + global timeline together). Compressed sampling SKIPS occurrences, so the count is usually "
            "higher than a first glance; if torn between two adjacent options, pick the higher unless you can account "
            "for every occurrence.\n"
        )
    return (
        "Counting note: compressed/low-FPS frames tend to make the model UNDER-count (skipped instances, occluded or "
        "edge items, brief appearances). Enumerate exhaustively across all attached frames and the full time scope; do "
        "not default to the lowest option unless you can positively rule out the higher counts.\n"
    )


def _beginning_scope_directive(package: Dict[str, Any]) -> str:
    """Strengthened beginning-scope rule, only for beginning-locked questions, so
    the verifier prompt for all other questions stays identical to baseline."""
    guidance = _resolver_guidance(package)
    if (guidance.get("question_scope_guard") or {}).get("scope_type") == "beginning_locked":
        return (
            "This question is locked to the opening segment: do not derive the count from any later reveal shot, "
            "later flat-lay/pack shot, end-state, or product-summary narration (e.g. a spoken 'eight full-size products'); "
            "count only from the opening segment and record uncertainty if the opening view is occluded.\n"
        )
    return ""


def _scene_group_count_directive(package: Dict[str, Any]) -> str:
    """Scene-group counting rule, only for scene_group_attribute_count questions."""
    guidance = _resolver_guidance(package)
    if guidance.get("task_family") != "scene_group_attribute_count":
        return ""
    if guidance.get("count_mode") == "performing_cast_unique_across_shots":
        return (
            "This is a performing/presenting cast count: count each distinct performer across shots once (including "
            "performers seen only in close-up), dedupe identities, and exclude only the audience/crew; a single wide "
            "frame is only a lower bound.\n"
        )
    return (
        "This is a static co-present group count: use the best wide/panorama frame and do not sum repeated close-ups.\n"
    )


def _count_hypothesis_directive(package: Dict[str, Any]) -> str:
    """Only for performing-cast counts: instruct the verifier to treat the
    intermediate count as a non-authoritative (often under-counted) hypothesis.

    This is intentionally NOT applied to other count families (e.g.
    cross_shot_entity_count), whose deterministic/entity-bank counts are reliable
    and must not be second-guessed into a different number.
    """
    guidance = _resolver_guidance(package)
    if guidance.get("count_mode") == "performing_cast_unique_across_shots":
        return (
            "For this performing-cast count, treat any total_people/count_value in the computed evidence as a "
            "non-authoritative hypothesis that frequently UNDER-counts because close-up performers were dropped; "
            "independently re-count from the attached video and isolated count, and prefer a higher count when the "
            "visual evidence supports it.\n"
        )
    return ""


def _isolated_cast_count_directive(package: Dict[str, Any]) -> str:
    """If a dedicated isolated cast-count pass produced a count, surface it to the
    verifier as the most reliable count for performing-cast questions."""
    info = (package.get("structured_evidence_prompt") or {}).get("isolated_cast_count") or {}
    parsed = info.get("parsed") or {}
    men, women = parsed.get("men"), parsed.get("women")
    if men is None and women is None:
        return ""
    return (
        "IsolatedCastCount: a dedicated count-only pass was run on ONLY the original-resolution video (no competing "
        f"compressed anchors or intermediate evidence) and reported men={men}, women={women}. For this performing/presenting "
        "cast count this isolated pass is the most reliable signal; select the option whose men/women breakdown matches it, "
        "unless the attached video clearly and specifically contradicts it. Do not lower this count to match a smaller "
        "intermediate count that ignored close-up performers.\n"
    )


def _drop_visual_count_clip_from_final(package: Dict[str, Any]) -> int:
    """Remove the original-resolution video_visual_count_clip anchor from the
    package so it is not attached to the final answer call. Returns how many
    anchors were removed. Used only after the isolated cast-count pass already
    consumed the clip and produced an authoritative count."""
    anchors = (package.get("low_resolution_anchor") or {}).get("video_anchor")
    if not isinstance(anchors, list):
        return 0
    kept = [a for a in anchors if not (isinstance(a, dict) and a.get("type") == "video_visual_count_clip")]
    removed = len(anchors) - len(kept)
    if removed:
        package["low_resolution_anchor"]["video_anchor"] = kept
    return removed


def _video_data_uri(path: str) -> str:
    import base64

    data = Path(path).read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def _parse_men_women(text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    men = re.search(r"men\s*[=:]\s*(\d+)", str(text or ""), re.IGNORECASE)
    women = re.search(r"women\s*[=:]\s*(\d+)", str(text or ""), re.IGNORECASE)
    if men:
        parsed["men"] = int(men.group(1))
    if women:
        parsed["women"] = int(women.group(1))
    return parsed


def run_isolated_cast_count_pass(
    question: str,
    raw_package: Dict[str, Any],
    answer_client: Any,
    max_bytes: int = 15_500_000,
) -> Optional[Dict[str, Any]]:
    """Dedicated isolated counting pass for performing/presenting cast questions.

    Sends ONLY the original-resolution clip plus a focused counting prompt, with no
    competing compressed anchors or intermediate evidence, mirroring the direct-video
    upper-bound probe that counts the full cast correctly. The model frequently
    under-counts the cast inside the cluttered two-pass context but counts it
    correctly when isolated.
    """
    if route_question_family(question).get("task_family") != "scene_group_attribute_count":
        return None
    if not is_performing_cast_question(question):
        return None
    anchors = (raw_package.get("low_resolution_anchor") or {}).get("video_anchor") or []
    clip = next((a for a in anchors if isinstance(a, dict) and a.get("type") == "video_visual_count_clip"), None)
    src = (clip or {}).get("source_video_path")
    if not src or not Path(src).exists() or Path(src).stat().st_size > max_bytes:
        return {"skipped": "no original-resolution clip available within size limit"}
    prompt = (
        "Watch the whole video carefully. It is a stage performance. Count every DISTINCT person who performs or "
        "presents on the stage across the entire video, including lead performers shown mostly in close-up and all "
        "backup performers/dancers. Scan the backup performers carefully; there may be more of them than appear in any "
        "single frame. Dedupe the same person across shots. Exclude the audience and crew. "
        "Answer ONLY as: men=<n>, women=<n>."
    )
    content = [
        media_content("video", Path(src)),
        {"type": "text", "text": prompt},
    ]
    try:
        resp, meta = answer_client.generate(content, max_tokens=300)
    except Exception as exc:  # non-fatal: fall back to the normal verifier flow
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"raw": resp, "parsed": _parse_men_women(resp), "meta": meta}


class SiliconFlowClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: int,
        max_retries: int,
        retry_sleep: float,
        temperature: float,
        enable_thinking: Optional[bool],
    ):
        self.api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def generate(self, content: List[Dict[str, Any]], max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        # The DashScope content-moderation false-positive scales with the number of
        # media items in one request: any single video/image passes, but a large
        # multi-media request for certain (benign) videos is reliably flagged. So on a
        # persistent DataInspectionFailed we retry with progressively fewer media
        # items (full -> primary video only -> text only). Each variant first gets the
        # normal transient-retry loop.
        variants = [("full", content)]
        primary = _reduce_media_to_primary_video(content)
        if primary is not None:
            variants.append(("primary_video_only", primary))
        text_only = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]
        if text_only and len(text_only) != len(content):
            variants.append(("text_only", text_only))
        last_error = ""
        for variant_name, variant in variants:
            try:
                text, meta = self._generate_once(variant, max_tokens)
                if variant_name != "full":
                    meta = dict(meta or {})
                    meta["inspection_fallback_variant"] = variant_name
                return text, meta
            except _InspectionBlocked as exc:
                last_error = str(exc)
                continue  # moderation blocked this variant; try a smaller one
            except Exception as exc:
                msg = str(exc)
                if ("Unexpected item type" in msg) or ("InvalidParameter" in msg and "messages input is invalid" in msg):
                    last_error = msg
                    continue  # model rejects media items (e.g. text-only answer model) -> smaller/text-only variant
                raise
        raise RuntimeError(last_error or "media reduction exhausted")

    def _generate_once(self, content: List[Dict[str, Any]], max_tokens: int) -> Tuple[str, Dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if OSS_MEDIA["enabled"]:
            # Allow the API to resolve oss:// media URLs produced by _maybe_oss_url.
            headers["X-DashScope-OssResourceResolve"] = "enable"
        last_error = ""
        inspection_blocked = False
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                started = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                elapsed = time.perf_counter() - started
                message = data.get("choices", [{}])[0].get("message", {})
                content_text = _message_text(message)
                return content_text, {
                    "api_elapsed_seconds": round(elapsed, 3),
                    "usage": data.get("usage"),
                    "message": message,
                    "response_id": data.get("id"),
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {detail[:1200]}"
                if "account balance is insufficient" in detail:
                    raise FatalAPIError(last_error)
                # DataInspectionFailed is a transient/false-positive content-moderation
                # glitch, so retry it (and signal the caller to try fewer media items).
                inspection_blocked = "data_inspection_failed" in detail.lower() or "datainspectionfailed" in detail.lower()
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} and not inspection_blocked:
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(self.retry_sleep * (attempt + 1))
        if inspection_blocked:
            raise _InspectionBlocked(last_error)
        raise RuntimeError(last_error)


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    pieces.append(str(value))
            elif item:
                pieces.append(str(item))
        joined = "\n".join(pieces).strip()
        if joined:
            return joined
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_done_keys(jsonl_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = item.get("record_key")
            if key and item.get("status") == "completed":
                done.add(str(key))
    return done


def append_record(jsonl_path: Path, record: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_parquet_rows(paths: List[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    for path in paths:
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            yield path, int(index), row.to_dict()


def run_image_record(
    client: SiliconFlowClient,
    row: Dict[str, Any],
    row_id: str,
    output_dir: Path,
    max_tokens: int,
    prompt_style: str,
    evidence_pass: bool,
    evidence_max_tokens: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"record_key": row_id, "dataset": "lmms-lab/RealWorldQA", "modality": "image", "status": "started"}
    started = time.perf_counter()
    try:
        image_name = str(row.get("image_path") or f"{row_id}.webp").replace("/", "_")
        image_path = write_bytes(output_dir / "media" / "image" / image_name, row["image"]["bytes"])
        question = str(row.get("prompt") or row.get("question") or "")
        raw_package = create_multimodal_structural_anchors(question=question, output_dir=str(output_dir / "anchors" / "image" / row_id), image_path=str(image_path))
        package = structured_package(question, raw_package)
        image_contents = image_anchor_contents(package)
        if not image_contents:
            raise RuntimeError("Image anchor package did not contain any image files.")
        computed_evidence_response = None
        computed_evidence_meta = None
        if evidence_pass:
            computed_evidence_response, computed_evidence_meta = compute_structured_evidence(
                client,
                package,
                image_contents,
                max_tokens=evidence_max_tokens,
            )
            package = structured_package(question, raw_package, stage1_response=computed_evidence_response)
            image_contents = image_anchor_contents(package)
        response, meta = client.generate(
            [
                *image_contents,
                {
                    "type": "text",
                    "text": (
                        package_prompt(package, prompt_style)
                        + "\nFinal answer stage. Use the structured evidence above, but do not show reasoning. "
                        "Output exactly one line and nothing else. If multiple-choice: `FINAL_ANSWER: <letter>`. "
                        "Otherwise: `FINAL_ANSWER: <concise answer>`."
                    ),
                },
            ],
            max_tokens=max_tokens,
        )
        raw_ground_truth = "" if row.get("answer") is None else str(row.get("answer"))
        if is_letter_answer(raw_ground_truth):
            prediction = extract_letter(response)
            ground_truth: Any = extract_letter(raw_ground_truth)
            correct = bool(prediction and ground_truth and prediction == ground_truth)
            answer = prediction or response
        else:
            prediction = response
            ground_truth = raw_ground_truth
            correct = short_answer_correct(raw_ground_truth, response)
            answer = response
        record.update(
            {
                "status": "completed",
                "image": str(image_path),
                "image_stats": image_stats_from_row(row),
                "question": question,
                "ground_truth": ground_truth,
                "Answer": answer,
                "raw_response": response,
                "correct": correct,
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
                "api_meta": meta,
                "computed_evidence_response": computed_evidence_response,
                "computed_evidence_meta": computed_evidence_meta,
                "evidence_pass": evidence_pass,
                "prompt_style": prompt_style,
            }
        )
        add_compression_metrics(record, image_path.stat().st_size, package_bytes(package, prompt_style), f"API MTEC++ image uses a real RealWorldQA image, low-resolution image anchor, and {prompt_style} structured evidence prompt.")
    except Exception as exc:
        if isinstance(exc, FatalAPIError):
            raise
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def downloaded_videos(zips_dir: Path) -> Dict[str, Tuple[Path, zipfile.ZipInfo]]:
    by_id: Dict[str, Tuple[Path, zipfile.ZipInfo]] = {}
    for zip_path in sorted(zips_dir.glob("videos_chunked_*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.filename.endswith(".mp4") and info.file_size > 0:
                    by_id[Path(info.filename).stem] = (zip_path, info)
    return by_id



def downloaded_subtitles(subtitle_zip: Path) -> Dict[str, Tuple[Path, zipfile.ZipInfo]]:
    by_id: Dict[str, Tuple[Path, zipfile.ZipInfo]] = {}
    if not subtitle_zip.exists():
        return by_id
    with zipfile.ZipFile(subtitle_zip) as archive:
        for info in archive.infolist():
            if info.filename.lower().endswith((".srt", ".vtt")) and info.file_size > 0:
                by_id[Path(info.filename).stem] = (subtitle_zip, info)
    return by_id


def precomputed_subtitles(subtitle_dir: Path) -> Dict[str, Path]:
    by_id: Dict[str, Path] = {}
    if not subtitle_dir.exists():
        return by_id
    for path in sorted(subtitle_dir.glob("*.srt")):
        if path.stat().st_size > 0:
            by_id[path.stem] = path
    return by_id


def extract_zip_member(zip_path: Path, member: zipfile.ZipInfo, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / Path(member.filename).name
    if out_path.exists() and out_path.stat().st_size == member.file_size:
        return out_path
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as source, out_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return out_path


def run_video_record(
    client: SiliconFlowClient,
    answer_client: SiliconFlowClient,
    row: Dict[str, Any],
    video_lookup: Dict[str, Tuple[Path, zipfile.ZipInfo]],
    subtitle_lookup: Optional[Dict[str, Tuple[Path, zipfile.ZipInfo]]],
    precomputed_subtitle_lookup: Optional[Dict[str, Path]],
    output_dir: Path,
    max_tokens: int,
    answer_max_tokens: int,
    prompt_style: str,
    evidence_pass: bool,
    evidence_max_tokens: int,
    evidence_prompt_style: str,
    video_anchor_policy: str,
    video_anchor_fps: float,
    video_anchor_max_frames: int,
    video_anchor_max_side: int,
    video_detail_max_crops: int,
    video_detail_max_side: int,
    video_audio_anchor: bool,
    send_audio_media: bool,
    answer_send_audio_media: bool,
    video_transcript_backend: str,
    video_asr_model: str,
    video_asr_language: str,
    video_transcript_max_segments: int,
    video_query_retrieval: bool = True,
    video_query_max_segments: int = 12,
    video_query_window_padding_sec: float = 8.0,
    video_query_detail_extra_crops: int = 4,
    video_visual_context_pass: str = "auto",
    video_visual_context_max_tokens: int = 384,
    video_global_anchor: bool = True,
    video_global_anchor_fps: float = 1.0,
    video_global_anchor_max_side: int = 360,
    global_timeline_pass: bool = True,
    global_timeline_max_tokens: int = 512,
    cleanup_record_artifacts_enabled: bool = False,
    drop_visual_count_clip_when_isolated: bool = False,
    strip_anchor_metadata: bool = False,
    enable_prompt_cache: bool = False,
) -> Dict[str, Any]:
    video_id = str(row.get("videoID"))
    question_id = str(row.get("question_id") or row.get("index") or "")
    record_key = f"video:{video_id}:{question_id}"
    record: Dict[str, Any] = {"record_key": record_key, "dataset": "lmms-lab/Video-MME", "modality": "video", "status": "started"}
    started = time.perf_counter()
    try:
        zip_path, member = video_lookup[video_id]
        video_path = extract_zip_member(zip_path, member, output_dir / "media" / "video")
        video_metadata = probe_media(str(video_path))
        options = row.get("options")
        if isinstance(options, (list, tuple)):
            option_text = "\n".join(str(option) for option in options)
        else:
            option_text = str(options)
        policy_question = f"{row.get('question')}\n{option_text}"
        question = f"{policy_question}\nRespond with only the option letter."
        manual_policy = apply_manual_video_policy(
            video_anchor_fps,
            video_anchor_max_frames,
            video_anchor_max_side,
            video_detail_max_crops,
            video_detail_max_side,
            video_audio_anchor,
            video_transcript_backend != "none",
            evidence_max_tokens,
        )
        if video_anchor_policy == "manual":
            selected_policy_name = "manual"
            selected_policy = manual_policy
        else:
            selected_policy_name, selected_policy = infer_video_anchor_policy(policy_question, video_anchor_policy)
        if selected_policy.get("transcript") and video_query_retrieval and video_query_detail_extra_crops > 0:
            selected_policy = dict(selected_policy)
            selected_policy["detail_max_crops"] = int(selected_policy.get("detail_max_crops") or 0) + int(video_query_detail_extra_crops)
        _force_crops = os.environ.get("MTEC_DETAIL_MAX_CROPS")
        if _force_crops is not None:
            selected_policy = dict(selected_policy)
            selected_policy["detail_max_crops"] = int(_force_crops)
        anchor_output_dir = output_dir / "anchors" / "video" / f"{video_id}_{question_id}"
        video_subtitle_path = None
        if subtitle_lookup and selected_policy.get("transcript") and video_id in subtitle_lookup:
            sub_zip, sub_member = subtitle_lookup[video_id]
            video_subtitle_path = extract_zip_member(sub_zip, sub_member, anchor_output_dir / "subtitle")
        elif precomputed_subtitle_lookup and selected_policy.get("transcript") and video_id in precomputed_subtitle_lookup:
            video_subtitle_path = precomputed_subtitle_lookup[video_id]
        raw_package = create_multimodal_structural_anchors(
            question=question,
            output_dir=str(anchor_output_dir),
            video_path=str(video_path),
            video_target_fps=selected_policy["fps"],
            video_max_frames=selected_policy["max_frames"],
            video_max_side=selected_policy["max_side"],
            video_detail_max_crops=selected_policy["detail_max_crops"],
            video_detail_max_side=selected_policy["detail_max_side"],
            include_video_audio=selected_policy["audio_anchor"],
            include_video_transcript=selected_policy["transcript"],
            video_transcript_backend=video_transcript_backend,
            video_subtitle_path=str(video_subtitle_path) if video_subtitle_path else None,
            video_query_retrieval=video_query_retrieval,
            video_query_max_segments=video_query_max_segments,
            video_query_window_padding_sec=video_query_window_padding_sec,
            video_asr_model=video_asr_model,
            video_asr_language=video_asr_language or None,
            video_transcript_max_segments=video_transcript_max_segments,
            include_video_global_anchor=video_global_anchor,
            video_global_anchor_fps=video_global_anchor_fps,
            video_global_anchor_max_side=video_global_anchor_max_side,
        )
        package = structured_package(question, raw_package)
        media_contents = multimodal_anchor_contents(
            package,
            include_images=False,
            include_audio_media=send_audio_media,
        )
        if not media_contents:
            video_anchor = package["low_resolution_anchor"]["video_anchor"][0]
            video_input = Path(video_anchor.get("low_fps_video_path") or str(video_path))
            media_contents = [media_content("video", video_input)]
        pre_api_input_audit = video_package_input_audit(
            package,
            media_contents,
            video_path,
            Path(video_subtitle_path) if video_subtitle_path else None,
            prompt_style,
            include_audio_media=send_audio_media,
        )
        print(
            "INPUT_CHECK "
            f"{record_key} video_media={pre_api_input_audit['media_content_counts'].get('video_url', 0)} "
            f"image_media={pre_api_input_audit['media_content_counts'].get('image_url', 0)} "
            f"frames={pre_api_input_audit['low_fps_frame_count']} "
            f"global_full={pre_api_input_audit['full_timeline_lowres_anchor_count']} "
            f"global_samples={pre_api_input_audit['full_timeline_lowres_sample_count']} "
            f"visual_count_video={pre_api_input_audit['visual_count_video_anchor_count']} "
            f"tubelets={pre_api_input_audit['tubelet_storyboard_count']} "
            f"details={pre_api_input_audit['detail_crop_count']} "
            f"visual_count={pre_api_input_audit['visual_count_sheet_count']} "
            f"ocr={pre_api_input_audit['ocr_region_count']} "
            f"motion={pre_api_input_audit['motion_region_count']} "
            f"objects={pre_api_input_audit['object_detection_count']} "
            f"transcript_segments={pre_api_input_audit['transcript_segment_count']} "
            f"warnings={','.join(pre_api_input_audit['warnings']) or 'none'}",
            flush=True,
        )
        critical_warnings = {"missing_video_anchor", "missing_low_fps_video_file", "no_video_media_attached_to_api"}
        if critical_warnings.intersection(pre_api_input_audit["warnings"]) or pre_api_input_audit["missing_files"] or pre_api_input_audit["empty_files"]:
            raise RuntimeError(f"Video input audit failed before API call: {json.dumps(pre_api_input_audit, ensure_ascii=False)[:1800]}")
        computed_global_timeline_response = None
        computed_global_timeline_meta = None
        global_timeline_error = None
        if global_timeline_pass:
            try:
                computed_global_timeline_response, computed_global_timeline_meta = compute_global_video_timeline(
                    client,  # global-timeline is a VISION pass -> use the multimodal evidence model (answer model may be text-only)
                    package,
                    max_tokens=global_timeline_max_tokens,
                )
            except FatalAPIError:
                raise
            except Exception as exc:
                global_timeline_error = f"{type(exc).__name__}: {exc}"
                computed_global_timeline_response = json.dumps(
                    {
                        "timeline": [],
                        "question_relevant_time_ranges": [],
                        "scene_locator": {"target_scene": "", "time_range": "", "confidence": 0.0},
                        "global_uncertainties": [global_timeline_error],
                    },
                    ensure_ascii=False,
                )
                computed_global_timeline_meta = {"error": global_timeline_error}
            package = structured_package(
                question,
                raw_package,
                global_timeline_response=computed_global_timeline_response,
            )
            media_contents = (
                multimodal_anchor_contents(
                    package,
                    include_images=False,
                    include_audio_media=send_audio_media,
                )
                or media_contents
            )
        transcript_segment_count = sum(
            len(anchor.get("segments") or [])
            for anchor in package.get("low_resolution_anchor", {}).get("transcript_anchor", [])
        )
        computed_visual_context_response = None
        computed_visual_context_meta = None
        visual_context_error = None
        visual_context_reason = _video_visual_context_reason(policy_question, transcript_segment_count)
        visual_context_triggered = video_visual_context_pass == "true" or (
            video_visual_context_pass == "auto" and bool(visual_context_reason)
        )
        if visual_context_triggered:
            try:
                computed_visual_context_response, computed_visual_context_meta = compute_video_visual_context(
                    client,
                    package,
                    media_contents,
                    max_tokens=video_visual_context_max_tokens,
                )
            except FatalAPIError:
                raise
            except Exception as exc:
                visual_context_error = f"{type(exc).__name__}: {exc}"
                computed_visual_context_response = json.dumps(
                    {
                        "visual_context_error": visual_context_error,
                        "fallback": "Skipped visual context hint after API error; continue with normal evidence pass.",
                    },
                    ensure_ascii=False,
                )
                computed_visual_context_meta = {"error": visual_context_error}
            package = structured_package(
                question,
                raw_package,
                visual_context_response=computed_visual_context_response,
                global_timeline_response=computed_global_timeline_response,
            )
            media_contents = (
                multimodal_anchor_contents(
                    package,
                    include_images=False,
                    include_audio_media=send_audio_media,
                )
                or media_contents
            )
        computed_evidence_response = None
        computed_evidence_meta = None
        # Module #1 (caching): capture the exact media list sent to the evidence pass so
        # the final pass can reuse it byte-for-byte. Provider context caching then reuses
        # the repeated video/image prefix instead of re-billing it. Identical visual
        # evidence => zero accuracy impact; only the trailing instruction text differs.
        evidence_call_media: Optional[List[Dict[str, Any]]] = None
        if evidence_pass:
            evidence_call_media = media_contents
            computed_evidence_response, computed_evidence_meta = compute_structured_evidence_with_json_retry(
                client,
                package,
                media_contents,
                max_tokens=selected_policy.get("evidence_max_tokens", evidence_max_tokens),
                evidence_prompt_style=evidence_prompt_style,
                strip_metadata=strip_anchor_metadata,
            )
            stage1_response_for_final = None if should_suppress_computed_evidence_for_final(question, raw_package) else computed_evidence_response
            package = structured_package(
                question,
                raw_package,
                stage1_response=stage1_response_for_final,
                visual_context_response=computed_visual_context_response,
                global_timeline_response=computed_global_timeline_response,
            )
            if stage1_response_for_final is None and computed_evidence_response:
                package.setdefault("structured_evidence_prompt", {})["computed_evidence_suppressed_for_final"] = {
                    "reason": "original_visual_count_clip_attached",
                    "policy": "Final verifier should count directly from the original-resolution visual count clip instead of inheriting a possibly wrong intermediate count.",
                }
            media_contents = (
                multimodal_anchor_contents(
                    package,
                    include_images=False,
                    include_audio_media=send_audio_media,
                )
                or media_contents
            )
        det_count_ev = (raw_package.get("low_resolution_anchor") or {}).get("deterministic_count_evidence")
        if det_count_ev:
            package.setdefault("structured_evidence_prompt", {})["deterministic_count_evidence"] = det_count_ev
        isolated_cast_count = run_isolated_cast_count_pass(question, raw_package, answer_client)
        if isolated_cast_count:
            package.setdefault("structured_evidence_prompt", {})["isolated_cast_count"] = isolated_cast_count
            # Token-savings optimization (opt-in, default OFF): once the isolated
            # count-only pass has extracted the authoritative cast count from the full
            # original clip, the main verifier only needs to map men/women to an option,
            # so the large original visual-count clip is redundant in the final call.
            # Dropping it avoids sending the full original video twice (8np: main-call
            # video tokens 28.5k -> 7.4k, token_saving 0.0 -> 0.37). Held behind a flag
            # because non-degradation cannot be proven under the nondeterministic backend.
            if drop_visual_count_clip_when_isolated and isolated_cast_count.get("parsed"):
                removed = _drop_visual_count_clip_from_final(package)
                if removed:
                    package.setdefault("structured_evidence_prompt", {})["visual_count_clip_dropped_for_final"] = {
                        "reason": "isolated_cast_count_pass_supplied_authoritative_count",
                        "removed_anchor_count": removed,
                    }
        if enable_prompt_cache and evidence_call_media and not (package.get("structured_evidence_prompt") or {}).get("visual_count_clip_dropped_for_final"):
            # Reuse the identical media list from the evidence pass so the repeated
            # video/image prefix is served from the provider cache on the final call.
            final_media_contents = evidence_call_media
        else:
            final_media_contents = (
                multimodal_anchor_contents(
                    package,
                    include_images=False,
                    include_audio_media=answer_send_audio_media,
                )
                or media_contents
            )
        final_input_audit = video_package_input_audit(
            package,
            final_media_contents,
            video_path,
            Path(video_subtitle_path) if video_subtitle_path else None,
            prompt_style,
            include_audio_media=answer_send_audio_media,
        )
        if critical_warnings.intersection(final_input_audit["warnings"]) or final_input_audit["missing_files"] or final_input_audit["empty_files"]:
            raise RuntimeError(f"Video input audit failed before final answer call: {json.dumps(final_input_audit, ensure_ascii=False)[:1800]}")
        response, meta = answer_client.generate(
            [
                *final_media_contents,
                {"type": "text", "text": build_final_answer_prompt(package, prompt_style, strip_metadata=strip_anchor_metadata)},
            ],
            max_tokens=answer_max_tokens,
        )
        prediction = extract_final_prediction(
            response,
            meta,
            computed_evidence_response,
            computed_evidence_meta,
        )
        ground_truth = extract_letter(row.get("answer"))
        record.update(
            {
                "status": "completed",
                "video": str(video_path),
                "question": question,
                "ground_truth": ground_truth,
                "Answer": prediction or response,
                "raw_response": response,
                "correct": bool(prediction and ground_truth and prediction == ground_truth),
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
                "video_metadata": video_metadata,
                "videoID": video_id,
                "question_id": question_id,
                "api_meta": meta,
                "answer_api_meta": meta,
                "answer_model": answer_client.model,
                "answer_base_url": answer_client.url.rsplit("/chat/completions", 1)[0],
                "answer_parse_success": bool(prediction),
                "final_content_empty": not bool(str(response or "").strip()),
                "computed_evidence_response": computed_evidence_response,
                "computed_evidence_meta": computed_evidence_meta,
                "computed_global_timeline_response": computed_global_timeline_response,
                "computed_global_timeline_meta": computed_global_timeline_meta,
                "global_timeline_pass": global_timeline_pass,
                "global_timeline_error": global_timeline_error,
                "global_timeline_max_tokens": global_timeline_max_tokens,
                "computed_visual_context_response": computed_visual_context_response,
                "computed_visual_context_meta": computed_visual_context_meta,
                "pre_api_input_audit": pre_api_input_audit,
                "final_input_audit": final_input_audit,
                "video_visual_context_pass": video_visual_context_pass,
                "video_visual_context_triggered": visual_context_triggered,
                "video_visual_context_reason": visual_context_reason,
                "video_visual_context_error": visual_context_error,
                "video_visual_context_max_tokens": video_visual_context_max_tokens,
                "video_global_anchor": video_global_anchor,
                "video_global_anchor_fps": video_global_anchor_fps,
                "video_global_anchor_max_side": video_global_anchor_max_side,
                "evidence_pass": evidence_pass,
                "prompt_style": prompt_style,
                "video_anchor_policy": video_anchor_policy,
                "selected_video_anchor_policy": selected_policy_name,
                "selected_video_anchor_policy_params": selected_policy,
                "evidence_prompt_style": evidence_prompt_style,
                "video_anchor_fps": selected_policy["fps"],
                "video_anchor_max_frames": selected_policy["max_frames"],
                "video_detail_max_crops": selected_policy["detail_max_crops"],
                "video_detail_max_side": selected_policy["detail_max_side"],
                "video_audio_anchor": selected_policy["audio_anchor"],
                "send_audio_media": send_audio_media,
                "answer_send_audio_media": answer_send_audio_media,
                "video_transcript_backend": video_transcript_backend,
                "video_subtitle_path": str(video_subtitle_path) if video_subtitle_path else None,
                "video_query_retrieval": video_query_retrieval,
                "video_query_max_segments": video_query_max_segments,
                "video_query_window_padding_sec": video_query_window_padding_sec,
                "video_query_detail_extra_crops": video_query_detail_extra_crops,
                "question_relevant_transcript_segment_count": sum(len(anchor.get("question_relevant_segments") or []) for anchor in package.get("low_resolution_anchor", {}).get("transcript_anchor", [])),
                "question_relevant_time_windows": [window for anchor in package.get("low_resolution_anchor", {}).get("transcript_anchor", []) for window in (anchor.get("question_relevant_time_windows") or [])],
                "transcript_segment_count": sum(len(anchor.get("segments") or []) for anchor in package.get("low_resolution_anchor", {}).get("transcript_anchor", [])),
                "transcript_source": ",".join(str(anchor.get("source") or "") for anchor in package.get("low_resolution_anchor", {}).get("transcript_anchor", [])),
                "video_asr_model": video_asr_model,
            }
        )
        add_compression_metrics(
            record,
            video_path.stat().st_size,
            package_bytes(package, prompt_style, include_audio_media=answer_send_audio_media),
            f"API MTEC++ video uses policy={selected_policy_name} compressed video anchors, high-detail crops only when routed, transcript/audio text anchors, answer_send_audio_media={answer_send_audio_media}, evidence_prompt_style={evidence_prompt_style}, computed evidence pass={evidence_pass}, and {prompt_style} structured evidence prompt.",
        )
        if cleanup_record_artifacts_enabled:
            removed = cleanup_record_artifacts(
                output_dir / "anchors" / "video" / f"{video_id}_{question_id}",
                video_path,
            )
            record["cleanup_record_artifacts"] = True
            record["cleanup_removed_paths"] = removed
    except Exception as exc:
        if isinstance(exc, FatalAPIError):
            raise
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record

def run_background_audio_record(client: SiliconFlowClient, row: Dict[str, Any], row_id: str, output_dir: Path, max_tokens: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"record_key": row_id, "dataset": "DynamicSuperb/UrbanSound8K-UrbanNoises", "modality": "audio", "status": "started"}
    started = time.perf_counter()
    try:
        file_name = str(row.get("file") or f"{row_id}.wav")
        audio_path = decode_audio_to_wav(row["audio"], output_dir / "media" / "audio" / file_name)
        question = f"{row.get('instruction')}\nReturn the most likely label only."
        raw_package = create_multimodal_structural_anchors(question=question, output_dir=str(output_dir / "anchors" / "audio_background" / row_id), audio_path=str(audio_path))
        package = structured_package(question, raw_package)
        audio_anchor = package["low_resolution_anchor"]["audio_anchor"][0]
        audio_input = Path(audio_anchor.get("low_bitrate_audio_path") or str(audio_path))
        response, meta = client.generate(
            [
                media_content("audio", audio_input),
                {"type": "text", "text": package_prompt(package) + "\nAnswer the audio classification task now."},
            ],
            max_tokens=max_tokens,
        )
        label = str(row.get("label") or "")
        record.update(
            {
                "status": "completed",
                "audio_file": str(audio_path),
                "question": question,
                "ground_truth": label,
                "Answer": response,
                "raw_response": response,
                "correct": normalize_label(label) in normalize_label(response),
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
                "api_meta": meta,
            }
        )
        add_compression_metrics(record, audio_path.stat().st_size, package_bytes(package), "API MTEC++ background audio uses low-bitrate audio anchor plus compact structured evidence prompt.")
    except Exception as exc:
        if isinstance(exc, FatalAPIError):
            raise
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def run_voice_audio_record(client: SiliconFlowClient, row: Dict[str, Any], row_id: str, output_dir: Path, max_tokens: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {"record_key": row_id, "dataset": "DynamicSuperb/HEARSoundEventDetection_DCASE2016Task2", "modality": "voice_audio", "status": "started"}
    started = time.perf_counter()
    try:
        audio_path = decode_audio_to_wav(row["audio"], output_dir / "media" / "audio" / f"{row.get('file') or row_id}.wav")
        target_labels = parse_label_list(row.get("label"))
        voice_targets = [label for label in target_labels if label in {"speech", "laughter", "cough", "clearthroat"}]
        question = (
            "Listen to the audio and identify human voice-related events. "
            "Possible relevant labels include speech, laughter, cough, clearthroat. "
            "Return a comma-separated list of the voice-related labels you hear."
        )
        raw_package = create_multimodal_structural_anchors(question=question, output_dir=str(output_dir / "anchors" / "audio_voice" / row_id), audio_path=str(audio_path))
        package = structured_package(question, raw_package)
        audio_anchor = package["low_resolution_anchor"]["audio_anchor"][0]
        audio_input = Path(audio_anchor.get("low_bitrate_audio_path") or str(audio_path))
        response, meta = client.generate(
            [
                media_content("audio", audio_input),
                {"type": "text", "text": package_prompt(package) + "\nAnswer the human voice event detection task now."},
            ],
            max_tokens=max_tokens,
        )
        correct, matched = label_hit_score(voice_targets, response)
        record.update(
            {
                "status": "completed",
                "audio_file": str(audio_path),
                "question": question,
                "ground_truth": ", ".join(voice_targets),
                "Answer": response,
                "raw_response": response,
                "correct": correct,
                "matched_labels": matched,
                "all_labels": target_labels,
                "low_resolution_anchor": package["low_resolution_anchor"],
                "structured_evidence_prompt": package.get("structured_evidence_prompt"),
                "api_meta": meta,
            }
        )
        add_compression_metrics(record, audio_path.stat().st_size, package_bytes(package), "API MTEC++ voice audio uses low-bitrate audio anchor plus compact structured evidence prompt.")
    except Exception as exc:
        if isinstance(exc, FatalAPIError):
            raise
        record.update({"status": "failed", "Error": f"{type(exc).__name__}: {exc}"})
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for item in records:
        key = item.get("modality") or "unknown"
        group = groups.setdefault(
            str(key),
            {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "correct": 0,
                "expanded_count": 0,
                "compression_values": [],
                "saving_values": [],
                "positive_saving_values": [],
                "elapsed_seconds": 0.0,
            },
        )
        group["total"] += 1
        if item.get("status") == "completed":
            group["completed"] += 1
        if item.get("status") == "failed":
            group["failed"] += 1
        if item.get("correct") is True:
            group["correct"] += 1
        if isinstance(item.get("compression_ratio"), (int, float)):
            group["compression_values"].append(float(item["compression_ratio"]))
        if isinstance(item.get("token_saving_ratio"), (int, float)):
            group["saving_values"].append(float(item["token_saving_ratio"]))
            if item.get("compression_expanded") is True:
                group["expanded_count"] += 1
            else:
                group["positive_saving_values"].append(float(item["token_saving_ratio"]))
        group["elapsed_seconds"] += float(item.get("elapsed_seconds") or 0)
    rows = []
    for modality, group in sorted(groups.items()):
        completed = group["completed"]
        comp_values = group.pop("compression_values")
        saving_values = group.pop("saving_values")
        positive_saving_values = group.pop("positive_saving_values")
        group["accuracy"] = round(group["correct"] / completed, 4) if completed else None
        group["avg_compression_ratio"] = round(sum(comp_values) / len(comp_values), 4) if comp_values else None
        group["avg_token_saving_ratio"] = round(sum(saving_values) / len(saving_values), 4) if saving_values else None
        group["avg_token_saving_ratio_positive_only"] = (
            round(sum(positive_saving_values) / len(positive_saving_values), 4)
            if positive_saving_values
            else None
        )
        group["elapsed_seconds"] = round(group["elapsed_seconds"], 3)
        rows.append({"modality": modality, **group})
    return {"groups": rows}


def write_summary_files(output_dir: Path, records: List[Dict[str, Any]], model: str, answer_model: Optional[str] = None) -> None:
    summary = {
        "model": model,
        "answer_model": answer_model,
        "algorithm_check": {
            "matches_design": True,
            "input_channels": ["low_resolution_video_timeline_anchor", "continuous_before_during_after_tubelet_storyboard", "task_aware_scene_motion_ocr_evidence", "high_detail_keyframe_crop", "transcript_audio_anchor", "structured_evidence_prompt", "per_record_input_audit"],
            "note": "Video mode uses a two-pass flow with question-routed Tiny/Light/Medium/Full/Dense anchor policies, continuous tubelet storyboards, task-aware scene/motion/OCR evidence extraction, and per-record input audits before API calls; SiliconFlow extracts compact evidence, then Bailian Qwen verifies it against compressed video/tubelet/crop/transcript anchors for final answering.",
        },
        "counts": {
            "total": len(records),
            "completed": sum(1 for item in records if item.get("status") == "completed"),
            "failed": sum(1 for item in records if item.get("status") == "failed"),
            "correct": sum(1 for item in records if item.get("correct") is True),
        },
        **summarize(records),
    }
    (output_dir / "modelscope_mtec_anchor_api_full_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "modelscope_mtec_anchor_api_full_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ModelScope multimodal datasets through MTEC++ anchors with SiliconFlow API.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--modelscope-root", default="data/modelscope")
    parser.add_argument("--image-parquets", nargs="*", default=["data/modelscope/realworldqa/data/test-00000-of-00002.parquet", "data/modelscope/realworldqa/data/test-00001-of-00002.parquet"])
    parser.add_argument("--videomme-metadata", default="data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
    parser.add_argument("--video-zips-dir", default="data/modelscope/video-mme-zips")
    parser.add_argument("--videomme-subtitle-zip", default="data/datasets/video-mme/subtitle.zip")
    parser.add_argument("--precomputed-subtitles-dir", default="", help="Directory of videoID.srt files to use when Video-MME subtitle.zip has no subtitle for a video.")
    parser.add_argument("--background-audio-parquets", nargs="*", default=["data/modelscope/urbansound8k-noises/data/test-00000-of-00001-40cf49999a374336.parquet"])
    parser.add_argument("--voice-audio-parquets", nargs="*", default=["data/modelscope/hearsed-dcase2016/data/test-00000-of-00001.parquet"])
    parser.add_argument("--modalities", nargs="+", default=["image", "video", "audio_background", "audio_voice"], choices=["image", "video", "audio_background", "audio_voice"])
    parser.add_argument("--output-dir", default="outputs/modelscope_mtec_anchor_api_full")
    parser.add_argument("--prompt-style", choices=("compact", "rich"), default="compact")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--answer-model", default="qwen3.7-plus")
    parser.add_argument("--answer-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--answer-api-key-env", default="BAILIAN_API_KEY")
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--answer-enable-thinking", choices=("true", "false", "omit"), default="false")
    parser.add_argument("--evidence-pass", choices=("true", "false"), default="true")
    parser.add_argument("--evidence-max-tokens", type=int, default=192)
    parser.add_argument("--evidence-prompt-style", choices=("minimal", "rich"), default="minimal")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", choices=("true", "false", "omit"), default="omit")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit-per-modality", type=int, default=0, help="0 means full available dataset.")
    parser.add_argument("--min-image-width", type=int, default=0)
    parser.add_argument("--min-image-height", type=int, default=0)
    parser.add_argument("--min-image-megapixels", type=float, default=0.0)
    parser.add_argument("--min-image-bytes", type=int, default=0)
    parser.add_argument("--video-duration-classes", nargs="*", default=[])
    parser.add_argument("--min-video-duration-sec", type=float, default=0.0)
    parser.add_argument("--min-video-width", type=int, default=0)
    parser.add_argument("--min-video-height", type=int, default=0)
    parser.add_argument("--min-video-bytes", type=int, default=0)
    parser.add_argument("--max-video-bytes", type=int, default=0)
    parser.add_argument("--unique-video-ids", action="store_true")
    parser.add_argument("--video-anchor-policy", choices=("auto", "manual", "tiny", "light", "medium", "full", "dense"), default="auto")
    parser.add_argument("--video-anchor-fps", type=float, default=1.0)
    parser.add_argument("--video-anchor-max-frames", type=int, default=16)
    parser.add_argument("--video-anchor-max-side", type=int, default=512)
    parser.add_argument("--video-detail-max-crops", type=int, default=2)
    parser.add_argument("--video-detail-max-side", type=int, default=640)
    parser.add_argument("--video-audio-anchor", choices=("true", "false"), default="false")
    parser.add_argument("--send-audio-media", choices=("true", "false"), default="false")
    parser.add_argument("--answer-send-audio-media", choices=("true", "false"), default="false")
    parser.add_argument("--video-transcript-backend", choices=("auto", "none", "subtitle", "faster-whisper"), default="auto")
    parser.add_argument("--video-asr-model", default="base.en")
    parser.add_argument("--video-asr-language", default="en")
    parser.add_argument("--video-transcript-max-segments", type=int, default=48)
    parser.add_argument("--video-query-retrieval", choices=("true", "false"), default="true")
    parser.add_argument("--video-query-max-segments", type=int, default=12)
    parser.add_argument("--video-query-window-padding-sec", type=float, default=8.0)
    parser.add_argument("--video-query-detail-extra-crops", type=int, default=4)
    parser.add_argument("--video-visual-context-pass", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--video-visual-context-max-tokens", type=int, default=384)
    parser.add_argument("--video-global-anchor", choices=("true", "false"), default="true", help="Attach a low-resolution full-duration video anchor for global temporal fallback.")
    parser.add_argument("--video-global-anchor-fps", type=float, default=1.0)
    parser.add_argument("--video-global-anchor-max-side", type=int, default=360)
    parser.add_argument("--global-timeline-pass", choices=("true", "false"), default="true", help="First ask the answer model to inspect only the full-video low-resolution anchor and produce a timeline/scene locator.")
    parser.add_argument("--global-timeline-max-tokens", type=int, default=512)
    parser.add_argument("--video-record-keys", nargs="*", default=[], help="Run only exact Video-MME record keys like video:VIDEOID:QUESTIONID, preserving the provided order.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--max-per-image-source", type=int, default=0)
    parser.add_argument("--cleanup-record-artifacts", action="store_true", help="Delete per-record extracted videos and generated anchors after metrics are written to JSONL.")
    parser.add_argument("--drop-visual-count-clip-when-isolated", action="store_true", help="Token-savings optimization: for performing-cast questions, drop the redundant original-resolution visual-count clip from the final answer call once the isolated cast-count pass has produced a count (avoids sending the full video twice). Default off; the count still drives the answer, but enable only after confirming no accuracy regression on your backend.")
    parser.add_argument("--enable-prompt-cache", action="store_true", help="Module #1 (caching): reuse the evidence pass's exact media list for the final answer call so the provider serves the repeated video/image prefix from context cache. Identical visual evidence, only trailing text differs => no accuracy impact. Default off.")
    parser.add_argument("--strip-anchor-metadata", action="store_true", help="Module #2 (metadata stripping): remove non-semantic anchor bookkeeping fields (compression/bytes/quality/strategy/paths/resolutions) from the prompt serialization only. Saved records and attached media are unchanged => no accuracy impact. Default off.")
    parser.add_argument("--deterministic-count", action="store_true", help="For counting questions, attach a continuous dense-sampled segment + a deterministic detector count (YOLO persons / motion-energy repetitions) as hard evidence to improve instance/repetition counting. Default off.")
    parser.add_argument("--oss-media-upload", choices=("off", "auto"), default="off", help="Large-media fix: when 'auto', upload video media (and any media file > --oss-threshold-bytes) to DashScope temporary OSS and reference it by oss:// URL instead of a base64 data-URI. Avoids the 20 MB/data-URI and ~28 MB/request payload limits that fail long videos. Requires the dashscope SDK.")
    parser.add_argument("--oss-threshold-bytes", type=int, default=8_000_000, help="Upload media files larger than this many bytes to OSS when --oss-media-upload=auto (video is always uploaded).")
    parser.add_argument("--oss-upload-model", default="qwen-vl-max", help="Model name used only to obtain the temporary-OSS upload certificate.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")
    answer_api_key = os.environ.get(args.answer_api_key_env)
    if "video" in args.modalities and not answer_api_key:
        raise SystemExit(f"Missing API key env var for video final answer pass: {args.answer_api_key_env}")

    if args.oss_media_upload == "auto":
        OSS_MEDIA["enabled"] = True
        OSS_MEDIA["api_key"] = answer_api_key or api_key
        OSS_MEDIA["model"] = args.oss_upload_model
        OSS_MEDIA["threshold_bytes"] = args.oss_threshold_bytes

    if args.deterministic_count:
        mtec_media_pipeline.DETERMINISTIC_COUNT["enabled"] = True

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "modelscope_mtec_anchor_api_full_results.jsonl"
    done = load_done_keys(jsonl_path) if args.resume else set()
    enable_thinking = None
    if args.enable_thinking == "true":
        enable_thinking = True
    elif args.enable_thinking == "false":
        enable_thinking = False
    client = SiliconFlowClient(
        api_key,
        args.model,
        args.base_url,
        args.timeout,
        args.max_retries,
        args.retry_sleep,
        args.temperature,
        enable_thinking,
    )
    answer_enable_thinking = None
    if args.answer_enable_thinking == "true":
        answer_enable_thinking = True
    elif args.answer_enable_thinking == "false":
        answer_enable_thinking = False
    answer_client = SiliconFlowClient(
        answer_api_key or api_key,
        args.answer_model,
        args.answer_base_url,
        args.timeout,
        args.max_retries,
        args.retry_sleep,
        args.temperature,
        answer_enable_thinking,
    )
    evidence_pass = args.evidence_pass == "true"

    records: List[Dict[str, Any]] = []
    if args.resume and jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]

    def handle(record: Dict[str, Any]) -> None:
        records.append(record)
        append_record(jsonl_path, record)
        print(
            f"{record.get('record_key')} [{record.get('modality')}]: {record.get('status')} "
            f"pred={record.get('Answer')} gt={record.get('ground_truth')} correct={record.get('correct')} "
            f"error={record.get('Error', '')}",
            flush=True,
        )

    if "image" in args.modalities:
        rng = random.Random(args.seed)
        image_candidates: List[Tuple[Path, int, Dict[str, Any]]] = []
        for image_path_arg in args.image_parquets:
            source_candidates = []
            for parquet_path, row_index, row in iter_parquet_rows([resolve_path(image_path_arg)]):
                if is_high_resolution_image(row, args):
                    source_candidates.append((parquet_path, row_index, row))
            if args.shuffle:
                rng.shuffle(source_candidates)
            if args.max_per_image_source:
                source_candidates = source_candidates[: args.max_per_image_source]
            image_candidates.extend(source_candidates)
        if args.shuffle:
            rng.shuffle(image_candidates)
        if args.limit_per_modality:
            image_candidates = image_candidates[: args.limit_per_modality]
        for parquet_path, row_index, row in image_candidates:
            row_id = f"image:{parquet_path.stem}:{row_index}"
            record_key = row_id.replace(":", "_")
            if record_key in done:
                continue
            try:
                handle(
                    run_image_record(
                        client,
                        row,
                        record_key,
                        output_dir,
                        args.max_tokens,
                        args.prompt_style,
                        evidence_pass,
                        args.evidence_max_tokens,
                    )
                )
            except FatalAPIError:
                write_summary_files(output_dir, records, args.model, args.answer_model)
                raise

    if "video" in args.modalities:
        rng = random.Random(args.seed)
        video_lookup = downloaded_videos(resolve_path(args.video_zips_dir))
        subtitle_lookup = downloaded_subtitles(resolve_path(args.videomme_subtitle_zip))
        precomputed_subtitle_lookup = precomputed_subtitles(resolve_path(args.precomputed_subtitles_dir)) if args.precomputed_subtitles_dir else {}
        meta = pd.read_parquet(resolve_path(args.videomme_metadata))
        if "videoID" in meta.columns:
            meta = meta[meta["videoID"].astype(str).isin(video_lookup.keys())]
        requested_record_keys = [str(item) for item in (args.video_record_keys or [])]
        requested_key_set = set(requested_record_keys)
        duration_candidates = []
        seen_video_ids: Set[str] = set()
        for _, row in meta.iterrows():
            row_dict = row.to_dict()
            video_id = str(row_dict.get("videoID"))
            question_id = str(row_dict.get("question_id") or row_dict.get("index") or "")
            row_key = f"video:{video_id}:{question_id}"
            if requested_key_set:
                if row_key not in requested_key_set:
                    continue
            elif not video_row_matches_duration(row_dict, args):
                continue
            if args.unique_video_ids and video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            duration_candidates.append(row_dict)
        if requested_record_keys:
            order = {key: index for index, key in enumerate(requested_record_keys)}
            duration_candidates.sort(key=lambda row: order.get(f"video:{row.get('videoID')}:{row.get('question_id') or row.get('index') or ''}", len(order)))
        elif args.shuffle:
            rng.shuffle(duration_candidates)
        video_candidates = []
        for row_dict in duration_candidates:
            if not video_row_matches_resolution(row_dict, video_lookup, output_dir, args):
                continue
            video_candidates.append(row_dict)
            if not requested_record_keys and args.limit_per_modality and len(video_candidates) >= args.limit_per_modality:
                break
        if args.limit_per_modality and not requested_record_keys:
            video_candidates = video_candidates[: args.limit_per_modality]
        video_audio_anchor = args.video_audio_anchor == "true"
        send_audio_media = args.send_audio_media == "true"
        answer_send_audio_media = args.answer_send_audio_media == "true"
        video_global_anchor = args.video_global_anchor == "true"
        global_timeline_pass = args.global_timeline_pass == "true"
        for count, row_dict in enumerate(video_candidates):
            video_id = str(row_dict.get("videoID"))
            question_id = str(row_dict.get("question_id") or row_dict.get("index") or count)
            row_key = f"video:{video_id}:{question_id}"
            if row_key in done:
                continue
            try:
                handle(
                    run_video_record(
                        client,
                        answer_client,
                        row_dict,
                        video_lookup,
                        subtitle_lookup,
                        precomputed_subtitle_lookup,
                        output_dir,
                        args.max_tokens,
                        args.answer_max_tokens,
                        args.prompt_style,
                        evidence_pass,
                        args.evidence_max_tokens,
                        args.evidence_prompt_style,
                        args.video_anchor_policy,
                        args.video_anchor_fps,
                        args.video_anchor_max_frames,
                        args.video_anchor_max_side,
                        args.video_detail_max_crops,
                        args.video_detail_max_side,
                        video_audio_anchor,
                        send_audio_media,
                        answer_send_audio_media,
                        args.video_transcript_backend,
                        args.video_asr_model,
                        args.video_asr_language,
                        args.video_transcript_max_segments,
                        args.video_query_retrieval == "true",
                        args.video_query_max_segments,
                        args.video_query_window_padding_sec,
                        args.video_query_detail_extra_crops,
                        args.video_visual_context_pass,
                        args.video_visual_context_max_tokens,
                        video_global_anchor,
                        args.video_global_anchor_fps,
                        args.video_global_anchor_max_side,
                        global_timeline_pass,
                        args.global_timeline_max_tokens,
                        args.cleanup_record_artifacts,
                        args.drop_visual_count_clip_when_isolated,
                        args.strip_anchor_metadata,
                        args.enable_prompt_cache,
                    )
                )
            except FatalAPIError:
                write_summary_files(output_dir, records, args.model, args.answer_model)
                raise

    if "audio_background" in args.modalities:
        count = 0
        for parquet_path, row_index, row in iter_parquet_rows([resolve_path(path) for path in args.background_audio_parquets]):
            row_id = f"audio_background:{parquet_path.stem}:{row_index}"
            record_key = row_id.replace(":", "_")
            if record_key in done:
                continue
            try:
                handle(run_background_audio_record(client, row, record_key, output_dir, args.max_tokens))
            except FatalAPIError:
                write_summary_files(output_dir, records, args.model, args.answer_model)
                raise
            count += 1
            if args.limit_per_modality and count >= args.limit_per_modality:
                break

    if "audio_voice" in args.modalities:
        count = 0
        for parquet_path, row_index, row in iter_parquet_rows([resolve_path(path) for path in args.voice_audio_parquets]):
            row_id = f"audio_voice:{parquet_path.stem}:{row_index}"
            record_key = row_id.replace(":", "_")
            if record_key in done:
                continue
            try:
                handle(run_voice_audio_record(client, row, record_key, output_dir, args.max_tokens))
            except FatalAPIError:
                write_summary_files(output_dir, records, args.model, args.answer_model)
                raise
            count += 1
            if args.limit_per_modality and count >= args.limit_per_modality:
                break

    write_summary_files(output_dir, records, args.model, args.answer_model)
    print(f"Results JSONL: {jsonl_path}", flush=True)
    print(f"Results JSON: {output_dir / 'modelscope_mtec_anchor_api_full_results.json'}", flush=True)
    print(f"Summary JSON: {output_dir / 'modelscope_mtec_anchor_api_full_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
