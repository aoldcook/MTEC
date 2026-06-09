import io
import json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


DEFAULT_TOTAL_BUDGET = 1000
DEFAULT_GLOBAL_ANCHOR_MAX_SIDE = 768
DEFAULT_GLOBAL_ANCHOR_QUALITY = 82


def _resampling_method():
    return Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _rounded_bbox(bbox: Optional[List[float]]) -> Optional[List[float]]:
    if bbox is None:
        return None
    return [round(max(0.0, min(1.0, float(value))), 4) for value in bbox]


def _image_size_from_bytes(image_bytes: bytes) -> Dict[str, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    return {"width": width, "height": height}


def create_image_global_anchor(
    image_path: str,
    max_side: int = DEFAULT_GLOBAL_ANCHOR_MAX_SIDE,
    quality: int = DEFAULT_GLOBAL_ANCHOR_QUALITY,
    anchor_id: str = "image_anchor_global",
) -> Tuple[bytes, str, Dict[str, Any]]:
    """
    Create the low-resolution whole-image structural anchor used by MTEC-Prompt++.
    The anchor keeps global layout while keeping image budget lower than the
    original input.
    """
    with Image.open(image_path) as image:
        source_width, source_height = image.size
        anchor_image = _to_rgb(image.copy())
        anchor_image.thumbnail((max_side, max_side), _resampling_method())
        anchor_bytes = _jpeg_bytes(anchor_image, quality)
        anchor_width, anchor_height = anchor_image.size

    metadata = {
        "anchor_id": anchor_id,
        "anchor_link": anchor_id,
        "type": "image_global_low",
        "role": "Preserve whole-image spatial layout and scene context.",
        "source_resolution": {"width": source_width, "height": source_height},
        "resolution": {"width": anchor_width, "height": anchor_height},
        "compression": {
            "max_side": max_side,
            "format": "jpeg",
            "quality": quality,
            "bytes": len(anchor_bytes),
        },
    }
    return anchor_bytes, "image/jpeg", metadata


def build_image_crop_anchor_metadata(
    crop_bytes: bytes,
    bbox_norm: Optional[List[float]],
    expanded_bbox_norm: Optional[List[float]],
    original_width: int,
    original_height: int,
    anchor_id: str = "image_anchor_crop_1",
) -> Dict[str, Any]:
    crop_size = _image_size_from_bytes(crop_bytes)
    return {
        "anchor_id": anchor_id,
        "anchor_link": anchor_id,
        "type": "image_crop_detail",
        "role": "Preserve high-value local detail selected from the question-relevant region.",
        "source_resolution": {"width": original_width, "height": original_height},
        "resolution": crop_size,
        "bbox_norm": _rounded_bbox(bbox_norm),
        "expanded_bbox_norm": _rounded_bbox(expanded_bbox_norm),
        "compression": {
            "format": "jpeg",
            "bytes": len(crop_bytes),
        },
    }


def infer_question_profile(question: str) -> Dict[str, Any]:
    text = (question or "").lower()

    ocr_keywords = (
        "text", "word", "number", "digit", "label", "sign", "table",
        "chart", "document", "read", "ocr", "formula", "button",
    )
    spatial_keywords = (
        "where", "position", "located", "left", "right", "above", "below",
        "near", "beside", "behind", "front", "between", "layout", "distance",
    )
    temporal_keywords = (
        "before", "after", "sequence", "order", "happen", "event", "time",
        "motion", "move", "change", "video", "frame",
    )
    audio_keywords = (
        "audio", "sound", "voice", "speech", "asr", "alarm", "noise",
        "music", "speaker",
    )

    needs_ocr = any(keyword in text for keyword in ocr_keywords)
    needs_spatial = any(keyword in text for keyword in spatial_keywords)
    needs_temporal = any(keyword in text for keyword in temporal_keywords)
    needs_audio = any(keyword in text for keyword in audio_keywords)

    if needs_audio and needs_temporal:
        question_type = "multimodal_sync_or_event"
    elif needs_temporal:
        question_type = "temporal_reasoning"
    elif needs_ocr:
        question_type = "ocr_or_detail_reading"
    elif needs_spatial:
        question_type = "spatial_reasoning"
    else:
        question_type = "visual_semantic_reasoning"

    required_modalities = ["visual"]
    if needs_temporal:
        required_modalities.append("temporal")
    if needs_audio:
        required_modalities.append("audio")

    anchor_priority = ["image_anchor_global"]
    if needs_ocr or needs_spatial:
        anchor_priority.append("image_anchor_crop")
    if needs_temporal:
        anchor_priority.append("video_anchor")
    if needs_audio:
        anchor_priority.append("audio_anchor")

    return {
        "question_type": question_type,
        "required_modalities": required_modalities,
        "anchor_priority": anchor_priority,
        "needs": {
            "ocr_or_detail": needs_ocr,
            "spatial": needs_spatial,
            "temporal": needs_temporal,
            "audio": needs_audio,
        },
    }


def allocate_budget(question: str, total_budget: int = DEFAULT_TOTAL_BUDGET) -> Dict[str, Any]:
    profile = infer_question_profile(question)
    needs = profile["needs"]

    anchor_ratio = 0.20
    if needs["spatial"]:
        anchor_ratio = 0.30
    elif needs["ocr_or_detail"]:
        anchor_ratio = 0.25
    elif needs["temporal"] or needs["audio"]:
        anchor_ratio = 0.25
    elif profile["question_type"] == "visual_semantic_reasoning":
        anchor_ratio = 0.15

    anchor_budget = int(round(total_budget * anchor_ratio))
    evidence_budget = max(0, total_budget - anchor_budget)
    return {
        "total_budget": total_budget,
        "anchor_budget": anchor_budget,
        "evidence_budget": evidence_budget,
        "anchor_ratio": round(anchor_ratio, 3),
        "evidence_ratio": round(1.0 - anchor_ratio, 3),
        "policy": "Dynamic image-anchor budget derived from the question profile.",
    }


def build_low_resolution_anchor_package(
    question: str,
    global_anchor: Optional[Dict[str, Any]],
    crop_anchor: Optional[Dict[str, Any]] = None,
    video_anchor: Optional[Any] = None,
    audio_anchor: Optional[Any] = None,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
) -> Dict[str, Any]:
    image_anchors: List[Dict[str, Any]] = []
    if global_anchor:
        image_anchors.append(global_anchor)
    if crop_anchor:
        image_anchors.append(crop_anchor)

    video_anchors = _as_anchor_list(video_anchor)
    audio_anchors = _as_anchor_list(audio_anchor)

    return {
        "low_resolution_anchor": {
            "image_anchor": image_anchors,
            "video_anchor": video_anchors,
            "audio_anchor": audio_anchors,
        },
        "compression_target": {
            "question_profile": infer_question_profile(question),
            "budget": allocate_budget(question, total_budget),
        },
    }


def _as_anchor_list(anchor: Optional[Any]) -> List[Dict[str, Any]]:
    if anchor is None:
        return []
    if isinstance(anchor, list):
        return [item for item in anchor if item]
    return [anchor]


def _trim_text(text: Optional[str], max_chars: int = 900) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_structured_evidence_prompt(
    question: str,
    stage1_response: Optional[str],
    bbox_norm: Optional[List[float]],
    expanded_bbox_norm: Optional[List[float]],
    global_anchor: Optional[Dict[str, Any]],
    crop_anchor: Optional[Dict[str, Any]] = None,
    video_anchor: Optional[Any] = None,
    audio_anchor: Optional[Any] = None,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
) -> Dict[str, Any]:
    anchor_package = build_low_resolution_anchor_package(
        question=question,
        global_anchor=global_anchor,
        crop_anchor=crop_anchor,
        video_anchor=video_anchor,
        audio_anchor=audio_anchor,
        total_budget=total_budget,
    )

    visual_evidence = []
    if global_anchor:
        visual_evidence.append(
            {
                "time": "image",
                "region": "whole_image",
                "source": "low_resolution_global_anchor",
                "content": "Whole-image low-resolution anchor preserves global layout and scene context.",
                "anchor_link": global_anchor["anchor_link"],
                "reason": "Prevents local crop evidence from losing surrounding context.",
                "confidence": 1.0,
            }
        )

    if bbox_norm:
        visual_evidence.append(
            {
                "time": "image",
                "region": {
                    "bbox_norm": _rounded_bbox(bbox_norm),
                    "expanded_bbox_norm": _rounded_bbox(expanded_bbox_norm),
                },
                "source": "localized_zoom_crop",
                "content": "Stage 1 selected this region as question-relevant local evidence.",
                "anchor_link": (
                    crop_anchor["anchor_link"]
                    if crop_anchor
                    else global_anchor["anchor_link"]
                    if global_anchor
                    else None
                ),
                "reason": "Grounds local detail evidence while the global anchor keeps spatial context.",
                "confidence": 0.8,
            }
        )

    video_anchors = anchor_package["low_resolution_anchor"]["video_anchor"]
    audio_anchors = anchor_package["low_resolution_anchor"]["audio_anchor"]
    temporal_chain = _build_temporal_chain(video_anchors)
    audio_evidence = _build_audio_evidence(audio_anchors)
    cross_modal_relations = _build_cross_modal_relations(video_anchors, audio_anchors)

    uncertainty = []
    if not video_anchors:
        uncertainty.append("No video anchor was supplied for this item.")
    if not audio_anchors:
        uncertainty.append("No audio anchor was supplied for this item.")
    if not bbox_norm:
        uncertainty.append("No valid localized crop anchor was produced because no bounding box was parsed.")

    structured_prompt = {
        "task": "Answer the user question using both low-resolution structural anchors and structured evidence.",
        "user_question": question,
        "compression_policy": {
            "anchor_policy": "Preserve low-cost visual structure through a whole-image anchor and selected local crop anchors.",
            "evidence_policy": "Preserve task-relevant, low-redundancy, anchor-grounded evidence.",
        },
        "global_summary": _trim_text(stage1_response),
        "temporal_chain": temporal_chain,
        "visual_evidence": visual_evidence,
        "audio_evidence": audio_evidence,
        "ocr_asr_evidence": [],
        "cross_modal_relations": cross_modal_relations,
        "merged_duplicate_evidence": [],
        "uncertainty": uncertainty,
    }

    return {
        **anchor_package,
        "structured_evidence_prompt": structured_prompt,
    }


def format_structured_evidence_prompt(prompt: Dict[str, Any], compact: bool = False) -> str:
    if compact:
        return format_compact_evidence_prompt(prompt)
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def format_compact_evidence_prompt(prompt: Dict[str, Any]) -> str:
    """Format model-facing evidence without dropping evidence details.

    The full JSON object is useful for logging and analysis, but sending all
    debug metadata, repeated field names, paths, bytes, and resolutions to the
    model is wasteful. This compact form keeps the evidence-bearing details:
    questions, timestamps, anchor links, change scores, audio event scores,
    visual regions, relations, and uncertainty notes.
    """
    structured = prompt.get("structured_evidence_prompt", prompt)
    anchors = prompt.get("low_resolution_anchor", {})
    lines = [
        "MTEC++ Evidence (compact; fields keep anchor grounding).",
        f"Q: {_one_line(structured.get('user_question'))}",
    ]

    summary = _one_line(structured.get("global_summary"))
    if summary:
        lines.append(f"Summary: {summary}")

    temporal_rows = _compact_temporal_rows(structured, anchors)
    if temporal_rows:
        lines.append("TemporalChain t|anchor|delta|note:")
        lines.extend(temporal_rows)

    visual_rows = _compact_visual_rows(structured)
    if visual_rows:
        lines.append("VisualEvidence time|region|anchor|conf|content|reason:")
        lines.extend(visual_rows)

    audio_rows = _compact_audio_rows(structured, anchors)
    if audio_rows:
        lines.append("AudioEvidence time|type|anchor|score|rms|content:")
        lines.extend(audio_rows)

    relations = [_one_line(item) for item in structured.get("cross_modal_relations", []) if item]
    if relations:
        lines.append("Relations:")
        lines.extend(f"- {item}" for item in relations)

    merged = [_one_line(item) for item in structured.get("merged_duplicate_evidence", []) if item]
    if merged:
        lines.append("MergedDuplicates:")
        lines.extend(f"- {item}" for item in merged)

    uncertainty = [_one_line(item) for item in structured.get("uncertainty", []) if item]
    if uncertainty:
        lines.append("Uncertainty:")
        lines.extend(f"- {item}" for item in uncertainty)

    return "\n".join(lines)


def _build_temporal_chain(video_anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chain: List[Dict[str, Any]] = []
    for anchor in video_anchors:
        for frame in anchor.get("frames", []):
            chain.append(
                {
                    "time": f"{frame.get('time_sec', 0.0):.2f}s",
                    "content": frame.get("summary", "Representative low-FPS video frame."),
                    "anchor_link": frame.get("anchor_link"),
                    "source": anchor.get("anchor_id"),
                    "change_score": frame.get("change_score"),
                }
            )
    return sorted(chain, key=lambda item: _safe_time_value(item["time"]))


def _build_audio_evidence(audio_anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for anchor in audio_anchors:
        for segment in anchor.get("audio_event_segments", []):
            evidence.append(
                {
                    "time": segment.get("time"),
                    "content": segment.get("content", "Detected acoustic event segment."),
                    "anchor_link": segment.get("anchor_link"),
                    "source": anchor.get("anchor_id"),
                    "event_type": segment.get("event_type"),
                    "score": segment.get("score"),
                }
            )
    return evidence


def _build_cross_modal_relations(
    video_anchors: List[Dict[str, Any]],
    audio_anchors: List[Dict[str, Any]],
) -> List[str]:
    if not video_anchors or not audio_anchors:
        return []
    frames: List[Dict[str, Any]] = []
    for anchor in video_anchors:
        for frame in anchor.get("frames", []):
            if frame.get("anchor_link") and frame.get("time_sec") is not None:
                frames.append(frame)

    events: List[Dict[str, Any]] = []
    for anchor in audio_anchors:
        for event in anchor.get("audio_event_segments", []):
            if event.get("anchor_link"):
                events.append(event)

    if not frames or not events:
        return [
            "Video frame anchors and audio event anchors share second-level timestamps; use anchor_link values to align visual changes with acoustic events."
        ]

    relations = []
    for event in events[:8]:
        event_mid = _event_midpoint(event)
        nearest = min(
            frames,
            key=lambda frame: abs(float(frame.get("time_sec") or 0.0) - event_mid),
        )
        delta = abs(float(nearest.get("time_sec") or 0.0) - event_mid)
        relations.append(
            (
                f"{event.get('anchor_link')}[{event.get('time')}] aligns with "
                f"{nearest.get('anchor_link')}[{float(nearest.get('time_sec') or 0.0):.2f}s]; "
                f"delta={delta:.2f}s; audio_type={event.get('event_type')}; "
                f"audio_score={_short_number(event.get('score'))}; "
                f"visual_change={_short_number(nearest.get('change_score'))}"
            )
        )
    return relations


def _event_midpoint(event: Dict[str, Any]) -> float:
    start = event.get("start_sec")
    end = event.get("end_sec")
    try:
        if start is not None and end is not None:
            return (float(start) + float(end)) / 2.0
        if start is not None:
            return float(start)
    except (TypeError, ValueError):
        pass
    return _safe_time_range_midpoint(str(event.get("time") or "0"))


def _safe_time_range_midpoint(value: str) -> float:
    try:
        value = value.rstrip("s")
        if "-" in value:
            start, end = value.split("-", 1)
            return (float(start) + float(end)) / 2.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_time_value(value: str) -> float:
    try:
        return float(value.rstrip("s"))
    except (TypeError, ValueError):
        return 0.0


def _compact_temporal_rows(
    structured: Dict[str, Any],
    anchors: Dict[str, Any],
) -> List[str]:
    frame_index: Dict[str, Dict[str, Any]] = {}
    for video_anchor in anchors.get("video_anchor", []):
        for frame in video_anchor.get("frames", []):
            if frame.get("anchor_link"):
                frame_index[frame["anchor_link"]] = frame

    rows = []
    for item in structured.get("temporal_chain", []):
        anchor_link = item.get("anchor_link") or ""
        frame = frame_index.get(anchor_link, {})
        change_score = item.get("change_score", frame.get("change_score"))
        frame_number = frame.get("frame_index")
        note = item.get("content") or frame.get("summary") or ""
        if frame_number is not None:
            note = f"frame={frame_number}; {note}"
        rows.append(
            "|".join(
                [
                    str(item.get("time", "")),
                    anchor_link,
                    _short_number(change_score),
                    _one_line(note),
                ]
            )
        )
    return rows


def _compact_visual_rows(structured: Dict[str, Any]) -> List[str]:
    rows = []
    for item in structured.get("visual_evidence", []):
        region = item.get("region")
        if isinstance(region, dict):
            region_text = json.dumps(region, ensure_ascii=False, separators=(",", ":"))
        else:
            region_text = str(region or "")
        rows.append(
            "|".join(
                [
                    str(item.get("time", "")),
                    _one_line(region_text),
                    str(item.get("anchor_link") or ""),
                    _short_number(item.get("confidence")),
                    _one_line(item.get("content")),
                    _one_line(item.get("reason")),
                ]
            )
        )
    return rows


def _compact_audio_rows(
    structured: Dict[str, Any],
    anchors: Dict[str, Any],
) -> List[str]:
    event_index: Dict[str, Dict[str, Any]] = {}
    for audio_anchor in anchors.get("audio_anchor", []):
        for event in audio_anchor.get("audio_event_segments", []):
            if event.get("anchor_link"):
                event_index[event["anchor_link"]] = event

    rows = []
    for item in structured.get("audio_evidence", []):
        anchor_link = item.get("anchor_link") or ""
        event = event_index.get(anchor_link, {})
        rows.append(
            "|".join(
                [
                    str(item.get("time", event.get("time", ""))),
                    str(item.get("event_type", event.get("event_type", ""))),
                    anchor_link,
                    _short_number(item.get("score", event.get("score"))),
                    _short_number(event.get("rms")),
                    _one_line(item.get("content", event.get("content", ""))),
                ]
            )
        )
    return rows


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _short_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)
