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
        "color", "logo", "letter", "screen", "caption", "small",
    )
    spatial_keywords = (
        "where", "position", "located", "left", "right", "above", "below",
        "near", "beside", "behind", "front", "between", "layout", "distance",
        "side", "direction", "facing", "relative",
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
    needs_counting = any(keyword in text for keyword in ("how many", "number of", "count", "many"))

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
            "counting": needs_counting,
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
    global_anchor: Optional[Any],
    crop_anchor: Optional[Dict[str, Any]] = None,
    video_anchor: Optional[Any] = None,
    audio_anchor: Optional[Any] = None,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
) -> Dict[str, Any]:
    image_anchors: List[Dict[str, Any]] = _as_anchor_list(global_anchor)
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

    image_anchors = anchor_package["low_resolution_anchor"]["image_anchor"]
    visual_evidence = []
    for image_anchor in image_anchors:
        if image_anchor.get("type") == "image_global_low":
            content = "Whole-image low-resolution anchor preserves global layout and scene context."
            reason = "Prevents local crop evidence from losing surrounding context."
            region: Any = "whole_image"
            confidence = 1.0
            time_value = "image"
        elif image_anchor.get("type") == "video_keyframe_detail_crop":
            content = "High-resolution video keyframe crop preserves detail that may be lost in the low-FPS video anchor."
            reason = "Use this crop for OCR, screen content, numbers, small objects, attributes, and frame-local state changes."
            region = {
                "bbox_norm": image_anchor.get("bbox_norm"),
                "expanded_bbox_norm": image_anchor.get("expanded_bbox_norm"),
                "region_hint": image_anchor.get("region_hint"),
                "linked_video_anchor": image_anchor.get("linked_video_anchor"),
                "frame_index": image_anchor.get("frame_index"),
                "question_relevant": image_anchor.get("question_relevant"),
                "query_window": image_anchor.get("query_window"),
                "selection_reason": image_anchor.get("selection_reason"),
            }
            confidence = 0.95 if image_anchor.get("question_relevant") else 0.9
            time_value = f"{float(image_anchor.get('time_sec') or 0.0):.2f}s"
        else:
            content = "Question-conditioned detail crop preserves local evidence that may be lost in the global anchor."
            reason = "Use this crop for small objects, text, counts, attributes, or spatial relations in its bbox."
            region = {
                "bbox_norm": image_anchor.get("bbox_norm"),
                "expanded_bbox_norm": image_anchor.get("expanded_bbox_norm"),
                "region_hint": image_anchor.get("region_hint"),
            }
            confidence = 0.85
            time_value = "image"
        visual_evidence.append(
            {
                "time": time_value,
                "region": region,
                "source": "low_resolution_global_anchor"
                if image_anchor.get("type") == "image_global_low"
                else "video_keyframe_detail_crop"
                if image_anchor.get("type") == "video_keyframe_detail_crop"
                else "question_conditioned_detail_crop",
                "content": content,
                "anchor_link": image_anchor.get("anchor_link"),
                "reason": reason,
                "confidence": confidence,
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
                    else image_anchors[0]["anchor_link"]
                    if image_anchors
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

    computed_evidence = _coerce_computed_evidence(stage1_response)
    structured_prompt = {
        "task": "Answer the user question using both low-resolution structural anchors and structured evidence.",
        "user_question": question,
        "compression_policy": {
            "anchor_policy": "Preserve low-cost visual structure through a whole-image anchor and selected local crop anchors.",
            "evidence_policy": "Preserve task-relevant, low-redundancy, anchor-grounded evidence.",
        },
        "question_analysis": _question_analysis(question),
        "evidence_graph_policy": {
            "G_global": "Global scene or event context from low-resolution anchors.",
            "G_local": "Crop/frame/audio grounded fine details.",
            "E_cross": "Cross-level support or contradiction between local details and global context.",
            "E_anchor": "Anchor-grounding links from evidence to anchor_link, bbox, or timestamp.",
        },
        "selection_policy": {
            "context_track": "Keep enough global layout, temporal coverage, and scene state to avoid local-only errors.",
            "salient_track": "Keep question-critical details, OCR, counts, rare objects, state changes, and event boundaries.",
            "redundancy_policy": "Merge repeated observations and keep the most grounded representative.",
            "rare_evidence_guard": "Do not discard small, rare, or low-frequency evidence when it directly affects the answer.",
        },
        "global_summary": _trim_text(stage1_response),
        "computed_evidence": computed_evidence,
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


def build_evidence_extraction_prompt(prompt: Dict[str, Any]) -> str:
    """Ask the vision model to compute task evidence before final answering."""
    structured = prompt.get("structured_evidence_prompt", prompt)
    anchors = prompt.get("low_resolution_anchor", {})
    question = structured.get("user_question") or ""
    analysis = structured.get("question_analysis") or _question_analysis(question)

    lines = [
        "MTEC++ Evidence Builder.",
        "Inspect every attached visual, video, and audio anchor. Compute evidence first; do not answer the user question yet.",
        f"Question: {_one_line(question)}",
        "QuestionDecomposition:",
        f"- target_evidence_types={','.join(analysis.get('target_evidence_types', []))}",
        f"- packing_policy={analysis.get('packing_policy')}",
        "AnchorUse:",
        "- global anchor: scene layout, object distribution, large-scale spatial relations, temporal context, and overall scene state.",
        "- crop anchors: small objects, readable text, colors, attributes, counts, local relations, and fine-grained state changes.",
        "- video anchor: start/middle/end coverage, action continuity, event order, scene transitions, and answer-relevant timestamps.",
        "- audio anchor: speech/ASR cues, speaker mentions, narration, music/sound effects, rhythm, silence, emphasis, and events that may not be visible.",
        "- cross-modal use: align audio events or narration with nearby video frames; use audio to recover details lost by low-FPS video, and use video to ground ambiguous audio.",
        "HybridEvidenceGraph:",
        "- G_global: whole-scene context, layout, event state, and long-range spatial/temporal relations.",
        "- G_local: crop-grounded fine details such as OCR, small objects, attributes, counts, and local states.",
        "- G_audio: speech, sound-event, rhythm, silence, and speaker/narrator evidence.",
        "- E_cross: relations where local details support or contradict global context.",
        "- E_avsync: audio-video synchrony, narration-to-frame links, sound-to-action links, and cross-modal contradictions.",
        "- E_anchor: every evidence item must be grounded to anchor_link and optional bbox/time.",
        "SelectionPolicy:",
        "- ContextTrack: preserve global layout/event context needed to avoid local-only mistakes.",
        "- SalientTrack: preserve rare or question-critical details even if small or low frequency.",
        "- Redundancy: merge repeated observations and keep only the most grounded representative.",
        "- Uncertainty: explicitly record ambiguous, unreadable, occluded, or conflicting evidence.",
    ]

    image_anchors = anchors.get("image_anchor", [])
    if image_anchors:
        lines.append("Anchors:")
        for anchor in image_anchors:
            lines.append(
                "- "
                f"{anchor.get('anchor_link')} type={anchor.get('type')} "
                f"region={_one_line(anchor.get('region_hint') or anchor.get('bbox_norm') or 'whole')} "
                f"bbox={_one_line(anchor.get('bbox_norm') or 'full')} "
                f"res={_resolution_text(anchor.get('resolution'))}"
            )

    video_anchors = anchors.get("video_anchor", [])
    audio_anchors = anchors.get("audio_anchor", [])
    transcript_anchors = anchors.get("transcript_anchor", [])
    if video_anchors or audio_anchors or transcript_anchors:
        lines.append("TemporalAnchors:")
        for anchor in video_anchors:
            lines.append(
                "- "
                f"{anchor.get('anchor_link')} type={anchor.get('type')} "
                f"duration={anchor.get('source_duration_sec')}s "
                f"frames={len(anchor.get('frames', []))}/{anchor.get('source_frame_count')} "
                f"target_fps={anchor.get('target_fps')} "
                f"res={_resolution_text(anchor.get('source_resolution'))}"
            )
            for frame in (anchor.get("frames") or [])[:12]:
                lines.append(
                    "  - "
                    f"{frame.get('anchor_link')} t={frame.get('time_sec')}s "
                    f"change={_short_number(frame.get('change_score'))}"
                )
        for anchor in audio_anchors:
            lines.append(
                "- "
                f"{anchor.get('anchor_link')} type={anchor.get('type')} "
                f"duration={anchor.get('source_duration_sec')}s "
                f"events={len(anchor.get('audio_event_segments') or [])} "
                f"bitrate={anchor.get('target_bitrate')}"
            )
            for segment in (anchor.get("audio_event_segments") or [])[:10]:
                lines.append(
                    "  - "
                    f"{segment.get('anchor_link')} t={segment.get('time_range_sec')} "
                    f"rms={_short_number(segment.get('rms'))}"
                )
        for anchor in transcript_anchors:
            lines.append(
                "- "
                f"{anchor.get('anchor_link')} type={anchor.get('type')} "
                f"source={anchor.get('source')} "
                f"segments={len(anchor.get('segments') or [])} "
                f"language={anchor.get('language')}"
            )
            for segment in (anchor.get("segments") or [])[:18]:
                lines.append(
                    "  - "
                    f"{segment.get('anchor_link')} t={segment.get('time_range_sec')} "
                    f"text={_one_line(segment.get('text'))}"
                )

    lines.append("RequiredComputations:")
    for item in analysis.get("required_computations", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "Return compact JSON only, with this schema:",
            '{"global_context":[],"local_details":[],"task_relevant_observations":[],"counts":[],"spatial_relations":[],"direction_orientation":[],"distance_depth":[],"temporal_segments":[],"audio_evidence":[],"speech_or_narration":[],"audio_visual_sync":[],"symbols_text_rules":[],"ocr_text":[],"cross_level_support":[],"anchor_grounding":[],"merged_duplicates":[],"rare_evidence":[],"option_elimination":[],"uncertainties":[],"preliminary_answer":""}',
            "Rules:",
            "- Every non-empty evidence item must cite one anchor_link.",
            "- For counting, list counted visible instances and ignored non-target instances.",
            "- For distance/depth, compare relative size, occlusion, perspective cues, scene geometry, and ground/contact cues; mark estimate uncertainty.",
            "- For symbols, signs, UI, labels, rules, or status indicators, identify the relevant object/context before reading color/text/state.",
            "- For direction/orientation, state the visual cue: front/rear, arrow head, body pose, or facing side.",
            "- For high-resolution images, compare the global anchor with crops before trusting any small detail.",
            "- For long videos, preserve start/middle/end context, event boundaries, state changes, and any evidence near answer-relevant timestamps.",
            "- For video questions, build a timestamped timeline and include evidence from both video frames and audio/narration.",
            "- For speech, subtitles, narration, or educational/explainer videos, extract mentioned entities, event order, and negated/not-mentioned items from audio when available.",
            "- Treat timestamped transcript segments as primary evidence for narration, dialogue, mentioned/not-mentioned events, and ordering of spoken facts.",
            "- Align transcript timestamps to nearby video frames and audio event segments before selecting an option.",
            "- For multiple-choice video questions, create option_elimination entries for every option using visual evidence, audio evidence, and cross-modal alignment.",
            "- If audio supports a detail not visible in the low-FPS video, keep it as audio_evidence with an anchor_link and explain the nearby visual context.",
            "- If audio and video disagree or one modality is missing/unclear, put the conflict in uncertainties instead of guessing.",
            "- If evidence conflicts across anchors, put the conflict in uncertainties.",
        ]
    )
    return "\n".join(lines)


def format_structured_evidence_prompt(prompt: Dict[str, Any], compact: bool = False) -> str:
    if compact:
        return format_compact_evidence_prompt(prompt)
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def format_rich_evidence_prompt(prompt: Dict[str, Any]) -> str:
    """Format a more task-directed evidence prompt for stronger API models.

    This keeps the same dual-channel input contract as the compact prompt:
    the model still receives the low-cost media anchor plus structured text.
    The richer form adds an explicit reasoning checklist so the model spends
    attention on details that are often lost after compression.
    """
    structured = prompt.get("structured_evidence_prompt", prompt)
    anchors = prompt.get("low_resolution_anchor", {})
    question = structured.get("user_question") or ""
    profile = prompt.get("compression_target", {}).get("question_profile") or infer_question_profile(question)
    budget = prompt.get("compression_target", {}).get("budget", {})

    lines = [
        "MTEC++ Evidence+.",
        "Use the attached low-cost media anchor as evidence; answer from visible/temporal details, not priors.",
        f"Question: {_one_line(question)}",
        f"Profile: {profile.get('question_type')} | required={','.join(profile.get('required_modalities', []))}",
    ]

    if budget:
        lines.append(
            "Budget: "
            f"anchor_ratio={budget.get('anchor_ratio')} evidence_ratio={budget.get('evidence_ratio')}"
        )

    analysis = structured.get("question_analysis") or _question_analysis(question)
    computations = analysis.get("required_computations", [])
    if computations:
        lines.append("RequiredComputedEvidence:")
        lines.extend(f"- {item}" for item in computations)

    graph_policy = structured.get("evidence_graph_policy", {})
    selection_policy = structured.get("selection_policy", {})
    if graph_policy or selection_policy:
        lines.append("PackingPolicy:")
        for key, value in graph_policy.items():
            lines.append(f"- {key}: {_one_line(value)}")
        for key, value in selection_policy.items():
            lines.append(f"- {key}: {_one_line(value)}")

    image_anchors = anchors.get("image_anchor", [])
    if image_anchors:
        lines.append("AnchorSummary:")
        for anchor in image_anchors:
            lines.append(
                "- "
                f"source_res={_resolution_text(anchor.get('source_resolution'))} "
                f"anchor_res={_resolution_text(anchor.get('resolution'))} "
                f"type={anchor.get('type')} "
                f"link={anchor.get('anchor_link')} "
                f"region={_one_line(anchor.get('region_hint') or anchor.get('bbox_norm') or 'whole')}"
            )

    video_anchors = anchors.get("video_anchor", [])
    if video_anchors:
        lines.append("AnchorSummary:")
        for anchor in video_anchors:
            lines.append(
                "- "
                f"duration={anchor.get('source_duration_sec')}s "
                f"frames={len(anchor.get('frames', []))}/{anchor.get('source_frame_count')} "
                f"target_fps={anchor.get('target_fps')} source_res={_resolution_text(anchor.get('source_resolution'))}"
            )
            rows = _compact_temporal_rows(structured, anchors)
            if rows:
                lines.append("  Timeline t|anchor|change|note:")
                lines.extend(f"  {row}" for row in _sample_rows(rows, max_rows=10))
            boundaries = anchor.get("event_boundaries") or []
            if boundaries:
                lines.append("  ChangePoints:")
                for boundary in boundaries[:4]:
                    lines.append(
                        "  - "
                        f"time={boundary.get('time_sec')}s anchor={boundary.get('anchor_link')} "
                        f"change={_short_number(boundary.get('change_score'))}"
                    )

    audio_anchors = anchors.get("audio_anchor", [])
    if audio_anchors:
        lines.append("AudioAnchorSummary:")
        for anchor in audio_anchors:
            summary = anchor.get("energy_summary") or {}
            lines.append(
                "- "
                f"duration={anchor.get('source_duration_sec')}s "
                f"events={len(anchor.get('audio_event_segments') or [])} "
                f"sample_rate={anchor.get('target_sample_rate')} "
                f"bitrate={anchor.get('target_bitrate')} "
                f"rms_mean={_short_number(summary.get('rms_mean'))} "
                f"rms_max={_short_number(summary.get('rms_max'))}"
            )
            segments = anchor.get("audio_event_segments") or []
            if segments:
                lines.append("  AudioEvents time|anchor|rms:")
                for segment in segments[:8]:
                    lines.append(
                        "  "
                        f"{segment.get('time_range_sec')}|{segment.get('anchor_link')}|"
                        f"{_short_number(segment.get('rms'))}"
                    )

    transcript_anchors = anchors.get("transcript_anchor", [])
    if transcript_anchors:
        lines.append("TranscriptEvidence:")
        for anchor in transcript_anchors:
            warnings = anchor.get("warnings") or []
            lines.append(
                "- "
                f"source={anchor.get('source')} "
                f"segments={len(anchor.get('segments') or [])} "
                f"language={anchor.get('language')} "
                f"warnings={_one_line('; '.join(warnings[:2])) if warnings else 'none'}"
            )
            for segment in (anchor.get("segments") or [])[:16]:
                lines.append(
                    "  "
                    f"{segment.get('time_range_sec')}|{segment.get('anchor_link')}|"
                    f"{_one_line(segment.get('text'))}"
                )

    visual_rows = _compact_visual_rows(structured)
    if visual_rows:
        lines.append("VisualGrounding time|region|anchor|conf|content|reason:")
        lines.extend(_sample_rows(visual_rows, max_rows=8))

    computed_rows = _compact_computed_evidence_rows(structured.get("computed_evidence"))
    if computed_rows:
        lines.append("ComputedEvidence:")
        lines.extend(computed_rows)

    checklist = _task_checklist(question, profile)
    if checklist:
        lines.append("TaskChecklist:")
        lines.extend(f"- {item}" for item in checklist[:8])

    lines.append(
        "AnswerPolicy: inspect every attached anchor image/video/audio first; use crop links for local details, the global link for layout, video for temporal order, and audio for speech/sound evidence. MCQ -> compare every option against visual, audio, and cross-modal evidence before returning only the best letter. Short answer -> return the shortest exact answer."
    )
    return "\n".join(lines)


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

    visual_context_rows = _compact_visual_context_rows(structured.get("visual_context_hint"))
    if visual_context_rows:
        lines.append("NoTranscriptVisualContext:")
        lines.extend(visual_context_rows)

    temporal_rows = _compact_temporal_rows(structured, anchors)
    if temporal_rows:
        lines.append("TemporalChain t|anchor|delta|note:")
        lines.extend(temporal_rows)

    visual_rows = _compact_visual_rows(structured)
    if visual_rows:
        lines.append("VisualEvidence time|region|anchor|conf|content|reason:")
        lines.extend(visual_rows)

    computed_rows = _compact_computed_evidence_rows(structured.get("computed_evidence"))
    if computed_rows:
        lines.append("ComputedEvidence:")
        lines.extend(computed_rows)

    audio_rows = _compact_audio_rows(structured, anchors)
    if audio_rows:
        lines.append("AudioEvidence time|type|anchor|score|rms|content:")
        lines.extend(audio_rows)

    transcript_rows = _compact_transcript_rows(anchors)
    if transcript_rows:
        lines.append("TranscriptEvidence time|anchor|source|text:")
        lines.extend(transcript_rows)

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


def _coerce_computed_evidence(stage1_response: Optional[str]) -> Any:
    if not stage1_response:
        return {}
    text = _strip_json_code_fence(stage1_response.strip())
    if not text:
        return {}
    for candidate in (text, _json_object_slice(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return {"raw_evidence": _trim_text(text, max_chars=2600)}


def _strip_json_code_fence(text: str) -> str:
    value = text.strip()
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


def _question_analysis(question: str) -> Dict[str, Any]:
    text = (question or "").lower()
    required = []
    targets = []
    if any(phrase in text for phrase in ("how many", "number of", "count", "many")):
        required.append("counting: enumerate visible target instances, cite anchors, and separate target vs non-target objects.")
        targets.append("counted_objects")
    if any(word in text for word in ("meter", "meters", "distance", "far", "within", "closer", "nearest")):
        required.append("distance_depth: estimate ground distance or relative closeness using perspective, size, occlusion, lane/curb cues, and contact with ground.")
        targets.append("distance_or_depth")
    if any(word in text for word in ("left", "right", "above", "below", "near", "beside", "between", "relative", "lane", "curb")):
        required.append("spatial_relation: determine left/right/above/below/near/between using the global anchor first, then verify local details in crops.")
        targets.append("spatial_relation")
    if any(word in text for word in ("facing", "direction", "arrow", "toward", "away", "north", "south", "east", "west")):
        required.append("direction_orientation: identify front/rear/head/arrow/body cues before choosing direction or orientation.")
        targets.append("direction_or_orientation")
    if any(word in text for word in ("traffic", "light", "stop", "sign", "speed limit", "park", "parking", "legal", "lane", "button", "screen", "dial", "gauge", "indicator", "status")):
        required.append("symbol_text_rule_state: identify the relevant sign/symbol/control/status object first; read color/text/number/state and decide applicability within the scene context.")
        targets.append("symbol_text_rule_state")
    if any(word in text for word in ("text", "word", "read", "number", "digit", "label", "sign", "color", "green", "red", "yellow", "blue")):
        required.append("ocr_attribute: inspect crops for text, digits, signs, colors, labels, and small attributes.")
        targets.append("ocr_or_attribute")
    if any(word in text for word in ("yes", "no", "whether", "is there", "are there", "do we", "does")):
        required.append("binary_verification: verify the queried condition directly and list the visible evidence for yes or no.")
        targets.append("binary_condition")
    if not required:
        required.append("semantic_grounding: identify the target object/event and cite the anchor evidence that supports the answer.")
        targets.append("semantic_target")
    return {
        "target_evidence_types": targets,
        "required_computations": required,
        "packing_policy": "Add computed evidence only when grounded to anchor links; preserve uncertainty rather than guessing.",
    }


def _compact_computed_evidence_rows(evidence: Any) -> List[str]:
    if not evidence:
        return []
    if isinstance(evidence, str):
        return [f"- raw={_one_line(evidence)}"]
    if not isinstance(evidence, dict):
        return [f"- raw={_one_line(evidence)}"]

    rows = []
    preferred_keys = [
        "global_context",
        "local_details",
        "task_relevant_observations",
        "counts",
        "spatial_relations",
        "direction_orientation",
        "distance_depth",
        "temporal_segments",
        "audio_evidence",
        "speech_or_narration",
        "audio_visual_sync",
        "symbols_text_rules",
        "ocr_text",
        "cross_level_support",
        "anchor_grounding",
        "merged_duplicates",
        "rare_evidence",
        "option_elimination",
        "uncertainties",
        "preliminary_answer",
        "raw_evidence",
    ]
    for key in preferred_keys:
        value = evidence.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            for item in value[:8]:
                rows.append(f"- {key}: {_one_line(_compact_json(item))}")
        else:
            rows.append(f"- {key}: {_one_line(_compact_json(value))}")
    return rows[:40]


def _compact_json(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _task_checklist(question: str, profile: Dict[str, Any]) -> List[str]:
    text = (question or "").lower()
    needs = profile.get("needs", {})
    checks = [
        "Identify the object or event asked about before deciding the answer.",
        "Use visible evidence in the anchor; if detail is unclear, prefer the option most directly supported by the anchor.",
    ]
    if needs.get("spatial") or any(word in text for word in ("left", "right", "front", "behind", "direction", "facing", "side")):
        checks.append("Check spatial relations carefully: left/right, front/back, above/below, facing direction, and relative position.")
    if needs.get("ocr_or_detail") or any(word in text for word in ("text", "sign", "number", "color", "word", "logo", "digit", "read")):
        checks.append("Inspect small visual details: text, signs, numbers, colors, labels, logos, and object counts.")
    if any(word in text for word in ("how many", "number of", "count", "many")):
        checks.append("Count visible instances one by one; avoid guessing from scene type.")
    if any(word in text for word in ("uphill", "downhill", "flat", "slope", "incline", "decline")):
        checks.append("For slope or incline questions, define the observer/travel direction first; if the surface drops away along that direction, answer downhill/decline, and if it rises away, answer uphill/incline.")
    if any(word in text for word in ("yes", "no", "whether", "is there", "are there", "does", "do ")):
        checks.append("For yes/no questions, verify the queried condition directly before answering yes or no.")
    if needs.get("temporal") or any(word in text for word in ("before", "after", "first", "then", "finally", "during", "while")):
        checks.append("For video, compare the earliest, middle, and latest anchor frames; reason over event order and changes.")
    if "options" in text or "\n" in question:
        checks.append("For multiple-choice, eliminate options contradicted by the anchor before choosing the final letter.")
    return checks


def _sample_rows(rows: List[str], max_rows: int) -> List[str]:
    if len(rows) <= max_rows:
        return rows
    if max_rows <= 3:
        return rows[:max_rows]
    head_count = max_rows // 2
    tail_count = max_rows - head_count
    return rows[:head_count] + rows[-tail_count:]


def _resolution_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    width = value.get("width")
    height = value.get("height")
    if width is None or height is None:
        return ""
    return f"{width}x{height}"


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


def _compact_visual_context_rows(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, dict):
        rows: List[str] = []
        summary = value.get("visual_summary")
        if summary:
            rows.append(f"summary|{_one_line(summary)}")
        keywords = value.get("question_keywords") or []
        if keywords:
            rows.append("keywords|" + _one_line(", ".join(str(item) for item in keywords[:16])))
        actions = value.get("actions") or []
        if actions:
            rows.append("actions|" + _one_line(", ".join(str(item) for item in actions[:16])))
        objects = value.get("visible_objects") or []
        if objects:
            rows.append("objects|" + _one_line(", ".join(str(item) for item in objects[:16])))
        for event in (value.get("event_chain") or [])[:8]:
            if isinstance(event, dict):
                rows.append(
                    "event|"
                    + "|".join(
                        [
                            _one_line(event.get("time")),
                            _one_line(event.get("action")),
                            _one_line(", ".join(str(item) for item in (event.get("objects") or [])[:8])),
                            _one_line(event.get("anchor")),
                        ]
                    )
                )
            else:
                rows.append("event|" + _one_line(event))
        option_cues = value.get("option_visual_cues") or {}
        if isinstance(option_cues, dict):
            for option, cue in list(option_cues.items())[:6]:
                if isinstance(cue, dict):
                    support = _one_line(", ".join(str(item) for item in (cue.get("support") or [])[:4]))
                    contradiction = _one_line(", ".join(str(item) for item in (cue.get("contradiction") or [])[:4]))
                    needs = _one_line(", ".join(str(item) for item in (cue.get("needs_check") or [])[:4]))
                    rows.append(f"option_{option}|support={support}|contradiction={contradiction}|needs={needs}")
                else:
                    rows.append(f"option_{option}|{_one_line(cue)}")
        absence = value.get("absence_checks") or []
        if absence:
            rows.append("absence|" + _one_line(", ".join(str(item) for item in absence[:8])))
        uncertainty = value.get("uncertainty") or []
        if uncertainty:
            rows.append("uncertainty|" + _one_line(", ".join(str(item) for item in uncertainty[:8])))
        return rows[:24]
    return ["raw|" + _one_line(value)]


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


def _compact_transcript_rows(anchors: Dict[str, Any]) -> List[str]:
    rows: List[str] = []
    for anchor in anchors.get("transcript_anchor", []):
        source = anchor.get("source") or anchor.get("backend") or ""
        for segment in _prioritized_transcript_segments(anchor, limit=28):
            role = segment.get("selection_role") or "transcript"
            score = segment.get("relevance_score")
            source_text = f"{source}/{role}"
            if score:
                source_text += f"/score={_short_number(score)}"
            rows.append(
                "|".join(
                    [
                        str(segment.get("time_range_sec", "")),
                        str(segment.get("anchor_link") or anchor.get("anchor_link") or ""),
                        source_text,
                        _one_line(segment.get("text")),
                    ]
                )
            )
    return rows


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
