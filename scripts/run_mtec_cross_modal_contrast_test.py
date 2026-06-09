import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoomrefine.mtec_media_pipeline import (  # noqa: E402
    create_audio_structural_anchor,
    create_video_structural_anchor,
)
from zoomrefine.mtec_prompt_plus import (  # noqa: E402
    build_structured_evidence_prompt,
    format_structured_evidence_prompt,
)


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-flash"
DEFAULT_VIDEO_PATH = "outputs/downloaded_video_test/downloaded_test_video.mp4"
QUESTION = (
    "Use the supplied evidence to identify the strongest cross-modal alignment. "
    "Which audio event time range lines up with a nearby video frame or scene change? "
    "Cite the audio anchor, video anchor, timestamps, and say whether the evidence is sufficient."
)


def encode_file_data_url(path: Path, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_prompt_text(
    question: str,
    video_anchor: Optional[Dict[str, Any]],
    audio_anchor: Optional[Dict[str, Any]],
) -> str:
    prompt = build_structured_evidence_prompt(
        question=question,
        stage1_response=None,
        bbox_norm=None,
        expanded_bbox_norm=None,
        global_anchor=None,
        video_anchor=video_anchor,
        audio_anchor=audio_anchor,
    )
    return format_structured_evidence_prompt(prompt, compact=True)


def run_case(
    client: Any,
    model: str,
    case_id: str,
    context_text: str,
    video_path: Optional[Path],
    max_tokens: int,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Case: {case_id}\n"
                f"{context_text}\n\n"
                f"Question: {QUESTION}\n"
                "Return a concise answer with cited anchor ids."
            ),
        }
    ]
    if video_path:
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": encode_file_data_url(video_path, "video/mp4")},
            }
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return {
            "case_id": case_id,
            "answer": answer,
            "usage": usage,
            "evidence_score": score_evidence_answer(answer),
        }
    except Exception as err:
        return {
            "case_id": case_id,
            "answer": "",
            "usage": None,
            "error": str(err),
            "evidence_score": score_evidence_answer(""),
        }


def score_evidence_answer(answer: str) -> Dict[str, Any]:
    text = answer.lower()
    has_audio_anchor = "audio_anchor_low_bitrate_event" in text
    has_video_anchor = "video_anchor_low_fps_frame" in text
    has_timestamp = bool(re.search(r"\b\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?s\b", text))
    has_sufficiency = any(word in text for word in ("sufficient", "insufficient", "enough", "not enough"))
    has_alignment_word = any(word in text for word in ("align", "nearby", "lines up", "correspond", "delta"))
    return {
        "has_audio_anchor": has_audio_anchor,
        "has_video_anchor": has_video_anchor,
        "has_timestamp": has_timestamp,
        "has_sufficiency_judgement": has_sufficiency,
        "has_alignment_language": has_alignment_word,
        "cross_modal_citation_score": sum(
            int(value)
            for value in (
                has_audio_anchor,
                has_video_anchor,
                has_timestamp,
                has_sufficiency,
                has_alignment_word,
            )
        ),
    }


def token_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for result in results:
        usage = result.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        summary[result["case_id"]] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "text_tokens": prompt_details.get("text_tokens"),
            "video_tokens": prompt_details.get("video_tokens"),
            "audio_tokens": prompt_details.get("audio_tokens"),
        }
    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_anchor = create_video_structural_anchor(
        video_path=str(video_path),
        output_dir=str(output_dir / "anchors"),
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        max_side=args.max_side,
    )
    audio_anchor = create_audio_structural_anchor(
        audio_path=str(video_path),
        output_dir=str(output_dir / "anchors"),
    )
    compressed_video_path = Path(video_anchor["low_fps_video_path"])

    video_only_text = build_prompt_text(QUESTION, video_anchor, None)
    audio_only_text = build_prompt_text(QUESTION, None, audio_anchor)
    cross_modal_text = build_prompt_text(QUESTION, video_anchor, audio_anchor)

    prompts = {
        "video_only": video_only_text,
        "audio_only": audio_only_text,
        "cross_modal": cross_modal_text,
    }
    prompt_stats = {
        key: {"chars": len(value), "estimated_text_tokens": estimate_text_tokens(value)}
        for key, value in prompts.items()
    }

    api_key = os.environ.get(args.api_key_env)
    model_results = []
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=args.base_url)
        model_results = [
            run_case(client, args.model, "video_only", video_only_text, compressed_video_path, args.max_tokens),
            run_case(client, args.model, "audio_only", audio_only_text, None, args.max_tokens),
            run_case(client, args.model, "cross_modal", cross_modal_text, compressed_video_path, args.max_tokens),
        ]

    result = {
        "source_video_path": str(video_path),
        "compressed_video_path": str(compressed_video_path),
        "question": QUESTION,
        "video_anchor_summary": {
            "source_frame_count": video_anchor.get("source_frame_count"),
            "selected_frame_count": len(video_anchor.get("frames", [])),
            "event_boundaries": video_anchor.get("event_boundaries", []),
            "frame_retention_ratio": video_anchor.get("compression", {}).get("frame_retention_ratio"),
        },
        "audio_anchor_summary": {
            "event_count": len(audio_anchor.get("audio_event_segments", [])),
            "events": audio_anchor.get("audio_event_segments", []),
            "energy_summary": audio_anchor.get("energy_summary", {}),
        },
        "cross_modal_relations": build_structured_evidence_prompt(
            question=QUESTION,
            stage1_response=None,
            bbox_norm=None,
            expanded_bbox_norm=None,
            global_anchor=None,
            video_anchor=video_anchor,
            audio_anchor=audio_anchor,
        )["structured_evidence_prompt"]["cross_modal_relations"],
        "prompt_stats": prompt_stats,
        "model": args.model if api_key else None,
        "model_results": model_results,
        "model_token_summary": token_summary(model_results),
        "model_run_status": "completed" if api_key else "not_run_missing_api_key",
    }
    result_path = output_dir / "mtec_cross_modal_contrast_test.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTEC cross-modal contrast test on one video.")
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", default="outputs/cross_modal_contrast_test")
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    result = run(args)
    print("MTEC cross-modal contrast test completed.")
    print(f"Source video: {result['source_video_path']}")
    print(f"Compressed video: {result['compressed_video_path']}")
    print(f"Video frames: {result['video_anchor_summary']['selected_frame_count']} selected")
    print(f"Audio events: {result['audio_anchor_summary']['event_count']} selected")
    print(f"Cross-modal relations: {len(result['cross_modal_relations'])}")
    print(f"Model run status: {result['model_run_status']}")
    for case_id, stats in result["prompt_stats"].items():
        print(f"{case_id}: {stats['chars']} chars, est_text_tokens={stats['estimated_text_tokens']}")
    for item in result["model_results"]:
        score = item["evidence_score"]["cross_modal_citation_score"]
        usage = item.get("usage") or {}
        print(f"{item['case_id']}: score={score}, prompt_tokens={usage.get('prompt_tokens')}")
    print(f"Result: {result['result_path']}")


if __name__ == "__main__":
    main()
