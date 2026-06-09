import argparse
import base64
import json
import os
import re
import sys
import urllib.request
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


DEFAULT_VIDEO_URL = (
    "https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/"
    "big_buck_bunny.mp4"
)
DEFAULT_MODEL = "qwen-vl-max-latest"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PATCH_SIZE = 28

QA_SET = [
    {
        "id": "main_subject",
        "question": "What kind of animal is the main character in the video?",
        "expected_keywords": ["rabbit", "bunny", "hare"],
    },
    {
        "id": "animation_type",
        "question": "Is the video animated or live-action?",
        "expected_keywords": ["animated", "animation", "cartoon", "3d"],
    },
    {
        "id": "outdoor_scene",
        "question": "Does the video show an outdoor grassy or natural scene?",
        "expected_keywords": ["yes", "outdoor", "grass", "grassy", "natural", "field"],
    },
]


def download_video(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, output_path)


def encode_file_data_url(path: Path, mime_type: str) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def estimate_frame_tokens(width: int, height: int, patch_size: int = PATCH_SIZE) -> int:
    if width <= 0 or height <= 0:
        return 0
    return max(1, ((width + patch_size - 1) // patch_size) * ((height + patch_size - 1) // patch_size))


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_token_usage(
    video_anchor: Dict[str, Any],
    audio_anchor: Dict[str, Any],
    structured_prompt_text: str,
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    source_resolution = video_anchor.get("source_resolution", {})
    source_width = int(source_resolution.get("width") or 0)
    source_height = int(source_resolution.get("height") or 0)
    source_frame_count = int(video_anchor.get("source_frame_count") or 0)
    source_visual_tokens = source_frame_count * estimate_frame_tokens(source_width, source_height)

    compressed_visual_tokens = 0
    for frame in video_anchor.get("frames", []):
        resolution = frame.get("resolution", {})
        compressed_visual_tokens += estimate_frame_tokens(
            int(resolution.get("width") or 0),
            int(resolution.get("height") or 0),
        )

    question_text = "\n".join(item["question"] for item in questions)
    original_text_tokens = estimate_text_tokens(question_text)
    compressed_text_tokens = estimate_text_tokens(question_text + "\n" + structured_prompt_text)
    audio_event_tokens = estimate_text_tokens(json.dumps(audio_anchor.get("audio_event_segments", []), ensure_ascii=False))

    original_total = source_visual_tokens + original_text_tokens
    compressed_total = compressed_visual_tokens + compressed_text_tokens + audio_event_tokens
    return {
        "method": "proxy_estimate_patch28",
        "original": {
            "visual_tokens": source_visual_tokens,
            "text_tokens": original_text_tokens,
            "total_tokens": original_total,
            "frame_count": source_frame_count,
            "resolution": {"width": source_width, "height": source_height},
        },
        "compressed": {
            "visual_tokens": compressed_visual_tokens,
            "text_tokens": compressed_text_tokens,
            "audio_event_tokens": audio_event_tokens,
            "total_tokens": compressed_total,
            "frame_count": len(video_anchor.get("frames", [])),
        },
        "compression_rate": _safe_ratio(compressed_total, original_total),
        "token_saving_rate": _safe_saving_rate(compressed_total, original_total),
    }


def run_model_comparison(
    original_video_path: Path,
    compressed_video_path: Path,
    structured_prompt_text: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError as err:
        return {"error": f"openai package is not installed: {err}"}

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    original_video_url = encode_file_data_url(original_video_path, "video/mp4")
    compressed_video_url = encode_file_data_url(compressed_video_path, "video/mp4")

    original_results = []
    compressed_results = []
    for item in QA_SET:
        original_results.append(
            _ask_video_model(
                client=client,
                model=args.model,
                video_url=original_video_url,
                question=item["question"],
                context_text="Answer the question from the original video.",
                max_tokens=args.max_tokens,
            )
        )
        compressed_results.append(
            _ask_video_model(
                client=client,
                model=args.model,
                video_url=compressed_video_url,
                question=item["question"],
                context_text=(
                    "Answer the question from the compressed MTEC-Prompt++ video anchor. "
                    "Use this structured evidence prompt as additional context:\n"
                    f"{structured_prompt_text}"
                ),
                max_tokens=args.max_tokens,
            )
        )

    original_metrics = _score_answers(original_results, QA_SET)
    compressed_metrics = _score_answers(compressed_results, QA_SET)
    return {
        "model": args.model,
        "base_url": args.base_url,
        "original_results": original_results,
        "compressed_results": compressed_results,
        "original_metrics": original_metrics,
        "compressed_metrics": compressed_metrics,
        "model_token_comparison": _usage_comparison(original_results, compressed_results),
    }


def _ask_video_model(
    client: Any,
    model: str,
    video_url: str,
    question: str,
    context_text: str,
    max_tokens: int,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{context_text}\n\nQuestion: {question}"},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return {
            "question": question,
            "answer": content,
            "usage": usage,
            "answerable": _is_answerable(content),
        }
    except Exception as err:
        return {
            "question": question,
            "answer": "",
            "usage": None,
            "answerable": False,
            "error": str(err),
        }


def _score_answers(results: List[Dict[str, Any]], qa_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = 0
    answerable = 0
    details = []
    for result, expected in zip(results, qa_set):
        answer = result.get("answer", "")
        is_correct = _keyword_match(answer, expected["expected_keywords"])
        is_answerable = bool(result.get("answerable"))
        correct += int(is_correct)
        answerable += int(is_answerable)
        details.append(
            {
                "id": expected["id"],
                "correct": is_correct,
                "answerable": is_answerable,
                "expected_keywords": expected["expected_keywords"],
            }
        )
    total = len(qa_set)
    return {
        "accuracy": _safe_ratio(correct, total),
        "answerability_rate": _safe_ratio(answerable, total),
        "correct": correct,
        "answerable": answerable,
        "total": total,
        "details": details,
    }


def _usage_comparison(
    original_results: List[Dict[str, Any]],
    compressed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    original_prompt_tokens = _sum_usage(original_results, "prompt_tokens")
    compressed_prompt_tokens = _sum_usage(compressed_results, "prompt_tokens")
    return {
        "original_prompt_tokens": original_prompt_tokens,
        "compressed_prompt_tokens": compressed_prompt_tokens,
        "compression_rate": _safe_ratio(compressed_prompt_tokens, original_prompt_tokens),
        "token_saving_rate": _safe_saving_rate(compressed_prompt_tokens, original_prompt_tokens),
    }


def _sum_usage(results: List[Dict[str, Any]], key: str) -> Optional[int]:
    total = 0
    seen = False
    for result in results:
        usage = result.get("usage") or {}
        if key in usage and usage[key] is not None:
            total += int(usage[key])
            seen = True
    return total if seen else None


def _keyword_match(answer: str, keywords: List[str]) -> bool:
    text = answer.lower()
    return any(re.search(rf"\b{re.escape(keyword.lower())}\b", text) for keyword in keywords)


def _is_answerable(answer: str) -> bool:
    if not answer.strip():
        return False
    unable_patterns = [
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not enough information",
        "cannot determine",
        "can't determine",
        "unsure",
    ]
    text = answer.lower()
    return not any(pattern in text for pattern in unable_patterns)


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def _safe_saving_rate(compressed: Optional[float], original: Optional[float]) -> Optional[float]:
    if compressed is None or original in (None, 0):
        return None
    return round(1.0 - float(compressed) / float(original), 6)


def _artifact_size_summary(original_video_path: Path, compressed_dir: Path) -> Dict[str, Any]:
    compressed_files = [path for path in compressed_dir.rglob("*") if path.is_file()]
    original_bytes = original_video_path.stat().st_size
    compressed_bytes = sum(path.stat().st_size for path in compressed_files)
    return {
        "original_bytes": original_bytes,
        "compressed_artifact_bytes": compressed_bytes,
        "compression_rate": _safe_ratio(compressed_bytes, original_bytes),
        "saving_rate": _safe_saving_rate(compressed_bytes, original_bytes),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in compressed_files
        ],
    }


def run_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_video_path = output_dir / "downloaded_test_video.mp4"
    download_video(args.video_url, original_video_path)

    video_anchor = create_video_structural_anchor(
        video_path=str(original_video_path),
        output_dir=str(output_dir / "compressed"),
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        max_side=args.max_side,
    )
    audio_anchor = create_audio_structural_anchor(
        audio_path=str(original_video_path),
        output_dir=str(output_dir / "compressed"),
    )
    prompt = build_structured_evidence_prompt(
        question="\n".join(item["question"] for item in QA_SET),
        stage1_response=None,
        bbox_norm=None,
        expanded_bbox_norm=None,
        global_anchor=None,
        video_anchor=video_anchor,
        audio_anchor=audio_anchor,
    )
    full_structured_prompt_text = format_structured_evidence_prompt(prompt)
    structured_prompt_text = format_structured_evidence_prompt(prompt, compact=True)
    compressed_video_path = Path(video_anchor["low_fps_video_path"])

    token_estimate = estimate_token_usage(video_anchor, audio_anchor, structured_prompt_text, QA_SET)
    model_result = (
        run_model_comparison(original_video_path, compressed_video_path, structured_prompt_text, args)
        if args.run_model
        else None
    )
    artifact_size = _artifact_size_summary(original_video_path, output_dir / "compressed")

    result = {
        "download": {
            "url": args.video_url,
            "path": str(original_video_path),
            "bytes": original_video_path.stat().st_size,
        },
        "compressed_video_path": str(compressed_video_path),
        "video_anchor": video_anchor,
        "audio_anchor": audio_anchor,
        "structured_evidence_prompt": prompt["structured_evidence_prompt"],
        "structured_prompt_text_stats": {
            "full_chars": len(full_structured_prompt_text),
            "compact_chars": len(structured_prompt_text),
            "char_compression_rate": _safe_ratio(len(structured_prompt_text), len(full_structured_prompt_text)),
            "char_saving_rate": _safe_saving_rate(len(structured_prompt_text), len(full_structured_prompt_text)),
        },
        "artifact_size": artifact_size,
        "token_estimate": token_estimate,
        "model_result": model_result,
        "accuracy": (
            model_result.get("compressed_metrics", {}).get("accuracy")
            if isinstance(model_result, dict)
            else None
        ),
        "answerability_rate": (
            model_result.get("compressed_metrics", {}).get("answerability_rate")
            if isinstance(model_result, dict)
            else None
        ),
        "model_run_status": (
            "completed"
            if model_result and "error" not in model_result
            else "not_run_missing_api_key"
            if args.run_model and not os.environ.get(args.api_key_env)
            else "not_requested"
            if not args.run_model
            else "failed"
        ),
    }
    result_path = output_dir / "mtec_video_model_comparison.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original video vs MTEC-Prompt++ compressed video.")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--output-dir", default="outputs/downloaded_video_test")
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    result = run_comparison(args)
    estimate = result["token_estimate"]
    print("MTEC video comparison completed.")
    print(f"Downloaded video: {result['download']['path']} ({result['download']['bytes']} bytes)")
    print(f"Compressed video: {result['compressed_video_path']}")
    print(f"Estimated original tokens: {estimate['original']['total_tokens']}")
    print(f"Estimated compressed tokens: {estimate['compressed']['total_tokens']}")
    print(f"Estimated compression rate: {estimate['compression_rate']}")
    print(f"Estimated token saving rate: {estimate['token_saving_rate']}")
    print(f"Artifact byte compression rate: {result['artifact_size']['compression_rate']}")
    print(f"Artifact byte saving rate: {result['artifact_size']['saving_rate']}")
    print(f"Model run status: {result['model_run_status']}")
    print(f"Result: {result['result_path']}")


if __name__ == "__main__":
    main()
