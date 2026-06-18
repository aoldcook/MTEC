import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoomrefine.mtec_media_pipeline import create_multimodal_structural_anchors
from zoomrefine.mtec_prompt_plus import (
    build_structured_evidence_prompt,
    format_structured_evidence_prompt,
)


DEFAULT_QUESTION = (
    "Describe the important visual, temporal, and acoustic evidence needed "
    "to answer a question about this multimodal input."
)


def _first_anchor(package: dict, key: str) -> Optional[dict]:
    anchors = package["low_resolution_anchor"].get(key, [])
    return anchors[0] if anchors else None


def run_media_anchor_smoke(args: argparse.Namespace) -> dict:
    package = create_multimodal_structural_anchors(
        question=args.question,
        output_dir=args.output_dir,
        image_path=args.image,
        video_path=args.video,
        audio_path=args.audio,
        total_budget=args.total_budget,
    )

    prompt = build_structured_evidence_prompt(
        question=args.question,
        stage1_response=None,
        bbox_norm=None,
        expanded_bbox_norm=None,
        global_anchor=package["low_resolution_anchor"].get("image_anchor"),
        video_anchor=package["low_resolution_anchor"].get("video_anchor"),
        audio_anchor=package["low_resolution_anchor"].get("audio_anchor"),
        total_budget=args.total_budget,
    )

    result = {
        **package,
        "structured_evidence_prompt": prompt["structured_evidence_prompt"],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "mtec_media_anchor_result.json"
    prompt_path = output_dir / "mtec_structured_evidence_prompt.txt"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(format_structured_evidence_prompt(prompt, compact=True), encoding="utf-8")
    result["result_path"] = str(result_path)
    result["prompt_path"] = str(prompt_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MTEC-Prompt++ media structural anchors.")
    parser.add_argument("--image", help="Optional image path for an image global anchor.")
    parser.add_argument("--video", help="Optional video path for low-FPS frame anchors.")
    parser.add_argument("--audio", help="Optional audio path for low-bitrate acoustic anchors.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--output-dir", default="outputs/mtec_media_anchor")
    parser.add_argument("--total-budget", type=int, default=1000)
    args = parser.parse_args()

    if not args.image and not args.video and not args.audio:
        raise SystemExit("Provide at least one of --image, --video, or --audio.")

    result = run_media_anchor_smoke(args)
    anchors = result["low_resolution_anchor"]
    print("MTEC-Prompt++ media anchor smoke completed.")
    print(f"Image anchors: {len(anchors.get('image_anchor', []))}")
    print(f"Video anchors: {len(anchors.get('video_anchor', []))}")
    print(f"Audio anchors: {len(anchors.get('audio_anchor', []))}")
    print(f"Result: {result['result_path']}")
    print(f"Structured prompt: {result['prompt_path']}")


if __name__ == "__main__":
    main()
