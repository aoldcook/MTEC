import argparse
import base64
import io
import json
import math
import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoomrefine.mtec_prompt_plus import (
    build_image_crop_anchor_metadata,
    build_structured_evidence_prompt,
    create_image_global_anchor,
    format_structured_evidence_prompt,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-flash"
DEFAULT_QUESTION = (
    "Describe the main visible subject in this image and identify the most "
    "important region for answering that question."
)


def encode_image(image_bytes: bytes, mime_type: str) -> str:
    payload = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def load_image_bytes(image_path: Path) -> tuple[bytes, int, int, str]:
    with Image.open(image_path) as img:
        width, height = img.size
        mime_type = Image.MIME.get(img.format, "image/png")
        buffer = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            mime_type = "image/jpeg"
            img.save(buffer, format="JPEG", quality=92, optimize=True)
        else:
            image_path_bytes = image_path.read_bytes()
            return image_path_bytes, width, height, mime_type
        return buffer.getvalue(), width, height, mime_type


def parse_bbox(text: str) -> list[float] | None:
    patterns = [
        r'"bbox"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
        r"(?:Bounding Box|Bounding box|bbox)\s*[:：]?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if not matches:
            continue
        coords = [float(value) for value in matches[-1]]
        x1, y1, x2, y2 = [max(0.0, min(1.0, value)) for value in coords]
        if x1 < x2 and y1 < y2:
            return [x1, y1, x2, y2]
    return None


def expand_bbox(bbox: list[float], padding: float) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, x1 - padding),
        max(0.0, y1 - padding),
        min(1.0, x2 + padding),
        min(1.0, y2 + padding),
    ]


def crop_image(image_path: Path, bbox: list[float], output_path: Path) -> tuple[bytes, str]:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        x1, y1, x2, y2 = bbox
        left = max(0, min(width - 1, math.floor(x1 * width)))
        top = max(0, min(height - 1, math.floor(y1 * height)))
        right = max(left + 1, min(width, math.ceil(x2 * width)))
        bottom = max(top + 1, min(height, math.ceil(y2 * height)))
        cropped = img.crop((left, top, right, bottom))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, format="JPEG", quality=95, optimize=True)
    return output_path.read_bytes(), "image/jpeg"


def call_model(client: OpenAI, model: str, messages: list[dict], max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def run_smoke_test(args: argparse.Namespace) -> dict:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running.")

    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_anchor_bytes, global_anchor_mime, global_anchor_metadata = create_image_global_anchor(str(image_path))
    width = global_anchor_metadata["source_resolution"]["width"]
    height = global_anchor_metadata["source_resolution"]["height"]
    global_anchor_path = output_dir / "mtec_image_global_anchor.jpg"
    global_anchor_path.write_bytes(global_anchor_bytes)
    image_url = encode_image(global_anchor_bytes, global_anchor_mime)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    stage1_prompt = f"""
You are testing a MTEC-Prompt++ / Zoom-Refine style image understanding pipeline.
The image is image_anchor_global: a low-resolution whole-image structural anchor.

Question: {args.question}

First, answer the question briefly.
Then identify the most task-relevant image region as normalized coordinates.
Return the bounding box exactly as:
Bounding Box: [x1, y1, x2, y2]

Use coordinates between 0 and 1, where x1 < x2 and y1 < y2.
""".strip()

    messages_stage1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": stage1_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    stage1 = call_model(client, args.model, messages_stage1, args.max_tokens)
    bbox = parse_bbox(stage1)
    if bbox is None:
        raise RuntimeError(f"Could not parse bbox from stage 1 response:\n{stage1}")

    expanded_bbox = expand_bbox(bbox, args.padding)
    crop_path = output_dir / "zoomrefine_crop.jpg"
    crop_bytes, crop_mime = crop_image(image_path, expanded_bbox, crop_path)
    crop_url = encode_image(crop_bytes, crop_mime)

    crop_anchor_metadata = build_image_crop_anchor_metadata(
        crop_bytes=crop_bytes,
        bbox_norm=bbox,
        expanded_bbox_norm=expanded_bbox,
        original_width=width,
        original_height=height,
    )
    structured_prompt = build_structured_evidence_prompt(
        question=args.question,
        stage1_response=stage1,
        bbox_norm=bbox,
        expanded_bbox_norm=expanded_bbox,
        global_anchor=global_anchor_metadata,
        crop_anchor=crop_anchor_metadata,
    )
    structured_prompt_text = format_structured_evidence_prompt(structured_prompt, compact=True)

    stage2_prompt = f"""
I am providing image_anchor_crop_1, a high-resolution crop from the original image.
Treat it as additional local detail evidence. It may contain the key region.

MTEC-Prompt++ structured evidence prompt with anchor links:
{structured_prompt_text}

Original question: {args.question}
Stage 1 response:
{stage1}

Review the original image context and the cropped image detail.
If the first response missed or misread something, correct it.
Finish with a concise final answer under the heading:
Refined Answer:
""".strip()

    messages_stage2 = [
        messages_stage1[0],
        {"role": "assistant", "content": stage1},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": stage2_prompt},
                {"type": "image_url", "image_url": {"url": crop_url}},
            ],
        },
    ]

    stage2 = call_model(client, args.model, messages_stage2, args.max_tokens)

    result = {
        "image": str(image_path),
        "image_size": {"width": width, "height": height},
        "model": args.model,
        "base_url": args.base_url,
        "question": args.question,
        "global_anchor_path": str(global_anchor_path),
        "compression_target": structured_prompt["compression_target"],
        "low_resolution_anchor": structured_prompt["low_resolution_anchor"],
        "structured_evidence_prompt": structured_prompt["structured_evidence_prompt"],
        "stage1_response": stage1,
        "bbox": bbox,
        "expanded_bbox": expanded_bbox,
        "crop_path": str(crop_path),
        "stage2_response": stage2,
    }

    result_path = output_dir / "zoomrefine_smoke_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-image Zoom-Refine API smoke test.")
    parser.add_argument("--image", required=True, help="Path to the test image.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/smoke_test")
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()

    result = run_smoke_test(args)
    print("Zoom-Refine smoke test completed.")
    print(f"Model: {result['model']}")
    print(f"Image size: {result['image_size']['width']}x{result['image_size']['height']}")
    print(f"BBox: {result['bbox']}")
    print(f"Expanded BBox: {result['expanded_bbox']}")
    print(f"Crop: {result['crop_path']}")
    print("Stage 1 response:")
    print(result["stage1_response"])
    print("Stage 2 response:")
    print(result["stage2_response"])


if __name__ == "__main__":
    main()
