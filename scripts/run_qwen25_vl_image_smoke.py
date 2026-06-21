import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zoomrefine.mtec_prompt_plus import create_image_global_anchor  # noqa: E402


DEFAULT_QUESTION = "Describe the main visible subject in this image and answer concisely."


def patch_torch_pytree_compat() -> None:
    """Allow newer Transformers builds to run on torch 2.1 images."""
    if not hasattr(torch, "compiler"):
        torch.compiler = types.SimpleNamespace()
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False

    try:
        import torch.utils._pytree as pytree
    except Exception:
        return
    if not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):
        def register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs):
            return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

        pytree.register_pytree_node = register_pytree_node


def resolve_project_path(path_text: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_text)))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def parse_bbox(text: str) -> Optional[List[float]]:
    match = re.search(
        r"(?:bbox|bounding box)\s*[:：]?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    coords = [max(0.0, min(1.0, float(value))) for value in match.groups()]
    return coords if coords[0] < coords[2] and coords[1] < coords[3] else None


def load_qwen_vl_dependencies():
    patch_torch_pytree_compat()
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Qwen2.5-VL local inference requires a recent transformers build "
            "with Qwen2_5_VLForConditionalGeneration support."
        ) from exc
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError("Install qwen-vl-utils before running local Qwen2.5-VL image inference.") from exc
    return AutoProcessor, Qwen2_5_VLForConditionalGeneration, process_vision_info


def build_messages(image_path: Path, question: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are evaluating the MTEC-Prompt++ image pipeline.
Use the image as a visual structural anchor and answer the question.

Question: {question}

Return:
Answer: <concise answer>
Bounding Box: [x1, y1, x2, y2]

Use normalized bbox coordinates between 0 and 1 for the most relevant region.
""".strip()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = resolve_project_path(args.model_path)
    image_path = resolve_project_path(args.image)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    AutoProcessor, ModelClass, process_vision_info = load_qwen_vl_dependencies()
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    anchor_bytes, anchor_mime, anchor_metadata = create_image_global_anchor(
        str(image_path),
        max_side=args.anchor_max_side,
        quality=args.anchor_quality,
    )
    anchor_path = output_dir / "qwen25_vl_global_anchor.jpg"
    anchor_path.write_bytes(anchor_bytes)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False)
    model = ModelClass.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_start

    messages = build_messages(anchor_path, args.question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    infer_start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    inference_seconds = time.perf_counter() - infer_start

    result = {
        "model": str(model_path),
        "image": str(image_path),
        "question": args.question,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "global_anchor_path": str(anchor_path),
        "global_anchor_mime": anchor_mime,
        "low_resolution_anchor": {"image_anchor": [anchor_metadata], "video_anchor": [], "audio_anchor": []},
        "compression_target": {
            "budget": {
                "anchor_ratio": round(len(anchor_bytes) / max(1, image_path.stat().st_size), 4),
                "evidence_ratio": None,
                "policy": "Smoke test reports byte ratio of global image anchor vs source image.",
            }
        },
        "stage1_response": response,
        "bbox": parse_bbox(response),
        "timing": {
            "load_seconds": round(load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
        },
    }

    result_path = output_dir / "qwen25_vl_3b_smoke_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Qwen2.5-VL image smoke test for MTEC-Prompt++.")
    parser.add_argument("--model-path", default="models/qwen2.5-vl-3b")
    parser.add_argument("--image", default="asset/Example.jpg")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--output-dir", default="outputs/qwen25_vl_3b_smoke")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--anchor-max-side", type=int, default=768)
    parser.add_argument("--anchor-quality", type=int, default=82)
    args = parser.parse_args()

    result = run(args)
    print("Qwen2.5-VL image smoke test completed.")
    print(f"Model: {result['model']}")
    print(f"Device: {result['device']} ({result['dtype']})")
    print(f"Anchor: {result['global_anchor_path']}")
    print(f"BBox: {result['bbox']}")
    print("Response:")
    print(result["stage1_response"])


if __name__ == "__main__":
    main()
