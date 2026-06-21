import json
import re
from collections import Counter
from pathlib import Path


PATHS = {
    "image": Path("outputs/highres_balanced_qwen36_35b_160_img/modelscope_mtec_anchor_api_full_results.jsonl"),
    "video": Path("outputs/highres720_unique_sized_video_qwen36_35b_12/modelscope_mtec_anchor_api_full_results.jsonl"),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_bucket(record):
    key = record.get("record_key", "")
    if "test_2d" in key:
        return "CV-Bench-2D"
    if "test_3d" in key:
        return "CV-Bench-3D"
    if "test-00000" in key or "test-00001" in key:
        return "RealWorldQA"
    return record.get("dataset") or "unknown"


def answer_type(record):
    ground_truth = str(record.get("ground_truth", "")).strip()
    if re.fullmatch(r"[A-F]", ground_truth, re.IGNORECASE):
        return "mcq_letter"
    if ground_truth.lower() in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"-?\d+(\.\d+)?", ground_truth):
        return "number"
    if ground_truth.lower() in {
        "red",
        "green",
        "blue",
        "yellow",
        "black",
        "white",
        "orange",
        "brown",
        "gray",
        "grey",
        "purple",
        "pink",
    }:
        return "color"
    return "short_text"


def question_type(question):
    text = (question or "").lower()
    if any(word in text for word in ["how many", "number of", "count"]):
        return "counting"
    if any(
        word in text
        for word in ["left", "right", "above", "below", "behind", "front", "closer", "nearer", "farthest", "nearest", "distance"]
    ):
        return "spatial_depth"
    if any(word in text for word in ["color", "red", "green", "blue", "yellow", "black", "white"]):
        return "attribute_color"
    if any(word in text for word in ["text", "sign", "read", "word", "letter", "number on", "label"]):
        return "ocr_text"
    if any(word in text for word in ["before", "after", "first", "then", "when", "during"]):
        return "temporal"
    if any(word in text for word in ["why", "because", "reason"]):
        return "causal"
    if any(word in text for word in ["what is", "what are", "which", "identify"]):
        return "recognition"
    return "other"


def print_breakdown(rows, name, get_bucket):
    correct = Counter()
    total = Counter()
    for row in rows:
        bucket = get_bucket(row)
        total[bucket] += 1
        correct[bucket] += int(row.get("correct") is True)
    print(f"-- by {name}")
    for bucket, count in total.most_common():
        ok = correct[bucket]
        print(f"{bucket}: {ok}/{count} acc={ok / count:.3f} wrong={count - ok}")


def main():
    for name, path in PATHS.items():
        rows = read_jsonl(path)
        ok = sum(row.get("correct") is True for row in rows)
        print(f"\n=== {name} total={len(rows)} correct={ok} acc={ok / len(rows):.4f} ===")
        print_breakdown(rows, "source", source_bucket)
        print_breakdown(rows, "answer_type", answer_type)
        print_breakdown(rows, "question_type", lambda row: question_type(row.get("question", "")))
        print("-- wrong samples")
        for row in [item for item in rows if item.get("correct") is not True][:40]:
            question = re.sub(r"\s+", " ", str(row.get("question", "")))[:180]
            print(
                json.dumps(
                    {
                        "key": row.get("record_key"),
                        "src": source_bucket(row),
                        "qtype": question_type(row.get("question", "")),
                        "atype": answer_type(row),
                        "pred": row.get("Answer"),
                        "gt": row.get("ground_truth"),
                        "q": question,
                        "saving": row.get("token_saving_ratio"),
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
