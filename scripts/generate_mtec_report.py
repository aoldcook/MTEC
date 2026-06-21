import argparse
import glob
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPORT_METRICS = (
    "Answer Accuracy",
    "Answerability Rate",
    "Evidence Coverage",
    "Temporal Chain Coverage",
    "Compression Ratio",
    "Latency Reduction",
    "Audio Evidence Coverage",
    "Cross-modal Support Coverage",
    "Visual Support Coverage",
    "Spatial Structure Preservation",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_letter(value: Any) -> Optional[str]:
    if value is None:
        return None
    match = re.search(r"\(?\s*([A-Ea-e])\s*\)?", str(value))
    return match.group(1).upper() if match else None


def extract_ground_truth(item: Dict[str, Any]) -> Optional[str]:
    for message in item.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            return extract_letter(message.get("content"))
    return extract_letter(item.get("ground_truth") or item.get("answer") or item.get("label"))


def extract_prediction(item: Dict[str, Any]) -> Optional[str]:
    for key in ("Rechecked Answer", "Answer", "predicted_answer", "prediction", "final_answer"):
        value = item.get(key)
        if value is not None:
            letter = extract_letter(value)
            if letter:
                return letter
    return None


def extract_prediction_text(item: Dict[str, Any]) -> str:
    for key in ("Rechecked Answer", "Answer", "predicted_answer", "prediction", "final_answer"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def extract_ground_truth_text(item: Dict[str, Any]) -> str:
    for key in ("ground_truth", "answer", "label"):
        value = item.get(key)
        if value is not None:
            return str(value)
    ground_truth = extract_ground_truth(item)
    return ground_truth or ""


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def collect_anchor_stats(data: Any) -> Dict[str, Any]:
    anchors = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("anchor_id") or value.get("anchor_link"):
                anchors.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    image = sum(1 for item in anchors if "image" in str(item.get("type", "")))
    video = sum(1 for item in anchors if "video" in str(item.get("type", "")))
    audio = sum(1 for item in anchors if "audio" in str(item.get("type", "")))
    return {
        "total_anchors": len(anchors),
        "image_anchors": image,
        "video_anchors": video,
        "audio_anchors": audio,
    }


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    answerable = 0
    comparable = 0
    correct = 0
    errors = 0
    examples = []

    for index, item in enumerate(records, start=1):
        if item.get("Error") or item.get("error"):
            errors += 1
        prediction_text = extract_prediction_text(item)
        ground_truth_text = extract_ground_truth_text(item)
        prediction = extract_prediction(item)
        ground_truth = extract_ground_truth(item)
        explicit_correct = item.get("correct")
        if prediction_text or prediction:
            answerable += 1
        if isinstance(explicit_correct, bool):
            comparable += 1
            if explicit_correct:
                correct += 1
        elif prediction and ground_truth:
            comparable += 1
            if prediction == ground_truth:
                correct += 1
        if len(examples) < 8:
            examples.append(
                {
                    "index": index,
                    "image": (item.get("images") or [item.get("image") or ""])[0]
                    if isinstance(item.get("images"), list)
                        else item.get("image", ""),
                    "ground_truth": ground_truth_text or ground_truth or "",
                    "prediction": prediction_text or prediction or "",
                    "error": item.get("Error") or item.get("error") or "",
                }
            )

    accuracy = safe_ratio(correct, comparable)
    answerability = safe_ratio(answerable, total)
    anchors = collect_anchor_stats(records)
    return {
        "kind": "evaluation_records",
        "total": total,
        "answerable": answerable,
        "errors": errors,
        "comparable": comparable,
        "correct": correct,
        "answer_accuracy": accuracy,
        "answerability_rate": answerability,
        "anchor_stats": anchors,
        "examples": examples,
    }


def summarize_smoke_result(data: Dict[str, Any]) -> Dict[str, Any]:
    compression = data.get("compression_target", {})
    budget = compression.get("budget", {}) if isinstance(compression, dict) else {}
    anchors = collect_anchor_stats(data)
    return {
        "kind": "smoke_result",
        "model": data.get("model", ""),
        "question": data.get("question", ""),
        "image": data.get("image", ""),
        "stage1_response": data.get("stage1_response") or data.get("response") or "",
        "stage2_response": data.get("stage2_response") or "",
        "compression_ratio": budget.get("anchor_ratio"),
        "evidence_ratio": budget.get("evidence_ratio"),
        "anchor_stats": anchors,
        "bbox": data.get("bbox") or data.get("Bounding_Box"),
        "crop_path": data.get("crop_path", ""),
    }


def summarize_manifest(data: Dict[str, Any]) -> Dict[str, Any]:
    records = data.get("records", [])
    statuses: Dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "kind": "download_manifest",
        "datasets_dir": data.get("datasets_dir", ""),
        "models_dir": data.get("models_dir", ""),
        "dataset_download_mode": data.get("dataset_download_mode", ""),
        "selected_datasets": data.get("selected_datasets", []),
        "selected_models": data.get("selected_models", []),
        "statuses": statuses,
    }


def summarize_file(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if isinstance(data, list):
        summary = summarize_records([item for item in data if isinstance(item, dict)])
    elif isinstance(data, dict) and "records" in data and "selected_models" in data:
        summary = summarize_manifest(data)
    elif isinstance(data, dict):
        summary = summarize_smoke_result(data)
    else:
        summary = {"kind": "unknown", "note": f"Unsupported top-level type: {type(data).__name__}"}
    summary["path"] = str(path)
    return summary


def pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_metric_cards(summaries: List[Dict[str, Any]]) -> str:
    evals = [item for item in summaries if item.get("kind") == "evaluation_records"]
    total = sum(int(item.get("total", 0)) for item in evals)
    correct = sum(int(item.get("correct", 0)) for item in evals)
    comparable = sum(int(item.get("comparable", 0)) for item in evals)
    answerable = sum(int(item.get("answerable", 0)) for item in evals)
    anchors = sum(int(item.get("anchor_stats", {}).get("total_anchors", 0)) for item in summaries)
    accuracy = safe_ratio(correct, comparable)
    answerability = safe_ratio(answerable, total)
    cards = [
        ("Answer Accuracy", pct(accuracy), f"{correct}/{comparable} comparable"),
        ("Answerability Rate", pct(answerability), f"{answerable}/{total} answered"),
        ("Anchor Evidence", str(anchors), "logged image/video/audio anchors"),
        ("Result Files", str(len(summaries)), "JSON inputs summarized"),
    ]
    return "\n".join(
        f"<section class='card'><h3>{text(label)}</h3><strong>{text(value)}</strong><p>{text(note)}</p></section>"
        for label, value, note in cards
    )


def render_summary_table(summaries: List[Dict[str, Any]]) -> str:
    rows = []
    for item in summaries:
        kind = item.get("kind")
        if kind == "evaluation_records":
            detail = (
                f"accuracy {pct(item.get('answer_accuracy'))}; "
                f"answerability {pct(item.get('answerability_rate'))}; "
                f"errors {item.get('errors', 0)}"
            )
        elif kind == "smoke_result":
            detail = f"model {item.get('model')}; anchors {item.get('anchor_stats', {}).get('total_anchors', 0)}"
        elif kind == "download_manifest":
            detail = f"models {item.get('selected_models')}; statuses {item.get('statuses')}"
        else:
            detail = item.get("note", "")
        rows.append(
            "<tr>"
            f"<td>{text(kind)}</td>"
            f"<td>{text(item.get('path'))}</td>"
            f"<td>{text(detail)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_examples(summaries: List[Dict[str, Any]]) -> str:
    rows = []
    for summary in summaries:
        if summary.get("kind") != "evaluation_records":
            continue
        for example in summary.get("examples", []):
            rows.append(
                "<tr>"
                f"<td>{text(Path(summary['path']).name)} #{example['index']}</td>"
                f"<td>{text(example.get('image'))}</td>"
                f"<td>{text(example.get('ground_truth'))}</td>"
                f"<td>{text(example.get('prediction'))}</td>"
                f"<td>{text(example.get('error'))}</td>"
                "</tr>"
            )
    return "\n".join(rows) or "<tr><td colspan='5'>No evaluation examples available.</td></tr>"


def render_smoke_sections(summaries: List[Dict[str, Any]]) -> str:
    sections = []
    for item in summaries:
        if item.get("kind") != "smoke_result":
            continue
        sections.append(
            "<section class='panel'>"
            f"<h2>{text(Path(item['path']).name)}</h2>"
            f"<p><b>Model:</b> {text(item.get('model'))}</p>"
            f"<p><b>Question:</b> {text(item.get('question'))}</p>"
            f"<p><b>BBox:</b> {text(item.get('bbox'))}</p>"
            "<h3>Stage 1 / Response</h3>"
            f"<pre>{text(item.get('stage1_response'))}</pre>"
            "<h3>Stage 2 / Refined Response</h3>"
            f"<pre>{text(item.get('stage2_response'))}</pre>"
            "</section>"
        )
    return "\n".join(sections)


def render_missing_metrics_note() -> str:
    return "\n".join(f"<li>{text(metric)}</li>" for metric in REPORT_METRICS)


def render_html(title: str, summaries: List[Dict[str, Any]]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{text(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f7f8fa; }}
    header {{ padding: 28px 32px 18px; background: #17202a; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #cad2dc; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 20px; }}
    .card, .panel {{ background: white; border: 1px solid #dce3ea; border-radius: 8px; padding: 16px; }}
    .card h3 {{ margin: 0 0 10px; font-size: 13px; color: #53606d; text-transform: uppercase; }}
    .card strong {{ display: block; font-size: 28px; margin-bottom: 6px; color: #111827; }}
    .card p {{ margin: 0; color: #697586; }}
    h2 {{ margin: 0 0 12px; font-size: 19px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dce3ea; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #edf1f5; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #eef3f7; color: #374151; }}
    pre {{ white-space: pre-wrap; background: #101820; color: #e5eef7; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    .two {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }}
    ul {{ columns: 2; padding-left: 22px; }}
    @media (max-width: 760px) {{ main {{ padding: 14px; }} ul {{ columns: 1; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{text(title)}</h1>
    <p>Generated at {text(generated_at)}. Metrics follow the MTEC-Prompt++ evaluation plan where the input files expose enough information.</p>
  </header>
  <main>
    <div class="grid">{render_metric_cards(summaries)}</div>
    <section class="panel">
      <h2>Run Files</h2>
      <table><thead><tr><th>Kind</th><th>Path</th><th>Summary</th></tr></thead><tbody>{render_summary_table(summaries)}</tbody></table>
    </section>
    <section class="panel">
      <h2>Evaluation Examples</h2>
      <table><thead><tr><th>Item</th><th>Image</th><th>GT</th><th>Prediction</th><th>Error</th></tr></thead><tbody>{render_examples(summaries)}</tbody></table>
    </section>
    {render_smoke_sections(summaries)}
    <section class="panel">
      <h2>Metric Checklist</h2>
      <p>The report is ready for the full experiment matrix. Unavailable metrics stay blank until the runner writes them into result JSON.</p>
      <ul>{render_missing_metrics_note()}</ul>
    </section>
  </main>
</body>
</html>
"""


def expand_inputs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(path.resolve() for path in paths if path.exists()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML report for MTEC-Prompt++ experiment outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON result files or glob patterns.")
    parser.add_argument("--output", default="outputs/reports/mtec_report.html", help="Output HTML path.")
    parser.add_argument("--summary-json", default="", help="Optional machine-readable summary JSON path.")
    parser.add_argument("--title", default="MTEC-Prompt++ Evaluation Report")
    args = parser.parse_args()

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No input JSON files found.")

    summaries = [summarize_file(path) for path in input_paths]
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(args.title, summaries), encoding="utf-8")

    summary_json = Path(args.summary_json).resolve() if args.summary_json else output_path.with_suffix(".summary.json")
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Report: {output_path}")
    print(f"Summary JSON: {summary_json}")
    print(f"Inputs summarized: {len(summaries)}")


if __name__ == "__main__":
    main()
