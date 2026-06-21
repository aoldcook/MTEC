import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_records(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict) and isinstance(data.get("records"), list):
            records.extend(item for item in data["records"] if isinstance(item, dict))
        elif isinstance(data, dict):
            records.append(data)
    return records


def text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def shorten(value: Any, limit: int = 260) -> str:
    raw = "" if value is None else str(value).strip()
    raw = " ".join(raw.split())
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1].rstrip() + "..."


def localize_path(value: Any) -> str:
    if not value:
        return ""
    path = str(value)
    remote_prefix = "/root/autodl-tmp/MTEC/"
    if path.startswith(remote_prefix):
        path = path[len(remote_prefix) :]
    return path


def result_text(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def nested_value(data: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_value(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = nested_value(data, key) if "." in key else data.get(key)
        if value is not None:
            return value
    return None


def ratio_text(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if 0 <= number <= 1:
        return f"{number * 100:.1f}%"
    return f"{number:.3g}"


def compression_ratio(record: Dict[str, Any]) -> Any:
    return first_value(
        record,
        (
            "compression_ratio",
            "anchor_ratio",
            "evidence_ratio",
            "compression_target.budget.anchor_ratio",
            "compression_target.budget.evidence_ratio",
        ),
    )


def token_saving_ratio(record: Dict[str, Any]) -> Any:
    explicit = first_value(
        record,
        (
            "token_saving_ratio",
            "token_reduction_ratio",
            "tokens_saved_ratio",
            "compression_target.budget.token_saving_ratio",
            "compression_target.budget.token_reduction_ratio",
        ),
    )
    if explicit is not None:
        return explicit
    original = first_value(record, ("original_tokens", "source_tokens", "raw_tokens", "compression_target.budget.original_tokens"))
    compressed = first_value(
        record,
        ("compressed_tokens", "anchor_tokens", "prompt_tokens", "compression_target.budget.compressed_tokens"),
    )
    try:
        original_number = float(original)
        compressed_number = float(compressed)
    except (TypeError, ValueError):
        return None
    if original_number <= 0:
        return None
    return max(0.0, 1.0 - compressed_number / original_number)


def media_link(record: Dict[str, Any]) -> str:
    for key in ("image", "video", "audio_file"):
        path = localize_path(record.get(key))
        if not path:
            continue
        label = Path(path).name or path
        return f'<a href="../{text(path)}">{text(label)}</a>'
    return ""


def render_table(records: List[Dict[str, Any]], title: str) -> str:
    rows = []
    for index, record in enumerate(records, start=1):
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{text(record.get('modality'))}</td>"
            f"<td>{text(record.get('dataset'))}</td>"
            f"<td>{text(record.get('status'))}</td>"
            f"<td>{media_link(record)}</td>"
            f"<td>{text(shorten(record.get('question')))}</td>"
            f"<td>{text(record.get('Answer') or record.get('prediction') or record.get('raw_response'))}</td>"
            f"<td>{text(record.get('ground_truth'))}</td>"
            f"<td>{text(ratio_text(compression_ratio(record)))}</td>"
            f"<td>{text(ratio_text(token_saving_ratio(record)))}</td>"
            f"<td>{text(result_text(record.get('correct')))}</td>"
            f"<td>{text(round(float(record.get('elapsed_seconds', 0)), 3) if record.get('elapsed_seconds') is not None else '')}</td>"
            f"<td>{text(shorten(record.get('note') or record.get('Error') or record.get('error'), 180))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{text(title)}</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; color: #111827; background: #ffffff; }}
    h1 {{ margin: 0 0 16px; font-size: 22px; font-weight: 650; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; vertical-align: top; font-size: 13px; line-height: 1.4; word-break: break-word; }}
    th {{ background: #f3f4f6; text-align: left; font-weight: 650; }}
    th:nth-child(1), td:nth-child(1) {{ width: 38px; }}
    th:nth-child(2), td:nth-child(2) {{ width: 96px; }}
    th:nth-child(4), td:nth-child(4) {{ width: 90px; }}
    th:nth-child(8), td:nth-child(8) {{ width: 110px; }}
    th:nth-child(9), td:nth-child(9) {{ width: 96px; }}
    th:nth-child(10), td:nth-child(10) {{ width: 86px; }}
    th:nth-child(11), td:nth-child(11) {{ width: 70px; }}
    th:nth-child(12), td:nth-child(12) {{ width: 72px; }}
    a {{ color: #075985; }}
  </style>
</head>
<body>
  <h1>{text(title)}</h1>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Modality</th>
        <th>Dataset</th>
        <th>Status</th>
        <th>Media</th>
        <th>Question / Prompt</th>
        <th>Prediction</th>
        <th>Ground Truth</th>
        <th>Compression Ratio</th>
        <th>Token Saving</th>
        <th>Correct</th>
        <th>Seconds</th>
        <th>Note</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple table-only HTML report for MTEC result JSON.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/reports/mtec_table_report.html")
    parser.add_argument("--title", default="MTEC-Prompt++ Results Table")
    args = parser.parse_args()

    input_paths = [Path(path).resolve() for path in args.inputs]
    records = load_records(input_paths)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_table(records, args.title), encoding="utf-8")
    print(f"Table report: {output_path}")
    print(f"Rows: {len(records)}")


if __name__ == "__main__":
    main()
