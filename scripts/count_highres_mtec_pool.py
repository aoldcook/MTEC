import io
import json
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_PARQUETS = [
    Path("data/datasets/cv-bench/test_2d.parquet"),
    Path("data/datasets/cv-bench/test_3d.parquet"),
    Path("data/modelscope/realworldqa/data/test-00000-of-00002.parquet"),
    Path("data/modelscope/realworldqa/data/test-00001-of-00002.parquet"),
]
THRESHOLDS = [
    (1000, 720, 1.0, 200_000),
    (900, 600, 0.8, 150_000),
    (800, 600, 0.6, 120_000),
]


def image_stats(image_cell):
    image_bytes = image_cell.get("bytes") if isinstance(image_cell, dict) else None
    if not image_bytes:
        return None
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    return int(width), int(height), round(width * height / 1_000_000, 4), len(image_bytes)


def main():
    summary = {"images": [], "videos": {}}
    for path in IMAGE_PARQUETS:
        if not path.exists():
            summary["images"].append({"path": str(path), "exists": False})
            continue
        df = pd.read_parquet(path)
        rows = []
        for index, row in df.iterrows():
            stats = image_stats(row.to_dict().get("image") or {})
            if stats:
                rows.append((*stats, int(index)))
        threshold_counts = []
        for min_width, min_height, min_mp, min_bytes in THRESHOLDS:
            passed = [
                item
                for item in rows
                if item[0] >= min_width
                and item[1] >= min_height
                and item[2] >= min_mp
                and item[3] >= min_bytes
            ]
            threshold_counts.append(
                {
                    "min_width": min_width,
                    "min_height": min_height,
                    "min_mp": min_mp,
                    "min_bytes": min_bytes,
                    "count": len(passed),
                }
            )
        top_samples = sorted(rows, key=lambda item: (item[2], item[3]), reverse=True)[:5]
        summary["images"].append(
            {
                "path": str(path),
                "rows": len(df),
                "valid_images": len(rows),
                "threshold_counts": threshold_counts,
                "top_samples": top_samples,
            }
        )

    meta_path = Path("data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
    zips_dir = Path("data/modelscope/video-mme-zips")
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        zip_ids = set()
        if zips_dir.exists():
            for zip_path in zips_dir.glob("*.zip"):
                try:
                    with zipfile.ZipFile(zip_path) as archive:
                        for name in archive.namelist():
                            if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                                zip_ids.add(Path(name).stem)
                except Exception:
                    continue
        available = meta
        if "videoID" in meta.columns and zip_ids:
            available = meta[meta["videoID"].astype(str).isin(zip_ids)]
        summary["videos"] = {
            "rows": len(meta),
            "available_rows": len(available),
            "duration_counts_all": meta["duration"].astype(str).str.lower().value_counts().to_dict()
            if "duration" in meta.columns
            else {},
            "duration_counts_available": available["duration"].astype(str).str.lower().value_counts().to_dict()
            if "duration" in available.columns
            else {},
            "medium_long_available": int(
                available[available["duration"].astype(str).str.lower().isin(["medium", "long"])].shape[0]
            )
            if "duration" in available.columns
            else 0,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
