import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tool.path_tool import get_abs_path, get_project_root  # noqa: E402


@dataclass(frozen=True)
class HubAsset:
    key: str
    repo_id: str
    repo_type: str
    group: str
    description: str
    metadata_allow_patterns: Tuple[str, ...] = ()
    gated: bool = False
    access_note: str = ""


FIRST_STAGE_DATASETS = ("cv-bench", "realworldqa", "video-mme", "music-avqa")
ADVANCED_DATASETS = ("mme-realworld-lite", "hr-bench", "longvideobench", "avqa")
RECOMMENDED_MODELS = (
    "qwen2.5-vl-3b",
    "qwen2.5-omni-7b",
    "qwen2.5-vl-32b",
)


DATASETS: Dict[str, HubAsset] = {
    "cv-bench": HubAsset(
        key="cv-bench",
        repo_id="nyu-visionx/CV-Bench",
        repo_type="dataset",
        group="first-stage",
        description="Image spatial structure benchmark for CV-Bench.",
        metadata_allow_patterns=("README.md", "*.jsonl", "build_img.py"),
    ),
    "realworldqa": HubAsset(
        key="realworldqa",
        repo_id="lmms-lab/RealWorldQA",
        repo_type="dataset",
        group="first-stage",
        description="Real-world image QA benchmark.",
        metadata_allow_patterns=("README.md",),
    ),
    "video-mme": HubAsset(
        key="video-mme",
        repo_id="lmms-lab/Video-MME",
        repo_type="dataset",
        group="first-stage",
        description="Video, subtitle, and audio benchmark for temporal evidence.",
        metadata_allow_patterns=("README.md", "videomme/**", "subtitle.zip"),
    ),
    "music-avqa": HubAsset(
        key="music-avqa",
        repo_id="UnFaZeD07/Music-AVQA",
        repo_type="dataset",
        group="first-stage",
        description="Audio-visual QA dataset for music videos.",
        metadata_allow_patterns=(
            "README.md",
            "*.json",
            "*.jsonl",
            "*.csv",
            "*.txt",
            "data/**/*.json",
            "data/**/*.jsonl",
            "data/**/*.csv",
            "data/**/*.txt",
        ),
    ),
    "mme-realworld-lite": HubAsset(
        key="mme-realworld-lite",
        repo_id="yifanzhang114/MME-RealWorld-Lite",
        repo_type="dataset",
        group="advanced",
        description="Lite split of MME-RealWorld for high-resolution image QA.",
        metadata_allow_patterns=("README.md", "*.tsv", "*.json", "*.jsonl", "*.csv"),
    ),
    "hr-bench": HubAsset(
        key="hr-bench",
        repo_id="DreamMr/HR-Bench",
        repo_type="dataset",
        group="advanced",
        description="4K/8K high-resolution image benchmark.",
        metadata_allow_patterns=("README.md", "LICENSE", "*.parquet"),
    ),
    "longvideobench": HubAsset(
        key="longvideobench",
        repo_id="longvideobench/LongVideoBench",
        repo_type="dataset",
        group="advanced",
        description="Long video understanding benchmark.",
        metadata_allow_patterns=("README.md", "*.parquet", "*.json", "*.jsonl", "*.csv"),
        gated=True,
        access_note="Accept the Hugging Face dataset terms and pass --token or set HF_TOKEN.",
    ),
    "avqa": HubAsset(
        key="avqa",
        repo_id="Joysw909/AVQA",
        repo_type="dataset",
        group="advanced",
        description="Audio-visual question answering benchmark on videos.",
        metadata_allow_patterns=("README.md", "*.json", "*.jsonl", "*.csv", "*.parquet"),
    ),
}


MODELS: Dict[str, HubAsset] = {
    "qwen2.5-vl-3b": HubAsset(
        key="qwen2.5-vl-3b",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        repo_type="model",
        group="recommended",
        description="Small VLM for quick debugging and ablations.",
    ),
    "qwen2.5-omni-7b": HubAsset(
        key="qwen2.5-omni-7b",
        repo_id="Qwen/Qwen2.5-Omni-7B",
        repo_type="model",
        group="recommended",
        description="Main audio-video-image model for MTEC-Prompt++.",
    ),
    "qwen2.5-vl-32b": HubAsset(
        key="qwen2.5-vl-32b",
        repo_id="Qwen/Qwen2.5-VL-32B-Instruct",
        repo_type="model",
        group="recommended",
        description="Large VLM for final selected experiments.",
    ),
}


def resolve_project_path(path_text: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_text))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return Path(get_abs_path(expanded))


def has_existing_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def select_dataset_keys(args: argparse.Namespace) -> List[str]:
    if args.datasets:
        return list(dict.fromkeys(args.datasets))
    if args.dataset_set == "none":
        return []
    if args.dataset_set == "first":
        return list(FIRST_STAGE_DATASETS)
    if args.dataset_set == "advanced":
        return list(ADVANCED_DATASETS)
    return list(FIRST_STAGE_DATASETS + ADVANCED_DATASETS)


def select_model_keys(args: argparse.Namespace) -> List[str]:
    if args.models:
        return list(dict.fromkeys(args.models))
    if args.model_set == "none":
        return []
    return list(RECOMMENDED_MODELS)


def list_assets() -> None:
    print("Datasets:")
    for key in sorted(DATASETS):
        asset = DATASETS[key]
        gated = " gated" if asset.gated else ""
        print(f"  {key:20s} {asset.repo_id:40s} {asset.group}{gated}")
    print("Models:")
    for key in sorted(MODELS):
        asset = MODELS[key]
        print(f"  {key:20s} {asset.repo_id:40s} {asset.group}")


def import_huggingface_tools():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install project requirements or run "
            "`pip install huggingface_hub` on the server."
        ) from exc

    try:
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        GatedRepoError = RepositoryNotFoundError = HfHubHTTPError = Exception

    return snapshot_download, GatedRepoError, RepositoryNotFoundError, HfHubHTTPError


def planned_record(asset: HubAsset, local_dir: Path, allow_patterns: Optional[Sequence[str]]) -> Dict[str, object]:
    return {
        "key": asset.key,
        "repo_id": asset.repo_id,
        "repo_type": asset.repo_type,
        "group": asset.group,
        "local_dir": str(local_dir),
        "metadata_allow_patterns": list(allow_patterns or []),
        "gated": asset.gated,
        "access_note": asset.access_note,
        "status": "planned",
    }


def download_asset(
    asset: HubAsset,
    base_dir: Path,
    cache_dir: Path,
    token: Optional[str],
    dataset_download_mode: str,
    max_workers: int,
    skip_existing: bool,
    dry_run: bool,
):
    local_dir = base_dir / asset.key
    allow_patterns: Optional[Sequence[str]] = None
    if asset.repo_type == "dataset" and dataset_download_mode == "metadata":
        allow_patterns = asset.metadata_allow_patterns or ("README.md",)

    if dry_run:
        return planned_record(asset, local_dir, allow_patterns)

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and has_existing_files(local_dir):
        record = planned_record(asset, local_dir, allow_patterns)
        record["status"] = "skipped_existing"
        return record

    snapshot_download, GatedRepoError, RepositoryNotFoundError, HfHubHTTPError = import_huggingface_tools()

    kwargs = {
        "repo_id": asset.repo_id,
        "repo_type": asset.repo_type,
        "revision": "main",
        "local_dir": str(local_dir),
        "cache_dir": str(cache_dir),
        "token": token,
        "max_workers": max_workers,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)

    record = planned_record(asset, local_dir, allow_patterns)
    try:
        downloaded_path = snapshot_download(**kwargs)
        record.update(
            {
                "status": "downloaded",
                "downloaded_path": downloaded_path,
            }
        )
    except GatedRepoError as exc:
        record.update(
            {
                "status": "failed_gated",
                "error": str(exc),
                "hint": asset.access_note or "Accept the repo terms and authenticate with --token or HF_TOKEN.",
            }
        )
    except RepositoryNotFoundError as exc:
        record.update(
            {
                "status": "failed_not_found",
                "error": str(exc),
                "hint": "Check the repo id or whether your token can access the repository.",
            }
        )
    except HfHubHTTPError as exc:
        record.update(
            {
                "status": "failed_http",
                "error": str(exc),
                "hint": "Check network access, HF_ENDPOINT, token permissions, or disk quota.",
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error": str(exc),
                "hint": "Check network access, token permissions, or disk quota.",
            }
        )
    return record


def write_manifest(manifest: Dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def print_plan(
    dataset_assets: Iterable[HubAsset],
    model_assets: Iterable[HubAsset],
    datasets_dir: Path,
    models_dir: Path,
    cache_dir: Path,
    dataset_download_mode: str,
) -> None:
    print(f"Project root: {get_project_root()}")
    print(f"Datasets dir: {datasets_dir}")
    print(f"Models dir:   {models_dir}")
    print(f"HF cache dir: {cache_dir}")
    print(f"Dataset download mode: {dataset_download_mode}")
    print("")
    print("Datasets to download:")
    for asset in dataset_assets:
        suffix = " (gated)" if asset.gated else ""
        print(f"  - {asset.key}: {asset.repo_id}{suffix}")
    print("Models to download:")
    for asset in model_assets:
        print(f"  - {asset.key}: {asset.repo_id}")


def run(args: argparse.Namespace) -> Dict[str, object]:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    datasets_dir = resolve_project_path(args.datasets_dir)
    models_dir = resolve_project_path(args.models_dir)
    cache_dir = resolve_project_path(args.cache_dir)
    output_dir = resolve_project_path(args.output_dir)

    dataset_keys = select_dataset_keys(args)
    model_keys = select_model_keys(args)
    dataset_assets = [DATASETS[key] for key in dataset_keys]
    model_assets = [MODELS[key] for key in model_keys]
    token = args.token or os.environ.get("HF_TOKEN")

    print_plan(
        dataset_assets=dataset_assets,
        model_assets=model_assets,
        datasets_dir=datasets_dir,
        models_dir=models_dir,
        cache_dir=cache_dir,
        dataset_download_mode=args.dataset_download_mode,
    )

    records: List[Dict[str, object]] = []
    for asset in dataset_assets:
        print(f"Downloading dataset {asset.key} from {asset.repo_id} ...")
        record = download_asset(
            asset=asset,
            base_dir=datasets_dir,
            cache_dir=cache_dir,
            token=token,
            dataset_download_mode=args.dataset_download_mode,
            max_workers=args.max_workers,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        print(f"  status: {record['status']}")
        records.append(record)
        if args.fail_fast and str(record["status"]).startswith("failed"):
            break

    if not args.fail_fast or not any(str(record["status"]).startswith("failed") for record in records):
        for asset in model_assets:
            print(f"Downloading model {asset.key} from {asset.repo_id} ...")
            record = download_asset(
                asset=asset,
                base_dir=models_dir,
                cache_dir=cache_dir,
                token=token,
                dataset_download_mode=args.dataset_download_mode,
                max_workers=args.max_workers,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            print(f"  status: {record['status']}")
            records.append(record)
            if args.fail_fast and str(record["status"]).startswith("failed"):
                break

    manifest: Dict[str, object] = {
        "project_root": get_project_root(),
        "datasets_dir": str(datasets_dir),
        "models_dir": str(models_dir),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "dataset_download_mode": args.dataset_download_mode,
        "hf_endpoint": args.hf_endpoint or os.environ.get("HF_ENDPOINT"),
        "dry_run": args.dry_run,
        "selected_datasets": dataset_keys,
        "selected_models": model_keys,
        "records": records,
        "registries": {
            "datasets": {key: asdict(value) for key, value in DATASETS.items()},
            "models": {key: asdict(value) for key, value in MODELS.items()},
        },
    }
    manifest_path = write_manifest(manifest, output_dir)
    manifest["manifest_path"] = str(manifest_path)
    print(f"Manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MTEC-Prompt++ datasets and model checkpoints for server deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-set",
        choices=("first", "advanced", "all", "none"),
        default="first",
        help="Preset dataset group from the experiment plan.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(sorted(DATASETS)),
        help="Explicit dataset keys. Overrides --dataset-set.",
    )
    parser.add_argument(
        "--model-set",
        choices=("recommended", "none"),
        default="recommended",
        help="Preset model group from the experiment plan.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(sorted(MODELS)),
        help="Explicit model keys. Overrides --model-set.",
    )
    parser.add_argument(
        "--dataset-download-mode",
        choices=("full", "metadata"),
        default="full",
        help="Use metadata to skip large media archives for quick pipeline checks.",
    )
    parser.add_argument("--datasets-dir", default="data/datasets", help="Project-relative or absolute dataset path.")
    parser.add_argument("--models-dir", default="models", help="Project-relative or absolute model path.")
    parser.add_argument("--cache-dir", default=".cache/huggingface", help="Project-relative or absolute HF cache path.")
    parser.add_argument("--output-dir", default="outputs/download_assets", help="Where to write the manifest.")
    parser.add_argument("--token", help="Hugging Face token. Defaults to HF_TOKEN from the environment.")
    parser.add_argument("--hf-endpoint", help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel file download workers.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip assets when local_dir already has files.")
    parser.add_argument("--dry-run", action="store_true", help="Print and write the plan without downloading files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed asset.")
    parser.add_argument("--list-assets", action="store_true", help="List known datasets and models, then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_assets:
        list_assets()
        return

    manifest = run(args)
    failed = [record for record in manifest["records"] if str(record["status"]).startswith("failed")]
    if failed:
        print(f"Failed assets: {len(failed)}. Check the manifest for details.")
        raise SystemExit(1)
    print("All requested assets are ready." if not args.dry_run else "Dry run completed.")


if __name__ == "__main__":
    main()
