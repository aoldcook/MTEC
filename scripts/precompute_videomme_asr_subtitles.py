import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_modelscope_mtec_anchor_api_full import (  # noqa: E402
    downloaded_subtitles,
    downloaded_videos,
    extract_zip_member,
)
from zoomrefine.mtec_media_pipeline import _extract_audio_wav_for_asr  # noqa: E402


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def selected_video_ids(
    metadata_path: Path,
    available_ids: Set[str],
    requested_ids: Set[str],
    unique: bool,
    limit: int,
) -> List[str]:
    meta = pd.read_parquet(metadata_path)
    ids: List[str] = []
    seen: Set[str] = set()
    for _, row in meta.iterrows():
        video_id = str(row.get("videoID") or "")
        if not video_id or video_id not in available_ids:
            continue
        if requested_ids and video_id not in requested_ids:
            continue
        if unique and video_id in seen:
            continue
        seen.add(video_id)
        ids.append(video_id)
        if limit and len(ids) >= limit:
            break
    return ids


def srt_timestamp(seconds: Any) -> str:
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        value = 0.0
    millis = int(round(value * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(path: Path, segments: Iterable[Dict[str, Any]]) -> int:
    lines: List[str] = []
    count = 0
    for segment in segments:
        time_range = segment.get("time_range_sec") or []
        if not isinstance(time_range, (list, tuple)) or len(time_range) < 2:
            continue
        text = " ".join(str(segment.get("text") or "").split())
        if not text:
            continue
        count += 1
        lines.extend(
            [
                str(count),
                f"{srt_timestamp(time_range[0])} --> {srt_timestamp(time_range[1])}",
                text,
                "",
            ]
        )
    if count:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    return count


def run_faster_whisper(
    video_path: Path,
    work_dir: Path,
    video_id: str,
    model_name: str,
    language: Optional[str],
    max_segments: int,
    vad_filter: bool,
) -> Dict[str, Any]:
    from faster_whisper import WhisperModel

    wav_path, audio_warning = _extract_audio_wav_for_asr(
        str(video_path),
        work_dir / "audio" / f"{video_id}.wav",
    )
    warnings = [audio_warning] if audio_warning else []
    if not wav_path:
        return {
            "source": "none",
            "segments": [],
            "warnings": warnings + ["Audio extraction produced no wav file."],
        }

    device = "cpu"
    compute_type = "int8"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
    except Exception:
        pass

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=1,
        vad_filter=vad_filter,
        word_timestamps=False,
    )
    segments: List[Dict[str, Any]] = []
    for index, segment in enumerate(raw_segments, start=1):
        text = " ".join(str(segment.text).split())
        if not text:
            continue
        segments.append(
            {
                "anchor_id": f"video_transcript_anchor_seg_{index:04d}",
                "anchor_link": f"video_transcript_anchor_seg_{index:04d}",
                "sequence_index": index,
                "time_range_sec": [round(float(segment.start), 3), round(float(segment.end), 3)],
                "text": text,
                "source": "faster_whisper_asr",
            }
        )
        if len(segments) >= max_segments:
            break
    detected = getattr(info, "language", None)
    if not segments:
        warnings.append(f"faster-whisper produced no segments; detected_language={detected}; vad_filter={vad_filter}")
    return {
        "source": "faster_whisper_asr" if segments else "none",
        "segments": segments,
        "warnings": warnings,
        "detected_language": detected,
        "wav_path": str(wav_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute missing Video-MME subtitles with faster-whisper ASR.")
    parser.add_argument("--videomme-metadata", default="data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
    parser.add_argument("--video-zips-dir", default="data/modelscope/video-mme-zips")
    parser.add_argument("--videomme-subtitle-zip", default="data/datasets/video-mme/subtitle.zip")
    parser.add_argument("--output-dir", default="outputs/videomme_asr_subtitles")
    parser.add_argument("--work-dir", default="outputs/videomme_asr_work")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="auto", help="Use 'auto' to let Whisper detect language.")
    parser.add_argument("--vad-filter", choices=("true", "false"), default="false")
    parser.add_argument("--max-segments", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--unique-video-ids", action="store_true")
    parser.add_argument("--video-ids", nargs="*", default=[])
    parser.add_argument("--include-existing-subtitles", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_lookup = downloaded_videos(resolve_path(args.video_zips_dir))
    official_subtitles = downloaded_subtitles(resolve_path(args.videomme_subtitle_zip))
    requested_ids = set(str(item) for item in args.video_ids if str(item).strip())
    video_ids = selected_video_ids(
        resolve_path(args.videomme_metadata),
        set(video_lookup.keys()),
        requested_ids=requested_ids,
        unique=args.unique_video_ids,
        limit=args.limit,
    )

    output_dir = resolve_path(args.output_dir)
    work_dir = resolve_path(args.work_dir)
    language: Optional[str] = None if args.language.lower() == "auto" else args.language
    diagnostics: List[Dict[str, Any]] = []

    for index, video_id in enumerate(video_ids, start=1):
        srt_path = output_dir / f"{video_id}.srt"
        diag_path = output_dir / f"{video_id}.json"
        if video_id in official_subtitles and not args.include_existing_subtitles:
            diagnostics.append({"videoID": video_id, "status": "skipped_official_subtitle_exists"})
            continue
        if srt_path.exists() and srt_path.stat().st_size > 0 and not args.overwrite:
            diagnostics.append({"videoID": video_id, "status": "skipped_existing_asr_subtitle", "srt_path": str(srt_path)})
            continue

        print(f"[{index}/{len(video_ids)}] ASR {video_id}", flush=True)
        try:
            zip_path, member = video_lookup[video_id]
            video_path = extract_zip_member(zip_path, member, work_dir / "media" / "video")
            transcript = run_faster_whisper(
                video_path=video_path,
                work_dir=work_dir,
                video_id=video_id,
                model_name=args.model,
                language=language,
                max_segments=args.max_segments,
                vad_filter=args.vad_filter == "true",
            )
            segment_count = write_srt(srt_path, transcript.get("segments") or [])
            record = {
                "videoID": video_id,
                "status": "completed" if segment_count else "no_segments",
                "segment_count": segment_count,
                "srt_path": str(srt_path) if segment_count else None,
                "source": transcript.get("source"),
                "warnings": transcript.get("warnings") or [],
                "detected_language": transcript.get("detected_language"),
                "wav_path": transcript.get("wav_path"),
            }
        except Exception as exc:
            record = {
                "videoID": video_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        diagnostics.append(record)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "total": len(diagnostics),
        "completed": sum(1 for item in diagnostics if item.get("status") == "completed"),
        "no_segments": sum(1 for item in diagnostics if item.get("status") == "no_segments"),
        "failed": sum(1 for item in diagnostics if item.get("status") == "failed"),
        "skipped": sum(1 for item in diagnostics if str(item.get("status", "")).startswith("skipped")),
        "model": args.model,
        "language": args.language,
        "vad_filter": args.vad_filter,
        "output_dir": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary, "records": diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
