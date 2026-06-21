import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from zoomrefine.mtec_media_pipeline import create_multimodal_structural_anchors


root = Path("/root/autodl-tmp/MTEC")
meta = pd.read_parquet(root / "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
lookup = {}
for zip_path in sorted((root / "data/modelscope/video-mme-zips").glob("videos_chunked_*.zip")):
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.filename.endswith(".mp4") and 20_000_000 <= info.file_size <= 120_000_000:
                lookup[Path(info.filename).stem] = (zip_path, info)

row = None
for _, item in meta.iterrows():
    video_id = str(item.get("videoID"))
    if str(item.get("duration")).lower() in {"medium", "long"} and video_id in lookup:
        row = item.to_dict()
        break
assert row, "no video row found"

video_id = str(row["videoID"])
zip_path, member = lookup[video_id]
output_dir = root / "outputs/transcript_smoke"
media_dir = output_dir / "media"
media_dir.mkdir(parents=True, exist_ok=True)
video_path = media_dir / Path(member.filename).name
with zipfile.ZipFile(zip_path) as archive, archive.open(member) as source, video_path.open("wb") as target:
    shutil.copyfileobj(source, target)

question = str(row.get("question")) + "\n" + str(row.get("options"))
package = create_multimodal_structural_anchors(
    question,
    str(output_dir / "anchors"),
    video_path=str(video_path),
    video_target_fps=3,
    video_max_frames=48,
    include_video_audio=True,
    include_video_transcript=True,
    video_transcript_backend="faster-whisper",
    video_asr_model="base.en",
    video_asr_language="en",
    video_transcript_max_segments=20,
)
transcript = (package.get("low_resolution_anchor", {}).get("transcript_anchor") or [{}])[0]
print(
    json.dumps(
        {
            "videoID": video_id,
            "source": transcript.get("source"),
            "segments": len(transcript.get("segments") or []),
            "warnings": transcript.get("warnings"),
            "preview": (transcript.get("text_preview") or "")[:500],
        },
        ensure_ascii=False,
        indent=2,
    )
)
