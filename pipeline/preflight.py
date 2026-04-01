"""
GAIAKTA preflight.py — validates a video file before pipeline ingestion.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from ingest import log_cost, log_lineage


def fail(reason: str) -> None:
    print(f"PREFLIGHT_FAIL reason={reason}")
    sys.exit(1)


def run_preflight(
    video_path: str,
    scan_id: str,
    min_duration: float,
    min_width: int,
    min_height: int,
) -> None:
    # 1. File exists and is readable
    if not os.path.exists(video_path):
        print(f"PREFLIGHT_FAIL reason=FILE_NOT_FOUND path={video_path}")
        sys.exit(1)
    if not os.access(video_path, os.R_OK):
        print(f"PREFLIGHT_FAIL reason=FILE_NOT_FOUND path={video_path}")
        sys.exit(1)

    # 2. File size > 0
    if os.path.getsize(video_path) == 0:
        fail("FILE_EMPTY")

    # 3. Run ffprobe
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        fail("FFPROBE_NOT_FOUND")

    if result.returncode != 0:
        fail("FILE_CORRUPT")

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        fail("FFPROBE_PARSE_ERROR")

    streams = probe.get("streams", [])
    fmt = probe.get("format", {})

    # 4. Find video stream
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        fail("NO_VIDEO_STREAM")

    # 5. Codec check
    codec = video_stream.get("codec_name", "")
    if codec not in ("h264", "hevc"):
        print(f"PREFLIGHT_FAIL reason=CODEC_INVALID codec={codec}")
        sys.exit(1)

    # 6. Duration check — prefer stream duration, fall back to format duration
    raw_duration = video_stream.get("duration") or fmt.get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        fail("FFPROBE_PARSE_ERROR")

    if duration < min_duration:
        print(
            f"PREFLIGHT_FAIL reason=TOO_SHORT "
            f"duration={duration:.1f}s minimum={min_duration:.1f}s"
        )
        sys.exit(1)

    # 7. Resolution check — handle portrait orientation
    try:
        w = int(video_stream["width"])
        h = int(video_stream["height"])
    except (KeyError, ValueError):
        fail("FFPROBE_PARSE_ERROR")

    landscape_ok = w >= min_width and h >= min_height
    portrait_ok = w >= min_height and h >= min_width
    if not (landscape_ok or portrait_ok):
        print(f"PREFLIGHT_FAIL reason=RESOLUTION_TOO_LOW resolution={w}x{h}")
        sys.exit(1)

    # All checks passed
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(
        f"PREFLIGHT_OK duration={duration:.1f}s resolution={w}x{h} "
        f"codec={codec} size={size_mb:.1f}MB"
    )

    log_lineage(
        scan_id,
        "preflight",
        video_path,
        {"duration": duration, "resolution": f"{w}x{h}", "codec": codec},
    )
    log_cost(scan_id, "preflight", "validation_run", 1)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== GAIAKTA preflight.py === {ts}")

    parser = argparse.ArgumentParser(description="GAIAKTA video preflight check")
    parser.add_argument("--video-path", required=True, help="Path to input video file")
    parser.add_argument("--scan-id", required=True, help="Scan job ID")
    parser.add_argument(
        "--min-duration", type=float, default=30.0,
        help="Minimum video duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--min-width", type=int, default=1920,
        help="Minimum video width in pixels (default: 1920)",
    )
    parser.add_argument(
        "--min-height", type=int, default=1080,
        help="Minimum video height in pixels (default: 1080)",
    )
    args = parser.parse_args()

    try:
        run_preflight(
            video_path=args.video_path,
            scan_id=args.scan_id,
            min_duration=args.min_duration,
            min_width=args.min_width,
            min_height=args.min_height,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
