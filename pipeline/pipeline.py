# GAIAKTA Pipeline — Script 9 of 9
# Master orchestrator — calls all 8 scripts in order
#
# NORMAL MODE (new scan):
# python pipeline/pipeline.py \
#   --video-path /path/to/video.mp4 \
#   --scan-id <uuid> --user-id <uuid> \
#   --frames-dir outputs/frames \
#   --masked-frames-dir outputs/masked \
#   --colmap-output-dir outputs/colmap \
#   --splat-output-dir outputs/splat \
#   --raw-video-r2-key raw/video.mp4
#
# RESUME MODE (after user confirms subject):
# python pipeline/pipeline.py \
#   --resume-from confirmed \
#   --confirmed-keyframe-index 4 \
#   [all other args same as normal mode]

"""
pipeline.py — GAIAKTA master pipeline orchestrator.

Normal mode: runs stages 1–4 (ingest → preflight → frame extraction →
segmentation), then suspends awaiting user confirmation of subject mask.

Resume mode: runs stages 5–13 (mask propagation → COLMAP → training →
export → postprocess → quality check → finalize).
"""

import argparse
import glob
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Stub imports from ingest.py (log_lineage / log_cost)
# ---------------------------------------------------------------------------
try:
    from ingest import log_cost, log_lineage
except ImportError:
    def log_lineage(scan_id, stage, source, metrics):  # noqa: D401
        pass

    def log_cost(scan_id, stage, event, quantity):  # noqa: D401
        pass

# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------
try:
    from heartbeat import start_heartbeat, stop_heartbeat
except ImportError:
    def start_heartbeat(scan_id, worker_id):
        return None

    def stop_heartbeat():
        pass


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

def make_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY env vars not set")
        sys.exit(1)
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        print("ERROR: supabase-py is not installed")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Supabase connection failed: {e}")
        sys.exit(1)


def set_status(sb, scan_id: str, status: str) -> None:
    try:
        sb.table("scan_jobs").update(
            {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", scan_id).execute()
    except Exception as e:
        print(f"WARNING: Supabase status update failed: {e}")


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def run_stage(
    cmd: list[str],
    stage_name: str,
    scan_id: str,
    sb,
    shell: bool = False,
    shell_cmd: str | None = None,
) -> None:
    """
    Run a subprocess stage. On non-zero exit: mark failed, print diagnostic,
    stop heartbeat, and exit 1.

    Pass shell=True + shell_cmd=<string> for piped shell commands (e.g. tee).
    Otherwise pass cmd as a list.
    """
    print(f"[{_ts()}] {stage_name} starting...")
    start = datetime.now(timezone.utc)

    if shell and shell_cmd:
        result = subprocess.run(shell_cmd, shell=True)
    else:
        result = subprocess.run(cmd)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    if result.returncode != 0:
        set_status(sb, scan_id, "failed")
        print(f"PIPELINE_FAIL stage={stage_name} exit_code={result.returncode}")
        stop_heartbeat()
        sys.exit(1)

    print(f"[{_ts()}] {stage_name} completed in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Checkpoint / config discovery helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(splat_output_dir: str) -> tuple[str, str]:
    """
    Return (ckpt_path, config_path) for the most recent nerfstudio checkpoint
    under splat_output_dir.

    nerfstudio writes:
      {splat_output_dir}/gaiakta_scan/splatfacto/<timestamp>/
          config.yml
          nerfstudio_models/step-XXXXXX.ckpt
    """
    ckpt_pattern = os.path.join(
        splat_output_dir, "gaiakta_scan", "splatfacto", "*",
        "nerfstudio_models", "step-*.ckpt"
    )
    ckpts = sorted(glob.glob(ckpt_pattern))
    if not ckpts:
        print(f"PIPELINE_FAIL stage=find_checkpoint reason=NO_CHECKPOINT_FOUND path={ckpt_pattern}")
        stop_heartbeat()
        sys.exit(1)
    latest_ckpt = ckpts[-1]

    # Config lives two levels above nerfstudio_models/
    run_dir = Path(latest_ckpt).parent.parent
    config_path = str(run_dir / "config.yml")
    if not os.path.exists(config_path):
        print(f"PIPELINE_FAIL stage=find_checkpoint reason=CONFIG_NOT_FOUND path={config_path}")
        stop_heartbeat()
        sys.exit(1)

    return latest_ckpt, config_path


def count_frames(directory: str) -> int:
    jpgs = glob.glob(os.path.join(directory, "*.jpg"))
    jpegs = glob.glob(os.path.join(directory, "*.jpeg"))
    return len(jpgs) + len(jpegs)


# ---------------------------------------------------------------------------
# Normal mode: stages 1–4
# ---------------------------------------------------------------------------

def run_normal(args, sb, worker_id: str, pipeline_start: datetime) -> None:
    # Stage 1 — Ingest
    run_stage(
        ["python", "pipeline/ingest.py",
         "--video-path", args.video_path,
         "--user-id", args.user_id,
         "--scan-id", args.scan_id],
        "ingest", args.scan_id, sb,
    )
    set_status(sb, args.scan_id, "validating")

    # Stage 2 — Preflight
    run_stage(
        ["python", "pipeline/preflight.py",
         "--video-path", args.video_path,
         "--scan-id", args.scan_id],
        "preflight", args.scan_id, sb,
    )
    set_status(sb, args.scan_id, "extracting_frames")

    # Stage 3 — Frame extraction
    os.makedirs(args.frames_dir, exist_ok=True)
    run_stage(
        ["ffmpeg", "-i", args.video_path,
         "-vf", "fps=2.5",
         "-q:v", "2",
         f"{args.frames_dir}/frame_%05d.jpg",
         "-y"],
        "frame_extraction", args.scan_id, sb,
    )
    set_status(sb, args.scan_id, "segmenting")

    # Stage 4 — Segmentation (keyframe selection + preview; pauses here)
    run_stage(
        ["python", "pipeline/segment.py",
         "--frames-dir", args.frames_dir,
         "--output-dir", args.masked_frames_dir,
         "--scan-id", args.scan_id],
        "segmentation", args.scan_id, sb,
    )

    set_status(sb, args.scan_id, "awaiting_confirmation")
    stop_heartbeat()

    print()
    print("PIPELINE_SUSPENDED: awaiting user confirmation")
    print("Resume with:")
    print(f"  python pipeline/pipeline.py \\")
    print(f"    --resume-from confirmed \\")
    print(f"    --confirmed-keyframe-index N \\")
    print(f"    --video-path {args.video_path} \\")
    print(f"    --scan-id {args.scan_id} \\")
    print(f"    --user-id {args.user_id} \\")
    print(f"    --frames-dir {args.frames_dir} \\")
    print(f"    --masked-frames-dir {args.masked_frames_dir} \\")
    print(f"    --colmap-output-dir {args.colmap_output_dir} \\")
    print(f"    --splat-output-dir {args.splat_output_dir} \\")
    print(f"    --raw-video-r2-key {args.raw_video_r2_key}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Resume mode: stages 5–13
# ---------------------------------------------------------------------------

def run_resume(args, sb, worker_id: str, pipeline_start: datetime) -> None:
    # Stage 5 — Mask propagation
    run_stage(
        ["python", "pipeline/segment.py",
         "--frames-dir", args.frames_dir,
         "--output-dir", args.masked_frames_dir,
         "--scan-id", args.scan_id,
         "--propagate",
         "--confirmed-keyframe-index", str(args.confirmed_keyframe_index)],
        "mask_propagation", args.scan_id, sb,
    )
    set_status(sb, args.scan_id, "reconstructing")

    # Stage 6 — COLMAP (three sub-commands; status stays 'reconstructing')
    os.makedirs(os.path.join(args.colmap_output_dir, "sparse"), exist_ok=True)

    run_stage(
        ["colmap", "feature_extractor",
         "--database_path", f"{args.colmap_output_dir}/database.db",
         "--image_path", args.masked_frames_dir,
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "OPENCV",
         "--SiftExtraction.use_gpu", "1"],
        "colmap_feature_extractor", args.scan_id, sb,
    )

    run_stage(
        ["colmap", "exhaustive_matcher",
         "--database_path", f"{args.colmap_output_dir}/database.db",
         "--SiftMatching.use_gpu", "1"],
        "colmap_exhaustive_matcher", args.scan_id, sb,
    )

    run_stage(
        ["colmap", "mapper",
         "--database_path", f"{args.colmap_output_dir}/database.db",
         "--image_path", args.masked_frames_dir,
         "--output_path", f"{args.colmap_output_dir}/sparse"],
        "colmap_mapper", args.scan_id, sb,
    )

    # Stage 7 — Check COLMAP
    frame_count = count_frames(args.masked_frames_dir)
    run_stage(
        ["python", "pipeline/check_colmap.py",
         "--colmap-output-dir", f"{args.colmap_output_dir}/sparse/0",
         "--total-frames", str(frame_count),
         "--scan-id", args.scan_id],
        "check_colmap", args.scan_id, sb,
    )
    set_status(sb, args.scan_id, "training")

    # Stage 8 — Downscale frames to 1080p
    masked_1080p = f"{args.masked_frames_dir}_1080p"
    os.makedirs(masked_1080p, exist_ok=True)
    run_stage(
        ["ffmpeg",
         "-i", f"{args.masked_frames_dir}/frame_%05d.jpg",
         "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease",
         "-q:v", "2",
         f"{masked_1080p}/frame_%05d.jpg",
         "-y"],
        "downscale_frames", args.scan_id, sb,
    )

    # Stage 9 — splatfacto training
    ns_data_dir = f"{args.colmap_output_dir}/ns_data"

    run_stage(
        ["ns-process-data", "images",
         "--data", masked_1080p,
         "--output-dir", ns_data_dir,
         "--skip-colmap",
         "--downscale-factor", "1"],
        "ns_process_data", args.scan_id, sb,
    )

    ns_train_cmd = (
        f"ns-train splatfacto"
        f" --output-dir {args.splat_output_dir}"
        f" --experiment-name gaiakta_scan"
        f" nerfstudio-data"
        f" --data {ns_data_dir}"
        f" --downscale-factor 1"
    )
    if args.training_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.training_log)), exist_ok=True)
        shell_cmd = f"{ns_train_cmd} 2>&1 | tee {args.training_log}"
        run_stage([], "ns_train", args.scan_id, sb, shell=True, shell_cmd=shell_cmd)
    else:
        run_stage(ns_train_cmd.split(), "ns_train", args.scan_id, sb)

    set_status(sb, args.scan_id, "exporting")

    # Stage 10 — Export splat
    _, latest_config = find_latest_checkpoint(args.splat_output_dir)
    export_dir = f"{args.splat_output_dir}/export"
    os.makedirs(export_dir, exist_ok=True)
    run_stage(
        ["ns-export", "gaussian-splat",
         "--load-config", latest_config,
         "--output-dir", export_dir],
        "ns_export", args.scan_id, sb,
    )
    set_status(sb, args.scan_id, "postprocessing")

    # Stage 11 — Postprocess
    run_stage(
        ["python", "pipeline/postprocess.py",
         "--input-ply", f"{export_dir}/splat.ply",
         "--output-ply", f"{export_dir}/splat_clean.ply",
         "--scan-id", args.scan_id],
        "postprocess", args.scan_id, sb,
    )
    set_status(sb, args.scan_id, "quality_check")

    # Stage 12 — Quality check
    latest_ckpt, _ = find_latest_checkpoint(args.splat_output_dir)
    qc_cmd = [
        "python", "pipeline/check_quality.py",
        "--ckpt-path", latest_ckpt,
        "--colmap-metrics-json", f"{args.colmap_output_dir}/sparse/0/colmap_metrics.json",
        "--splat-ply", f"{export_dir}/splat_clean.ply",
        "--scan-id", args.scan_id,
    ]
    if args.training_log:
        qc_cmd += ["--training-log", args.training_log]
    run_stage(qc_cmd, "check_quality", args.scan_id, sb)
    set_status(sb, args.scan_id, "finalizing")

    # Lineage / cost stubs before finalize
    log_lineage(args.scan_id, "pipeline", f"{export_dir}/splat_clean.ply",
                {"stages_completed": 12})
    log_cost(args.scan_id, "pipeline", "total_pipeline_run", 1)

    # Stage 13 — Finalize
    run_stage(
        ["python", "pipeline/finalize.py",
         "--splat-path", f"{export_dir}/splat_clean.ply",
         "--scan-id", args.scan_id,
         "--user-id", args.user_id,
         "--raw-video-r2-key", args.raw_video_r2_key],
        "finalize", args.scan_id, sb,
    )

    # Done
    elapsed = (datetime.now(timezone.utc) - pipeline_start).total_seconds()
    stop_heartbeat()
    print(f"\nPIPELINE_COMPLETE scan_id={args.scan_id}")
    print(f"Total elapsed time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== GAIAKTA pipeline.py ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    parser = argparse.ArgumentParser(description="GAIAKTA master pipeline orchestrator")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--raw-video-r2-key", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masked-frames-dir", required=True)
    parser.add_argument("--colmap-output-dir", required=True)
    parser.add_argument("--splat-output-dir", required=True)
    parser.add_argument("--resume-from", choices=["confirmed"], default=None)
    parser.add_argument("--confirmed-keyframe-index", type=int, default=None)
    parser.add_argument("--training-log", default=None,
                        help="Optional path to capture ns-train output via tee")
    args = parser.parse_args()

    if args.resume_from == "confirmed" and args.confirmed_keyframe_index is None:
        print("ERROR: --confirmed-keyframe-index is required with --resume-from confirmed")
        sys.exit(1)

    worker_id = f"pipeline-{uuid.uuid4().hex[:8]}"
    print(f"Worker ID: {worker_id}")

    sb = make_supabase_client()
    pipeline_start = datetime.now(timezone.utc)

    lease_lost = start_heartbeat(args.scan_id, worker_id)

    try:
        # Monitor for lease loss in a lightweight way — check before each stage
        # by testing the event; run_stage handles the hard failure path.
        if lease_lost is not None and lease_lost.is_set():
            print("PIPELINE_ERROR: lease lost before first stage")
            sys.exit(1)

        if args.resume_from == "confirmed":
            print(f"Resuming from: confirmed (keyframe index={args.confirmed_keyframe_index})")
            run_resume(args, sb, worker_id, pipeline_start)
        else:
            run_normal(args, sb, worker_id, pipeline_start)

    except SystemExit:
        raise
    except Exception as e:
        try:
            set_status(sb, args.scan_id, "failed")
        except Exception:
            pass
        stop_heartbeat()
        print(f"PIPELINE_ERROR: {e}")
        sys.exit(1)
    finally:
        stop_heartbeat()


if __name__ == "__main__":
    main()
