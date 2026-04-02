# PIPELINE NOTE: to capture train_loss in future runs:
# ns-train splatfacto ... 2>&1 | tee outputs/training.log
# Then pass: --training-log outputs/training.log

"""
check_quality.py — GAIAKTA Gaussian Splat pipeline quality checker.

Parses COLMAP metrics, optional training log, checkpoint, and splat PLY header,
computes a composite quality score, updates Supabase, and prints a summary.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

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
# Helpers
# ---------------------------------------------------------------------------

def fail(reason: str, code: int = 1) -> None:
    print(f"QUALITY_FAIL reason={reason}")
    sys.exit(code)


def err(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}")
    sys.exit(code)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_train_loss(log_path: str) -> float | None:
    """Parse last 'Train Loss:' value from log. Returns None with warning if not found."""
    if not os.path.exists(log_path):
        print(f"WARNING: Training log not found: {log_path}")
        return None

    pattern = re.compile(r"Train Loss:\s*([\d.eE+\-]+)")
    last_loss = None

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                last_loss = float(m.group(1))

    if last_loss is None:
        print("WARNING: No 'Train Loss:' lines found in training log — train_loss set to None")

    return last_loss


def parse_checkpoint(ckpt_path: str) -> tuple[int, bool]:
    """Return (training_step, training_completed)."""
    try:
        import torch
    except ImportError:
        err("torch is not installed — required to parse checkpoint")

    if not os.path.exists(ckpt_path):
        err(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, weights_only=False)
    except Exception as e:
        err(f"Failed to load checkpoint: {e}")

    training_step = ckpt["step"]
    training_completed = training_step >= 29000
    return training_step, training_completed


def parse_reprojection_error(metrics_path: str) -> float:
    """Return mean_reprojection_error_px from colmap_metrics.json."""
    if not os.path.exists(metrics_path):
        fail("MISSING_COLMAP_METRICS")

    try:
        with open(metrics_path, "r") as fh:
            data = json.load(fh)
    except json.JSONDecodeError:
        fail("INVALID_METRICS_JSON")

    if "mean_reprojection_error_px" not in data:
        fail("MISSING_COLMAP_METRICS")

    return float(data["mean_reprojection_error_px"])


def parse_gaussian_count(ply_path: str) -> int | None:
    """Read PLY header and return vertex count, or None if not found."""
    if not os.path.exists(ply_path):
        print(f"WARNING: Splat PLY not found: {ply_path} — gaussian_count set to None")
        return None

    try:
        with open(ply_path, "rb") as fh:
            for raw_line in fh:
                if raw_line.strip() == b"end_header":
                    break
                if raw_line.startswith(b"element vertex"):
                    parts = raw_line.split()
                    return int(parts[2])
    except Exception as e:
        print(f"WARNING: Could not parse PLY header: {e} — gaussian_count set to None")
        return None

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(
    reprojection_error: float,
    train_loss: float | None,
    gaussian_count: int | None,
) -> tuple[str, str | None]:
    """Return (score, tip_or_None)."""
    loss_good = train_loss < 0.03 if train_loss is not None else True
    loss_fair = train_loss < 0.06 if train_loss is not None else True
    reproj_good = reprojection_error < 0.6
    reproj_fair = reprojection_error < 1.0
    count_good = gaussian_count > 100000 if gaussian_count is not None else True
    count_fair = gaussian_count > 50000 if gaussian_count is not None else True

    if loss_good and reproj_good and count_good:
        return "Good", None

    if loss_fair and reproj_fair and count_fair:
        reproj_high = not reproj_good
        loss_high = not loss_good

        if reproj_high and loss_high:
            tip = "Rescan with slower movement, more angles, and better lighting"
        elif reproj_high:
            tip = "Try scanning with more overlap — walk more slowly around the subject"
        elif not count_good:
            tip = "Scan may have insufficient coverage — try scanning from more angles"
        else:
            tip = "Quality is marginal — consider rescanning for best results"
        return "Fair", tip

    # Poor
    reproj_high = not reproj_fair
    loss_high = not loss_fair
    count_low = not count_fair

    if reproj_high and loss_high:
        tip = "Rescan with slower movement, more angles, and better lighting"
    elif reproj_high:
        tip = "Try scanning with more overlap — walk more slowly around the subject"
    elif count_low:
        tip = "Scan may have insufficient coverage — try scanning from more angles"
    else:
        tip = "Quality is marginal — consider rescanning for best results"

    return "Poor", tip


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

def update_supabase(scan_id: str, quality_detail: dict) -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        err("SUPABASE_URL and SUPABASE_SERVICE_KEY env vars must be set")

    try:
        from supabase import create_client
    except ImportError:
        err("supabase-py is not installed")

    client = create_client(url, key)

    try:
        (
            client.table("scan_jobs")
            .update(
                {
                    "quality_score": quality_detail["score"],
                    "quality_detail": quality_detail,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            .eq("id", scan_id)
            .execute()
        )
    except Exception as e:
        err(f"Supabase update failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== GAIAKTA check_quality.py ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    parser = argparse.ArgumentParser(description="GAIAKTA quality checker")
    parser.add_argument("--ckpt-path", required=True, help="Path to nerfstudio checkpoint")
    parser.add_argument("--colmap-metrics-json", required=True, help="Path to colmap_metrics.json")
    parser.add_argument("--splat-ply", required=True, help="Path to output splat .ply file")
    parser.add_argument("--scan-id", required=True, help="Scan job ID in Supabase")
    parser.add_argument("--training-log", default=None, help="(Optional) Path to training log file")
    args = parser.parse_args()

    try:
        # --- Training log (optional) ---
        if args.training_log is not None:
            print(f"\nParsing training log: {args.training_log}")
            train_loss = parse_train_loss(args.training_log)
            if train_loss is not None:
                print(f"  Final train loss: {train_loss:.6f}")
        else:
            train_loss = None

        # --- Checkpoint ---
        print(f"\nParsing checkpoint: {args.ckpt_path}")
        training_step, training_completed = parse_checkpoint(args.ckpt_path)
        print(f"  Training step: {training_step}  completed={training_completed}")

        # --- COLMAP metrics ---
        print(f"\nParsing COLMAP metrics: {args.colmap_metrics_json}")
        reprojection_error = parse_reprojection_error(args.colmap_metrics_json)
        print(f"  Mean reprojection error: {reprojection_error:.3f}px")

        # --- Splat PLY ---
        print(f"\nParsing splat PLY header: {args.splat_ply}")
        gaussian_count = parse_gaussian_count(args.splat_ply)
        if gaussian_count is not None:
            print(f"  Gaussian count: {gaussian_count}")

        # --- Score ---
        score, tip = compute_score(reprojection_error, train_loss, gaussian_count)

        quality_detail = {
            "train_loss": train_loss,
            "reprojection_error_px": reprojection_error,
            "gaussian_count": gaussian_count,
            "training_completed": training_completed,
            "training_step": training_step,
            "score": score,
            "tip": tip,
        }

        # --- Supabase ---
        print(f"\nUpdating Supabase for scan_id={args.scan_id} ...")
        update_supabase(args.scan_id, quality_detail)
        print("  Supabase updated.")

        # --- Lineage / cost stubs ---
        log_lineage(args.scan_id, "quality_check", args.colmap_metrics_json, quality_detail)
        log_cost(args.scan_id, "quality_check", "quality_computed", 1)

        # --- Output ---
        loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        print(
            f"\nQUALITY_OK score={score} "
            f"reprojection={reprojection_error:.3f}px "
            f"gaussians={gaussian_count} "
            f"loss={loss_str} "
            f"completed={training_completed}"
        )
        if tip:
            print(f"TIP: {tip}")

    except SystemExit:
        raise
    except Exception as e:
        err(str(e))


if __name__ == "__main__":
    main()
