#!/usr/bin/env python3
# GAIAKTA Pipeline — Script 1 of 9
# Status: COMPLETE — smoke tested, Supabase verified
# Date: March 2026
"""GAIAKTA ingest.py — video ingestion, validation, and Supabase registration."""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def log_lineage(scan_id: str, stage: str, artifact_path: str, metadata_dict: dict) -> bool:
    # TODO: write to scan_artifacts table when lineage explorer is implemented
    return True


def log_cost(scan_id: str, stage: str, metric: str, value: float) -> bool:
    # TODO: write to scan_costs table when billing is implemented
    return True


def compute_sha256(path: Path) -> str:
    CHUNK = 8 * 1024 * 1024  # 8MB
    REPORT_EVERY = 500 * 1024 * 1024  # 500MB

    h = hashlib.sha256()
    processed = 0
    last_report = 0

    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            processed += len(chunk)
            mb_processed = processed // (1024 * 1024)
            if processed - last_report >= REPORT_EVERY:
                print(f"Hashing... {mb_processed}MB processed")
                last_report = processed

    return h.hexdigest()


def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        print(f"ERROR: Supabase connection failed: {e}")
        sys.exit(1)


def today_midnight_utc() -> str:
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today.isoformat()


def check_duplicate(sb, file_hash: str, scan_id: str):
    try:
        result = (
            sb.table("scan_jobs")
            .select("id")
            .eq("file_hash", file_hash)
            .neq("id", scan_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        print(f"ERROR: Supabase connection failed: {e}")
        sys.exit(1)

    if result.data:
        existing_id = result.data[0]["id"]
        print(f"DUPLICATE: scan_id={existing_id}")
        sys.exit(2)


def check_quota(sb, user_id: str):
    midnight = today_midnight_utc()
    try:
        result = (
            sb.table("scan_jobs")
            .select("file_size_bytes")
            .eq("user_id", user_id)
            .gte("created_at", midnight)
            .execute()
        )
    except Exception as e:
        print(f"ERROR: Supabase connection failed: {e}")
        sys.exit(1)

    total = sum(
        row["file_size_bytes"] for row in result.data if row.get("file_size_bytes") is not None
    )
    limit = 10 * 1024 * 1024 * 1024  # 10GB
    if total > limit:
        gb_used = total / (1024 ** 3)
        print(f"QUOTA_EXCEEDED: {gb_used:.1f}GB used today")
        sys.exit(3)


def check_rate_limit(sb, user_id: str):
    midnight = today_midnight_utc()
    try:
        result = (
            sb.table("scan_jobs")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .gte("created_at", midnight)
            .execute()
        )
    except Exception as e:
        print(f"ERROR: Supabase connection failed: {e}")
        sys.exit(1)

    count = result.count if result.count is not None else len(result.data)
    if count >= 10:
        print(f"RATE_LIMITED: {count} jobs submitted today")
        sys.exit(4)


def update_scan_job(sb, scan_id: str, file_hash: str, file_size: int):
    now = datetime.now(timezone.utc).isoformat()
    try:
        sb.table("scan_jobs").update({
            "file_hash": file_hash,
            "file_size_bytes": file_size,
            "status": "validating",
            "updated_at": now,
        }).eq("id", scan_id).execute()
    except Exception as e:
        print(f"ERROR: Supabase connection failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GAIAKTA video ingest")
    parser.add_argument("--video-path", required=True, help="Local path to raw video file")
    parser.add_argument("--user-id", required=True, help="UUID of the user")
    parser.add_argument("--scan-id", required=True, help="UUID of the scan job")
    parser.add_argument("--dry-run", action="store_true", help="Skip all Supabase writes")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== GAIAKTA ingest.py === {ts}")

    # Validate file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: video file not found: {args.video_path}")
        sys.exit(1)

    file_size = video_path.stat().st_size
    if file_size == 0:
        print("ERROR: video file is empty")
        sys.exit(1)

    # Hash
    try:
        file_hash = compute_sha256(video_path)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not args.dry_run:
        sb = get_supabase_client()
        check_duplicate(sb, file_hash, args.scan_id)
        check_quota(sb, args.user_id)
        check_rate_limit(sb, args.user_id)

        update_scan_job(sb, args.scan_id, file_hash, file_size)
    else:
        print(
            f"DRY_RUN: would UPDATE scan_jobs SET "
            f"file_hash={file_hash}, file_size_bytes={file_size}, "
            f"status=validating WHERE id={args.scan_id}"
        )

    size_mb = file_size / (1024 * 1024)
    print(f"INGEST_OK hash={file_hash} size={size_mb:.1f}MB")

    log_lineage(args.scan_id, "ingest", str(video_path), {"hash": file_hash, "size_bytes": file_size})
    log_cost(args.scan_id, "ingest", "file_size_bytes", file_size)

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
