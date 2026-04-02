"""
finalize.py — GAIAKTA Gaussian Splat pipeline finalizer.

Idempotent 7-step finalization: upload splat to R2, verify integrity,
write scan record, validate CDN, mark complete, notify user, clean up raw video.
Safe to re-run after any crash — resumes from last completed step.
"""

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timezone

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

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

def fail(step: int, reason: str, code: int = 1) -> None:
    print(f"FINALIZE_FAIL step={step} reason={reason}")
    sys.exit(code)


def err(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}")
    sys.exit(code)


def sha256_file(path: str) -> str:
    """Compute SHA-256 of a file in 8 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Environment / client setup
# ---------------------------------------------------------------------------

def load_env() -> dict:
    """Load and validate required env vars. Exit 1 if missing."""
    required = ["R2_ENDPOINT", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET",
                "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print("ERROR: required env vars not set")
        sys.exit(1)

    return {
        "r2_endpoint":    os.environ["R2_ENDPOINT"],
        "r2_access_key":  os.environ["R2_ACCESS_KEY"],
        "r2_secret_key":  os.environ["R2_SECRET_KEY"],
        "r2_bucket":      os.environ["R2_BUCKET"],
        "supabase_url":   os.environ["SUPABASE_URL"],
        "supabase_key":   os.environ["SUPABASE_SERVICE_KEY"],
        "r2_public_url":  os.environ.get("R2_PUBLIC_URL", ""),
        "onesignal_app":  os.environ.get("ONESIGNAL_APP_ID", ""),
        "onesignal_key":  os.environ.get("ONESIGNAL_API_KEY", ""),
    }


def make_r2_client(env: dict):
    try:
        return boto3.client(
            "s3",
            endpoint_url=env["r2_endpoint"],
            aws_access_key_id=env["r2_access_key"],
            aws_secret_access_key=env["r2_secret_key"],
        )
    except (BotoCoreError, Exception) as e:
        err(f"R2 connection failed: {e}")


def make_supabase_client(env: dict):
    try:
        from supabase import create_client
    except ImportError:
        err("supabase-py is not installed")
    try:
        return create_client(env["supabase_url"], env["supabase_key"])
    except Exception as e:
        err(f"Supabase failed: {e}")


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def get_finalize_step(sb, scan_id: str) -> int:
    try:
        res = (
            sb.table("scan_jobs")
            .select("finalize_step")
            .eq("id", scan_id)
            .single()
            .execute()
        )
        return int(res.data.get("finalize_step") or 0)
    except Exception as e:
        err(f"Supabase failed: {e}")


def set_finalize_step(sb, scan_id: str, step: int, extra: dict | None = None) -> None:
    payload = {"finalize_step": step, "updated_at": datetime.now(timezone.utc).isoformat()}
    if extra:
        payload.update(extra)
    try:
        sb.table("scan_jobs").update(payload).eq("id", scan_id).execute()
    except Exception as e:
        err(f"Supabase failed: {e}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step1_upload(r2, sb, env: dict, scan_id: str, splat_path: str,
                 current_step: int) -> str:
    """Upload splat.ply to R2. Returns local SHA-256 hash."""
    if current_step >= 1:
        print("FINALIZE_STEP_1_SKIP already done")
        # Re-derive hash so Step 2 can use it even when resuming
        return sha256_file(splat_path)

    print(f"Step 1: uploading {splat_path} to R2 ...")
    local_hash = sha256_file(splat_path)
    r2_key = f"splats/{scan_id}/splat.ply"

    try:
        resp = r2.upload_file(splat_path, env["r2_bucket"], r2_key)
    except (BotoCoreError, ClientError, Exception) as e:
        err(f"R2 connection failed: {e}")

    # Fetch ETag via head_object
    try:
        head = r2.head_object(Bucket=env["r2_bucket"], Key=r2_key)
        etag = head.get("ETag", "").strip('"')
    except Exception as e:
        err(f"R2 connection failed: {e}")

    print(f"FINALIZE_STEP_1_OK uploaded splat.ply etag={etag} hash={local_hash}")
    set_finalize_step(sb, scan_id, 1, {"upload_etag": etag})
    return local_hash


def step2_verify(r2, sb, env: dict, scan_id: str, local_hash: str,
                 current_step: int) -> None:
    """Download splat from R2 and verify SHA-256 matches local hash."""
    if current_step >= 2:
        print("FINALIZE_STEP_2_SKIP already done")
        return

    print("Step 2: verifying upload integrity ...")
    r2_key = f"splats/{scan_id}/splat.ply"

    try:
        obj = r2.get_object(Bucket=env["r2_bucket"], Key=r2_key)
        data = obj["Body"].read()
    except (BotoCoreError, ClientError, Exception) as e:
        err(f"R2 connection failed: {e}")

    remote_hash = sha256_bytes(data)
    if remote_hash != local_hash:
        fail(2, "CHECKSUM_MISMATCH")

    print("FINALIZE_STEP_2_OK checksum verified")
    set_finalize_step(sb, scan_id, 2)


def step3_write_scan(sb, scan_id: str, user_id: str, current_step: int) -> None:
    """Insert scan record into scans table."""
    if current_step >= 3:
        print("FINALIZE_STEP_3_SKIP already done")
        return

    print("Step 3: writing scan record to Supabase ...")
    r2_path = f"splats/{scan_id}/splat.ply"
    try:
        sb.table("scans").upsert(
            {
                "id": scan_id,
                "user_id": user_id,
                "r2_path": r2_path,
                "status": "pending_publish",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="id",
            ignore_duplicates=True,
        ).execute()
    except Exception as e:
        err(f"Supabase failed: {e}")

    print("FINALIZE_STEP_3_OK scan record written")
    set_finalize_step(sb, scan_id, 3)


def step4_validate_cdn(sb, env: dict, scan_id: str, current_step: int) -> None:
    """Validate CDN URL via HTTP HEAD with 3 retries."""
    if current_step >= 4:
        print("FINALIZE_STEP_4_SKIP already done")
        return

    r2_path = f"splats/{scan_id}/splat.ply"

    if not env["r2_public_url"]:
        print("FINALIZE_STEP_4_SKIP no R2_PUBLIC_URL configured")
        set_finalize_step(sb, scan_id, 4)
        return

    url = f"{env['r2_public_url'].rstrip('/')}/{r2_path}"
    print(f"Step 4: validating CDN URL {url} ...")

    for attempt in range(3):
        try:
            resp = requests.head(url, timeout=15)
            if resp.status_code < 400:
                print("FINALIZE_STEP_4_OK cdn url verified")
                set_finalize_step(sb, scan_id, 4)
                return
        except requests.RequestException:
            pass
        if attempt < 2:
            print(f"  CDN check attempt {attempt + 1} failed, retrying in 10s ...")
            time.sleep(10)

    fail(4, "CDN_NOT_REACHABLE")


def step5_mark_complete(sb, scan_id: str, current_step: int) -> None:
    """Mark scan_jobs and scans as complete/published."""
    if current_step >= 5:
        print("FINALIZE_STEP_5_SKIP already done")
        return

    print("Step 5: marking complete ...")
    try:
        sb.table("scan_jobs").update({"status": "complete",
                                      "updated_at": datetime.now(timezone.utc).isoformat()}) \
            .eq("id", scan_id).execute()
        sb.table("scans").update({"status": "published"}) \
            .eq("id", scan_id).execute()
    except Exception as e:
        err(f"Supabase failed: {e}")

    print("FINALIZE_STEP_5_OK marked complete")
    set_finalize_step(sb, scan_id, 5)


def step6_notify(sb, env: dict, scan_id: str, user_id: str, current_step: int) -> None:
    """Send OneSignal push notification."""
    if current_step >= 6:
        print("FINALIZE_STEP_6_SKIP already done")
        return

    if not env["onesignal_app"] or not env["onesignal_key"]:
        print("FINALIZE_STEP_6_SKIP OneSignal not configured")
        set_finalize_step(sb, scan_id, 6)
        return

    print("Step 6: sending OneSignal notification ...")
    try:
        resp = requests.post(
            "https://onesignal.com/api/v1/notifications",
            headers={"Authorization": f"Basic {env['onesignal_key']}",
                     "Content-Type": "application/json"},
            json={
                "app_id": env["onesignal_app"],
                "include_external_user_ids": [user_id],
                "contents": {"en": "Your GAIAKTA scan is ready to view!"},
                "data": {"scan_id": scan_id},
            },
            timeout=15,
        )
        resp.raise_for_status()
        print("FINALIZE_STEP_6_OK notification sent")
    except Exception as e:
        print(f"WARNING: OneSignal notification failed (non-fatal): {e}")

    set_finalize_step(sb, scan_id, 6)


def step7_delete_raw(r2, sb, env: dict, scan_id: str,
                     raw_key: str, current_step: int) -> None:
    """Delete raw video from R2."""
    if current_step >= 7:
        print("FINALIZE_STEP_7_SKIP already done")
        return

    if not raw_key or raw_key.lower() == "none":
        print("FINALIZE_STEP_7_SKIP no raw video key provided")
        set_finalize_step(sb, scan_id, 7)
        return

    print(f"Step 7: deleting raw video {raw_key} from R2 ...")
    try:
        r2.delete_object(Bucket=env["r2_bucket"], Key=raw_key)
    except (BotoCoreError, ClientError, Exception) as e:
        err(f"R2 connection failed: {e}")

    print("FINALIZE_STEP_7_OK raw video deleted")
    set_finalize_step(sb, scan_id, 7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== GAIAKTA finalize.py ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    parser = argparse.ArgumentParser(description="GAIAKTA pipeline finalizer")
    parser.add_argument("--splat-path", required=True, help="Local path to clean splat.ply")
    parser.add_argument("--scan-id", required=True, help="Scan job ID in Supabase")
    parser.add_argument("--user-id", required=True, help="User ID for notification")
    parser.add_argument("--raw-video-r2-key", required=True,
                        help="R2 object key of raw video to delete (or 'none')")
    args = parser.parse_args()

    try:
        # --- Validate splat file exists ---
        if not os.path.exists(args.splat_path):
            print("FINALIZE_FAIL reason=SPLAT_NOT_FOUND")
            sys.exit(1)

        splat_size = os.path.getsize(args.splat_path)

        # --- Load env + clients ---
        env = load_env()
        r2 = make_r2_client(env)
        sb = make_supabase_client(env)

        # --- Resumability ---
        current_step = get_finalize_step(sb, args.scan_id)
        if current_step > 0:
            print(f"Resuming from step {current_step}")

        # --- Run steps ---
        local_hash = step1_upload(r2, sb, env, args.scan_id, args.splat_path, current_step)
        step2_verify(r2, sb, env, args.scan_id, local_hash, current_step)
        step3_write_scan(sb, args.scan_id, args.user_id, current_step)
        step4_validate_cdn(sb, env, args.scan_id, current_step)
        step5_mark_complete(sb, args.scan_id, current_step)
        step6_notify(sb, env, args.scan_id, args.user_id, current_step)
        step7_delete_raw(r2, sb, env, args.scan_id, args.raw_video_r2_key, current_step)

        # --- Lineage / cost stubs ---
        r2_path = f"splats/{args.scan_id}/splat.ply"
        # Retrieve etag stored during step 1 for lineage record
        try:
            head = r2.head_object(Bucket=env["r2_bucket"], Key=r2_path)
            etag = head.get("ETag", "").strip('"')
        except Exception:
            etag = ""
        log_lineage(args.scan_id, "finalize", args.splat_path,
                    {"r2_path": r2_path, "etag": etag})
        log_cost(args.scan_id, "finalize", "splat_size_bytes", splat_size)

        print(f"\nFINALIZE_COMPLETE scan_id={args.scan_id}")

    except SystemExit:
        raise
    except Exception as e:
        err(str(e))


if __name__ == "__main__":
    main()
