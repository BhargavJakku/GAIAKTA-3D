"""
GAIAKTA heartbeat.py — lease management for scan_jobs.

Worker mode:  maintains a heartbeat lease on a running job.
Orchestrator mode:  reclaims stale leases and expired confirmations.
"""

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta

from supabase import create_client, Client

from ingest import log_cost, log_lineage  # noqa: F401 — log_lineage imported for pipeline use


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

def _get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: env vars not set")
        sys.exit(1)
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Worker mode — background heartbeat thread
# ---------------------------------------------------------------------------

_stop_event: threading.Event = threading.Event()
_lease_lost_event: threading.Event = threading.Event()
_heartbeat_thread: threading.Thread | None = None
_worker_scan_id: int | None = None
_worker_id: str | None = None


def start_heartbeat(scan_id: int, worker_id: str) -> threading.Event:
    """
    Start the daemon heartbeat thread.

    Returns the lease_lost_event; pipeline.py should monitor this and abort
    if it becomes set.
    """
    global _stop_event, _lease_lost_event, _heartbeat_thread
    global _worker_scan_id, _worker_id

    _worker_scan_id = scan_id
    _worker_id = worker_id
    _stop_event = threading.Event()
    _lease_lost_event = threading.Event()

    client = _get_client()

    # Claim the lease immediately on startup.
    _claim_lease(client, scan_id, worker_id)

    _heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(client, scan_id, worker_id),
        daemon=True,
        name="gaiakta-heartbeat",
    )
    _heartbeat_thread.start()
    return _lease_lost_event


def stop_heartbeat() -> None:
    """Signal the heartbeat thread to stop and release the lease."""
    global _heartbeat_thread

    _stop_event.set()
    if _heartbeat_thread is not None:
        _heartbeat_thread.join(timeout=10)
        _heartbeat_thread = None

    if _worker_scan_id is not None and _worker_id is not None:
        try:
            client = _get_client()
            _release_lease(client, _worker_scan_id, _worker_id)
        except Exception as exc:
            print(f"HEARTBEAT: error releasing lease: {exc}")


def get_lease_expiry():
    return (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

def _claim_lease(client: Client, scan_id: int, worker_id: str) -> None:
    client.table("scan_jobs").update(
        {
            "worker_id": worker_id,
            "lease_expires_at": get_lease_expiry(),
        }
    ).eq("id", scan_id).execute()


def _release_lease(client: Client, scan_id: int, worker_id: str) -> None:
    client.table("scan_jobs").update(
        {"worker_id": None, "lease_expires_at": None}
    ).eq("id", scan_id).eq("worker_id", worker_id).execute()


def _heartbeat_loop(client: Client, scan_id: int, worker_id: str) -> None:
    consecutive_errors = 0

    while not _stop_event.wait(timeout=60):
        try:
            result = (
                client.table("scan_jobs")
                .update({"lease_expires_at": get_lease_expiry()})
                .eq("id", scan_id)
                .eq("worker_id", worker_id)
                .execute()
            )

            rows_affected = len(result.data) if result.data else 0
            if rows_affected == 0:
                print(
                    f"LEASE_LOST: job was reclaimed by orchestrator "
                    f"(scan_id={scan_id})"
                )
                _lease_lost_event.set()
                return

            consecutive_errors = 0
            log_cost(scan_id, "heartbeat", "lease_renewal", 1)

        except Exception as exc:
            consecutive_errors += 1
            print(f"HEARTBEAT: Supabase error ({consecutive_errors}/3): {exc}")
            if consecutive_errors >= 3:
                print("HEARTBEAT_FAILED: 3 consecutive errors")
                _lease_lost_event.set()
                return


# ---------------------------------------------------------------------------
# Orchestrator mode
# ---------------------------------------------------------------------------

def run_orchestrator(client: Client) -> None:
    reclaimed = 0
    dead_lettered = 0
    abandoned = 0

    # --- stale leases ---
    stale_statuses_excluded = (
        "complete",
        "failed",
        "dead_lettered",
        "abandoned_confirmation",
        "awaiting_confirmation",
        "uploaded",
    )

    try:
        stale_result = (
            client.table("scan_jobs")
            .select("id, retry_count, status")
            .lt("lease_expires_at", "now()")
            .not_.in_("status", stale_statuses_excluded)
            .execute()
        )
        stale_jobs = stale_result.data or []
    except Exception as exc:
        print(f"ORCHESTRATOR: error querying stale leases: {exc}")
        stale_jobs = []

    for job in stale_jobs:
        job_id = job["id"]
        retry_count = (job.get("retry_count") or 0) + 1
        try:
            if retry_count >= 3:
                client.table("scan_jobs").update(
                    {"status": "dead_lettered", "retry_count": retry_count}
                ).eq("id", job_id).execute()
                print(f"DEAD_LETTERED: scan_id={job_id} after 3 retries")
                dead_lettered += 1
            else:
                client.table("scan_jobs").update(
                    {
                        "status": "queued",
                        "worker_id": None,
                        "lease_expires_at": None,
                        "retry_count": retry_count,
                    }
                ).eq("id", job_id).execute()
                print(f"RECLAIMED: scan_id={job_id} retry={retry_count}")
                reclaimed += 1
        except Exception as exc:
            print(f"ORCHESTRATOR: error processing stale job {job_id}: {exc}")

    # --- expired confirmations ---
    try:
        expired_result = (
            client.table("scan_jobs")
            .select("id")
            .lt("confirmation_expires_at", "now()")
            .eq("status", "awaiting_confirmation")
            .execute()
        )
        expired_jobs = expired_result.data or []
    except Exception as exc:
        print(f"ORCHESTRATOR: error querying expired confirmations: {exc}")
        expired_jobs = []

    for job in expired_jobs:
        job_id = job["id"]
        try:
            client.table("scan_jobs").update(
                {"status": "abandoned_confirmation"}
            ).eq("id", job_id).execute()
            print(f"ABANDONED: scan_id={job_id} confirmation expired")
            abandoned += 1
        except Exception as exc:
            print(f"ORCHESTRATOR: error abandoning job {job_id}: {exc}")

    print(
        f"ORCHESTRATOR_DONE reclaimed={reclaimed} "
        f"dead_lettered={dead_lettered} abandoned={abandoned}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GAIAKTA heartbeat / orchestrator")
    parser.add_argument(
        "--mode",
        choices=["worker", "orchestrator"],
        required=True,
        help="Run as 'worker' (keep-alive) or 'orchestrator' (reclaim stale jobs)",
    )
    parser.add_argument("--scan-id", type=str, help="Scan job ID (worker mode)")
    parser.add_argument("--worker-id", type=str, help="Worker identifier (worker mode)")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== GAIAKTA heartbeat.py === {ts} | mode={args.mode}")

    client = _get_client()

    if args.mode == "worker":
        if args.scan_id is None or args.worker_id is None:
            print("ERROR: --scan-id and --worker-id are required in worker mode")
            sys.exit(1)

        lease_lost = start_heartbeat(args.scan_id, args.worker_id)
        print(
            f"HEARTBEAT: started for scan_id={args.scan_id} worker_id={args.worker_id}"
        )
        try:
            # When run standalone, block until the lease is lost or interrupted.
            lease_lost.wait()
        except KeyboardInterrupt:
            print("HEARTBEAT: interrupted, releasing lease")
        finally:
            stop_heartbeat()

    elif args.mode == "orchestrator":
        run_orchestrator(client)


if __name__ == "__main__":
    main()
