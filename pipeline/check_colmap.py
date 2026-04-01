"""
GAIAKTA check_colmap.py — validates COLMAP sparse reconstruction quality.
"""

import argparse
import json
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np

from ingest import log_cost, log_lineage

# ---------------------------------------------------------------------------
# COLMAP camera model → number of intrinsic parameters
# ---------------------------------------------------------------------------
_CAMERA_MODEL_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE  — f, cx, cy
    1: 4,   # PINHOLE         — fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL   — f, cx, cy, k
    3: 5,   # RADIAL          — f, cx, cy, k1, k2
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


# ---------------------------------------------------------------------------
# Binary parsers
# ---------------------------------------------------------------------------

def _read_cameras(path: str) -> dict:
    """Return {camera_id: {'model_id', 'width', 'height', 'params'}}."""
    cameras = {}
    with open(path, "rb") as f:
        (num_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_cameras):
            (camera_id,) = struct.unpack("<I", f.read(4))
            (model_id,)  = struct.unpack("<i", f.read(4))
            (width,)     = struct.unpack("<Q", f.read(8))
            (height,)    = struct.unpack("<Q", f.read(8))
            num_params   = _CAMERA_MODEL_PARAMS.get(model_id, 0)
            params       = list(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_images(path: str) -> dict:
    """Return {image_id: {'qvec', 'tvec', 'camera_id', 'name', 'xys', 'point3D_ids'}}."""
    images = {}
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            (image_id,)   = struct.unpack("<I", f.read(4))
            qvec          = struct.unpack("<4d", f.read(32))   # qw qx qy qz
            tvec          = struct.unpack("<3d", f.read(24))   # tx ty tz
            (camera_id,)  = struct.unpack("<I", f.read(4))

            # Null-terminated name
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_chars.append(ch)
            name = b"".join(name_chars).decode("utf-8", errors="replace")

            (num_points2D,) = struct.unpack("<Q", f.read(8))
            xys         = []
            point3D_ids = []
            for _ in range(num_points2D):
                x, y       = struct.unpack("<2d", f.read(16))
                (p3d_id,)  = struct.unpack("<q", f.read(8))   # int64
                xys.append((x, y))
                point3D_ids.append(p3d_id)

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images


def _read_points3D(path: str) -> dict:
    """Return {point3D_id: {'xyz', 'rgb', 'error', 'track_length'}}."""
    points = {}
    with open(path, "rb") as f:
        (num_points,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_points):
            (point3D_id,)    = struct.unpack("<Q", f.read(8))
            xyz              = struct.unpack("<3d", f.read(24))
            rgb              = struct.unpack("<3B", f.read(3))
            (error,)         = struct.unpack("<d", f.read(8))
            (track_length,)  = struct.unpack("<Q", f.read(8))
            # skip track elements — image_id uint32 + point2D_idx uint32 each
            f.read(8 * track_length)
            points[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track_length": track_length,
            }
    return points


# ---------------------------------------------------------------------------
# Pose spread helpers
# ---------------------------------------------------------------------------

def _pose_spread_area(images: dict) -> float:
    """Convex hull area of camera XZ positions (scipy) or bounding box fallback."""
    positions = np.array([img["tvec"] for img in images.values()])
    xz = positions[:, [0, 2]]  # X and Z columns

    if len(xz) < 3:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xz)
        return float(hull.volume)  # volume == area in 2D
    except ImportError:
        pass
    except Exception:
        pass

    # Bounding box fallback
    x_range = float(xz[:, 0].max() - xz[:, 0].min())
    z_range = float(xz[:, 1].max() - xz[:, 1].min())
    return x_range * z_range


# ---------------------------------------------------------------------------
# Main check logic
# ---------------------------------------------------------------------------

def _colmap_fail(reason: str) -> None:
    print(f"COLMAP_FAIL reason={reason}")
    sys.exit(1)


def run_check(
    colmap_output_dir: str,
    total_frames: int,
    scan_id: str,
    min_registered_ratio: float,
    min_point_count: int,
    min_track_length: float,
    max_reprojection_error: float,
) -> None:

    # --- verify required files exist ---
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    for filename in required:
        fpath = os.path.join(colmap_output_dir, filename)
        if not os.path.exists(fpath):
            print(f"COLMAP_FAIL reason=MISSING_FILES missing={filename}")
            sys.exit(1)

    cameras_path  = os.path.join(colmap_output_dir, "cameras.bin")
    images_path   = os.path.join(colmap_output_dir, "images.bin")
    points3d_path = os.path.join(colmap_output_dir, "points3D.bin")

    # --- parse files ---
    try:
        cameras = _read_cameras(cameras_path)
    except struct.error as e:
        print(f"COLMAP_FAIL reason=CORRUPT_OUTPUT file=cameras.bin")
        sys.exit(1)

    try:
        images = _read_images(images_path)
    except struct.error as e:
        print(f"COLMAP_FAIL reason=CORRUPT_OUTPUT file=images.bin")
        sys.exit(1)

    try:
        points3d = _read_points3D(points3d_path)
    except struct.error as e:
        print(f"COLMAP_FAIL reason=CORRUPT_OUTPUT file=points3D.bin")
        sys.exit(1)

    # --- metric 1: registered ratio ---
    registered_ratio = len(images) / total_frames if total_frames > 0 else 0.0
    if registered_ratio < min_registered_ratio:
        print(
            f"COLMAP_FAIL reason=LOW_REGISTERED_RATIO "
            f"value={registered_ratio:.3f} threshold={min_registered_ratio:.3f}"
        )
        sys.exit(1)

    # --- metric 2: point count ---
    point_count = len(points3d)
    if point_count < min_point_count:
        print(
            f"COLMAP_FAIL reason=LOW_POINT_COUNT "
            f"value={point_count} threshold={min_point_count}"
        )
        sys.exit(1)

    # --- metric 3: mean track length ---
    if point_count > 0:
        track_lengths = np.array([p["track_length"] for p in points3d.values()], dtype=np.float64)
        mean_track_length = float(np.mean(track_lengths))
    else:
        mean_track_length = 0.0

    if mean_track_length < min_track_length:
        print(
            f"COLMAP_FAIL reason=LOW_TRACK_LENGTH "
            f"value={mean_track_length:.2f} threshold={min_track_length:.2f}"
        )
        sys.exit(1)

    # --- metric 4: mean reprojection error ---
    if point_count > 0:
        errors = np.array([p["error"] for p in points3d.values()], dtype=np.float64)
        mean_reprojection_error = float(np.mean(errors))
    else:
        mean_reprojection_error = 0.0

    if mean_reprojection_error > max_reprojection_error:
        print(
            f"COLMAP_FAIL reason=HIGH_REPROJECTION_ERROR "
            f"value={mean_reprojection_error:.3f}px threshold={max_reprojection_error:.3f}px"
        )
        sys.exit(1)

    # --- metric 5: pose spread (warn only) ---
    pose_spread_area = _pose_spread_area(images)
    if pose_spread_area < 1.0:
        print(f"COLMAP_WARN reason=POOR_POSE_SPREAD area={pose_spread_area:.3f}")

    # --- all checks passed ---
    metrics = {
        "registered_images": len(images),
        "total_frames": total_frames,
        "registered_ratio": round(registered_ratio, 6),
        "point_count": point_count,
        "mean_track_length": round(mean_track_length, 4),
        "mean_reprojection_error_px": round(mean_reprojection_error, 6),
        "pose_spread_area": round(pose_spread_area, 4),
        "camera_count": len(cameras),
    }

    print(json.dumps(metrics, indent=2))
    print(
        f"COLMAP_OK registered={len(images)} points={point_count} "
        f"track_length={mean_track_length:.2f} reprojection={mean_reprojection_error:.3f}px"
    )

    metrics_path = os.path.join(colmap_output_dir, "colmap_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log_lineage(scan_id, "colmap_check", colmap_output_dir, metrics)
    log_cost(scan_id, "colmap_check", "points_reconstructed", point_count)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== GAIAKTA check_colmap.py === {ts}")

    parser = argparse.ArgumentParser(description="GAIAKTA COLMAP quality check")
    parser.add_argument("--colmap-output-dir", required=True,
                        help="Path to COLMAP sparse/0 directory")
    parser.add_argument("--total-frames", type=int, required=True,
                        help="Total number of input frames submitted to COLMAP")
    parser.add_argument("--scan-id", required=True, help="Scan job ID")
    parser.add_argument("--min-registered-ratio", type=float, default=0.60,
                        help="Minimum fraction of frames that must be registered (default: 0.60)")
    parser.add_argument("--min-point-count", type=int, default=10000,
                        help="Minimum number of 3D points (default: 10000)")
    parser.add_argument("--min-track-length", type=float, default=3.0,
                        help="Minimum mean track length (default: 3.0)")
    parser.add_argument("--max-reprojection-error", type=float, default=1.5,
                        help="Maximum mean reprojection error in pixels (default: 1.5)")
    args = parser.parse_args()

    try:
        run_check(
            colmap_output_dir=args.colmap_output_dir,
            total_frames=args.total_frames,
            scan_id=args.scan_id,
            min_registered_ratio=args.min_registered_ratio,
            min_point_count=args.min_point_count,
            min_track_length=args.min_track_length,
            max_reprojection_error=args.max_reprojection_error,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
