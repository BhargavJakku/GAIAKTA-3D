"""
segment.py — GAIAKTA Gaussian Splat pipeline segmenter.

Normal mode: selects keyframes, runs SAM2, scores mask confidence,
generates preview images, and pauses for user confirmation.

Propagation mode: applies the confirmed keyframe mask to all frames.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
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
# Helpers
# ---------------------------------------------------------------------------

def fail(reason: str, code: int = 1) -> None:
    print(f"SEGMENT_FAIL reason={reason}")
    sys.exit(code)


def err(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}")
    sys.exit(code)


def supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: env vars not set")
        sys.exit(1)
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        err("supabase-py is not installed")
    except Exception as e:
        err(f"Supabase failed: {e}")


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def load_frames(frames_dir: Path) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.jpeg"))
    # Deduplicate while preserving sort order (glob may overlap on case-insensitive FS)
    seen = set()
    unique = []
    for f in sorted(frames, key=lambda p: p.name):
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def select_keyframe_indices(total: int, num_keyframes: int) -> list[int]:
    """Pick num_keyframes evenly spaced indices, always including first and last."""
    if num_keyframes >= total:
        return list(range(total))
    if num_keyframes <= 2:
        return [0, total - 1]
    step = (total - 1) / (num_keyframes - 1)
    indices = sorted(set([0] + [round(i * step) for i in range(1, num_keyframes - 1)] + [total - 1]))
    return indices


# ---------------------------------------------------------------------------
# SAM2 helpers
# ---------------------------------------------------------------------------

def load_sam2(checkpoint: str, config_stem: str, device: str):
    try:
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        fail("SAM2_LOAD_FAILED: sam2 package not installed")

    try:
        # Change working directory to sam2 package root before loading.
        # build_sam2() resolves configs relative to cwd — this is the
        # correct portable approach; finally block guarantees restoration.
        sam2_root = os.path.dirname(os.path.dirname(sam2.__file__))
        original_dir = os.getcwd()
        try:
            os.chdir(sam2_root)
            model = build_sam2(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                checkpoint,
                device=device,
            )
        finally:
            os.chdir(original_dir)
        predictor = SAM2ImagePredictor(model)
        return predictor
    except Exception as e:
        fail(f"SAM2_LOAD_FAILED: {e}")



def make_point_grid(h: int, w: int, grid_size: int = 5) -> "np.ndarray":
    """Return a (grid_size*grid_size, 2) array of XY points in the middle 60% of the frame."""
    import numpy as np
    margin_y = int(h * 0.2)
    margin_x = int(w * 0.2)
    ys = np.linspace(margin_y, h - margin_y, grid_size)
    xs = np.linspace(margin_x, w - margin_x, grid_size)
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(float)
    return points


def best_mask(predictor, image_rgb: "np.ndarray") -> tuple["np.ndarray", float]:
    """
    Run SAM2 with a 5×5 point grid as positive prompts.
    Return the mask with the highest centrality score
    (iou_score * (1 - normalised distance from centre)).
    """
    import numpy as np

    h, w = image_rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_dist = ((cx ** 2) + (cy ** 2)) ** 0.5

    points = make_point_grid(h, w)
    labels = np.ones(len(points), dtype=int)

    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    best = None
    best_score = -1.0
    for mask, iou in zip(masks, scores):
        # Centroid of mask
        ys_m, xs_m = np.where(mask)
        if len(xs_m) == 0:
            continue
        centroid_x = xs_m.mean()
        centroid_y = ys_m.mean()
        dist = ((centroid_x - cx) ** 2 + (centroid_y - cy) ** 2) ** 0.5
        centrality = float(iou) * (1.0 - dist / max_dist)
        if centrality > best_score:
            best_score = centrality
            best = mask

    if best is None:
        # Fallback: return a full-frame mask with score 0
        import numpy as np
        best = np.ones((h, w), dtype=bool)
        best_score = 0.0

    # Always return a boolean mask regardless of SAM2's output dtype
    return best.astype(bool), best_score


# ---------------------------------------------------------------------------
# Preview generation
# ---------------------------------------------------------------------------

def save_preview(original_path: Path, mask: "np.ndarray", out_path: Path) -> None:
    """Save a side-by-side image: original | masked."""
    import numpy as np
    from PIL import Image

    img = np.array(Image.open(original_path).convert("RGB"))
    mask_bool = mask.astype(bool)
    masked = img.copy()
    masked[~mask_bool] = 0

    side_by_side = np.concatenate([img, masked], axis=1)
    Image.fromarray(side_by_side).save(str(out_path), quality=85)


# ---------------------------------------------------------------------------
# Normal mode
# ---------------------------------------------------------------------------

def run_normal(args, sb) -> None:
    import numpy as np
    from PIL import Image

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    # --- Step 1: Load frames ---
    if not frames_dir.exists():
        fail("FRAMES_DIR_NOT_FOUND")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        fail(f"OUTPUT_DIR_ERROR: {e}")

    frames = load_frames(frames_dir)
    n_frames = len(frames)
    if n_frames < 3:
        print(f"SEGMENT_FAIL reason=TOO_FEW_FRAMES count={n_frames}")
        sys.exit(1)
    print(f"SEGMENT: found {n_frames} frames")

    # --- Step 2: Select keyframes ---
    kf_indices = select_keyframe_indices(n_frames, args.num_keyframes)
    keyframes = [frames[i] for i in kf_indices]
    print(f"SEGMENT: selected {len(keyframes)} keyframes")

    # --- Step 3: Load SAM2 ---
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    predictor = load_sam2(args.sam2_checkpoint, args.sam2_config, device)
    print(f"SEGMENT: SAM2 loaded on {device}")

    # --- Step 4: Segment each keyframe ---
    kf_masks = []
    kf_scores = []
    for i, kf_path in enumerate(keyframes):
        img_rgb = np.array(Image.open(kf_path).convert("RGB"))
        mask, score = best_mask(predictor, img_rgb)
        kf_masks.append(mask)
        kf_scores.append(score)
        print(f"SEGMENT: keyframe {i} confidence={score:.3f}")

    # --- Step 5: Compute overall confidence ---
    mean_conf = float(np.mean(kf_scores))
    print(f"SEGMENT: mean confidence={mean_conf:.3f}")
    try:
        sb.table("scan_jobs").update(
            {"mask_confidence": mean_conf,
             "updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", args.scan_id).execute()
    except Exception as e:
        err(f"Supabase failed: {e}")

    # --- Step 6: Generate preview images ---
    preview_indices = [0, len(keyframes) // 2, len(keyframes) - 1]
    for out_num, pi in enumerate(preview_indices, start=1):
        out_path = output_dir / f"preview_{out_num}.jpg"
        save_preview(keyframes[pi], kf_masks[pi], out_path)
    print(f"SEGMENT: previews saved to {output_dir}")

    # --- Step 7: Set confirmation TTL and pause ---
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    try:
        sb.table("scan_jobs").update(
            {"status": "awaiting_confirmation",
             "confirmation_expires_at": expires,
             "updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", args.scan_id).execute()
    except Exception as e:
        err(f"Supabase failed: {e}")

    print("\nSEGMENT: keyframe index reference:")
    for i, (idx, kf_path) in enumerate(zip(kf_indices, keyframes)):
        print(f"  [{i}] frame_index={idx}  file={kf_path.name}")

    log_cost(args.scan_id, "segment", "keyframes_processed", len(keyframes))

    print(f"\nSEGMENT_OK confidence={mean_conf:.3f} awaiting_user_confirmation")
    print("SEGMENT: pass --confirmed-keyframe-index N to resume")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Propagation mode
# ---------------------------------------------------------------------------

def run_propagate(args, sb) -> None:
    import numpy as np
    from PIL import Image

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    if not frames_dir.exists():
        fail("FRAMES_DIR_NOT_FOUND")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        fail(f"OUTPUT_DIR_ERROR: {e}")

    frames = load_frames(frames_dir)
    n_frames = len(frames)
    if n_frames < 3:
        print(f"SEGMENT_FAIL reason=TOO_FEW_FRAMES count={n_frames}")
        sys.exit(1)

    kf_index = args.confirmed_keyframe_index
    kf_indices = select_keyframe_indices(n_frames, args.num_keyframes)

    if kf_index < 0 or kf_index >= len(kf_indices):
        fail(f"INVALID_KEYFRAME_INDEX: {kf_index} (valid 0–{len(kf_indices) - 1})")

    confirmed_frame_path = frames[kf_indices[kf_index]]
    print(f"PROPAGATE: using keyframe {kf_index} as reference ({confirmed_frame_path.name})")

    # --- Step 1: Load SAM2 and get confirmed keyframe mask ---
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    predictor = load_sam2(args.sam2_checkpoint, args.sam2_config, device)
    img_rgb = np.array(Image.open(confirmed_frame_path).convert("RGB"))
    ref_mask, score = best_mask(predictor, img_rgb)
    print(f"PROPAGATE: reference mask confidence={score:.3f}")

    # --- Step 2: Apply mask to ALL frames ---
    # Note: single confirmed keyframe mask applied uniformly (MVP).
    # SAM2 video propagation for per-frame temporal consistency is V2.
    masked_count = 0
    for i, frame_path in enumerate(frames):
        img = np.array(Image.open(frame_path).convert("RGB"))

        # Resize mask if frame dimensions differ from reference
        if img.shape[:2] != ref_mask.shape[:2]:
            from PIL import Image as PILImage
            h, w = img.shape[:2]
            mask_img = PILImage.fromarray(ref_mask.astype("uint8") * 255)
            mask_img = mask_img.resize((w, h), PILImage.NEAREST)
            frame_mask = np.array(mask_img).astype(bool)
        else:
            frame_mask = ref_mask.astype(bool)

        img[~frame_mask] = 0
        Image.fromarray(img).save(str(output_dir / frame_path.name), quality=92)
        masked_count += 1

        if masked_count % 50 == 0:
            print(f"PROPAGATE: {masked_count}/{n_frames} frames")

    print(f"PROPAGATE_OK frames_masked={masked_count} saved to {output_dir}")

    try:
        sb.table("scan_jobs").update(
            {"status": "confirmed",
             "updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", args.scan_id).execute()
    except Exception as e:
        err(f"Supabase failed: {e}")

    log_lineage(args.scan_id, "segment", str(output_dir),
                {"frames_masked": masked_count, "keyframe_used": kf_index})

    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== GAIAKTA segment.py ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    parser = argparse.ArgumentParser(description="GAIAKTA SAM2 segmenter")
    parser.add_argument("--frames-dir", required=True,
                        help="Folder of extracted JPEGs")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write masked frames / previews")
    parser.add_argument("--scan-id", required=True,
                        help="Scan job ID in Supabase")
    parser.add_argument("--num-keyframes", type=int, default=7,
                        help="Number of keyframes to sample (default 7)")
    parser.add_argument("--confidence-threshold", type=float, default=0.75,
                        help="Minimum acceptable mask confidence (default 0.75)")
    parser.add_argument("--propagate", action="store_true",
                        help="Run mask propagation mode")
    parser.add_argument("--confirmed-keyframe-index", type=int, default=None,
                        help="Keyframe index confirmed by user (required with --propagate)")
    parser.add_argument("--sam2-checkpoint",
                        default="/mnt/b/projects/GAIAKTA/models/sam2/sam2.1_hiera_large.pt",
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", default="sam2.1_hiera_l",
                        help="SAM2 config stem (resolved from sam2 package install location)")
    args = parser.parse_args()

    if args.propagate and args.confirmed_keyframe_index is None:
        fail("MISSING_KEYFRAME_INDEX")

    try:
        sb = supabase_client()

        if args.propagate:
            run_propagate(args, sb)
        else:
            run_normal(args, sb)

    except SystemExit:
        raise
    except Exception as e:
        err(str(e))


if __name__ == "__main__":
    main()
