"""
GAIAKTA postprocess.py — filters low-opacity Gaussians from a 3DGS .ply file.
"""

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
from plyfile import PlyData, PlyElement

from ingest import log_cost, log_lineage


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_postprocess(
    input_ply: str,
    output_ply: str,
    scan_id: str,
    opacity_threshold: float,
) -> None:

    # --- load ---
    try:
        plydata = PlyData.read(input_ply)
    except FileNotFoundError:
        print("POSTPROCESS_FAIL reason=FILE_NOT_FOUND")
        sys.exit(1)

    vertex = plydata["vertex"]
    prop_names = [p.name for p in vertex.properties]

    if "opacity" not in prop_names:
        print("POSTPROCESS_FAIL reason=NO_OPACITY_PROPERTY")
        sys.exit(1)

    raw_opacity = np.array(vertex["opacity"], dtype=np.float32)
    input_count = len(raw_opacity)
    print(f"POSTPROCESS: loaded {input_count} Gaussians from {input_ply}")

    # Detect logits: if any value is outside [0, 1], treat all as logits
    if raw_opacity.min() < 0.0 or raw_opacity.max() > 1.0:
        opacity = sigmoid(raw_opacity.astype(np.float64)).astype(np.float32)
    else:
        opacity = raw_opacity

    # --- filter ---
    keep_mask = opacity >= opacity_threshold
    removed = int((~keep_mask).sum())
    output_count = int(keep_mask.sum())
    print(
        f"POSTPROCESS: removed {removed} Gaussians below opacity threshold "
        f"{opacity_threshold}"
    )

    # --- sanity checks ---
    removed_pct = removed / input_count * 100.0 if input_count > 0 else 0.0

    if output_count < input_count * 0.3:
        print(
            f"POSTPROCESS_WARN: removed more than 70% of Gaussians "
            f"({removed_pct:.1f}%) — verify opacity threshold is appropriate"
        )

    if output_count < 10000:
        print(
            f"POSTPROCESS_FAIL: only {output_count} Gaussians remaining — "
            f"scan may be unusable"
        )
        sys.exit(1)

    # --- write output preserving exact property structure ---
    # Build a filtered structured array with the same dtype as the original
    original_data = vertex.data
    filtered_data = original_data[keep_mask]

    output_element = PlyElement.describe(filtered_data, "vertex")
    # Preserve any non-vertex elements (e.g. face, edge) unchanged
    output_elements = [output_element] + [
        el for el in plydata.elements if el.name != "vertex"
    ]
    PlyData(output_elements, text=plydata.text, byte_order=plydata.byte_order).write(
        output_ply
    )

    print(
        f"POSTPROCESS_OK input={input_count} output={output_count} "
        f"removed={removed} removed_pct={removed_pct:.1f}%"
    )

    log_lineage(
        scan_id,
        "postprocess",
        output_ply,
        {"input_gaussians": input_count, "output_gaussians": output_count},
    )
    log_cost(scan_id, "postprocess", "gaussians_processed", input_count)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== GAIAKTA postprocess.py === {ts}")

    parser = argparse.ArgumentParser(
        description="GAIAKTA Gaussian Splat postprocessor — opacity filtering"
    )
    parser.add_argument("--input-ply", required=True, help="Input .ply file path")
    parser.add_argument("--output-ply", required=True, help="Output .ply file path")
    parser.add_argument("--scan-id", required=True, help="Scan job ID")
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.1,
        help="Remove Gaussians with sigmoid(opacity) below this value (default: 0.1)",
    )
    args = parser.parse_args()

    try:
        run_postprocess(
            input_ply=args.input_ply,
            output_ply=args.output_ply,
            scan_id=args.scan_id,
            opacity_threshold=args.opacity_threshold,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
