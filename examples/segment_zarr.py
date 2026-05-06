#!/usr/bin/env python3
"""
3D segmentation of OME-Zarr volumes using Cellpose models.

Reads Zarr lazily, supports ROI extraction, and runs one or two models
(e.g., cell + blood) with configurable mask IDs.

Usage:
    # Single model
    python examples/segment_zarr.py data/processed/volume.zarr \
        --model data/models/best_model_cell \
        --outdir data/processed/segmentation

    # Dual model (cell + blood with different mask IDs)
    python examples/segment_zarr.py data/processed/volume.zarr \
        --model data/models/best_model_cell \
        --model2 data/models/best_model_blood --mask-id2 2 \
        --outdir data/processed/segmentation

    # With ROI (z_start:z_end,y_start:y_end,x_start:x_end)
    python examples/segment_zarr.py data/processed/volume.zarr \
        --model data/models/best_model_cell \
        --roi 0:100,256:768,256:768 \
        --outdir data/processed/segmentation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import zarr

try:
    from cellpose import models, core
    HAS_CELLPOSE = True
except ImportError:
    HAS_CELLPOSE = False


def parse_roi(roi_str: str, shape: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
    """
    Parse ROI string into slices.

    Format: "z_start:z_end,y_start:y_end,x_start:x_end"
    Use empty values for defaults: ":100,:" means z=0:100, y=all, x=all

    Args:
        roi_str: ROI specification string
        shape: Volume shape (Z, Y, X) for validation

    Returns:
        Tuple of three slices (z_slice, y_slice, x_slice)
    """
    parts = roi_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"ROI must have 3 parts (z,y,x), got {len(parts)}: {roi_str}")

    slices = []
    dim_names = ["Z", "Y", "X"]

    for i, (part, dim_size, name) in enumerate(zip(parts, shape, dim_names)):
        part = part.strip()
        if not part or part == ":":
            slices.append(slice(None))
            continue

        if ":" not in part:
            # Single index - treat as start:start+1
            idx = int(part)
            if idx < 0:
                idx = dim_size + idx
            slices.append(slice(idx, idx + 1))
            continue

        bounds = part.split(":")
        start = int(bounds[0]) if bounds[0] else None
        end = int(bounds[1]) if bounds[1] else None

        # Validate bounds
        if start is not None and start < 0:
            start = dim_size + start
        if end is not None and end < 0:
            end = dim_size + end
        if start is not None and end is not None and start >= end:
            raise ValueError(f"{name} slice start ({start}) must be < end ({end})")
        if end is not None and end > dim_size:
            raise ValueError(f"{name} slice end ({end}) exceeds dimension size ({dim_size})")

        slices.append(slice(start, end))

    return tuple(slices)


def load_zarr_roi(zarr_path: Path, roi: Tuple[slice, slice, slice] | None = None,
                  level: int = 0) -> np.ndarray:
    """
    Lazily load a region from an OME-Zarr volume.

    Args:
        zarr_path: Path to OME-Zarr directory
        roi: Optional tuple of slices (z, y, x). If None, loads entire volume.
        level: Pyramid level to load (0 = full resolution)

    Returns:
        3D numpy array (Z, Y, X)
    """
    root = zarr.open_group(zarr_path, mode="r")

    # Get dataset path from multiscales metadata
    multiscales = root.attrs.get("multiscales", [])
    if not multiscales:
        raise ValueError(f"No multiscales metadata in {zarr_path}")

    datasets = multiscales[0].get("datasets", [])
    if level >= len(datasets):
        raise ValueError(f"Level {level} not available (max: {len(datasets) - 1})")

    arr_path = datasets[level]["path"]
    arr = root[arr_path]

    print(f"Zarr array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Zarr chunks: {arr.chunks}")

    if roi is None:
        print(f"Loading full volume...")
        data = arr[:]
    else:
        z_sl, y_sl, x_sl = roi
        print(f"Loading ROI: z={_slice_str(z_sl, arr.shape[0])}, "
              f"y={_slice_str(y_sl, arr.shape[1])}, x={_slice_str(x_sl, arr.shape[2])}")
        data = arr[z_sl, y_sl, x_sl]

    print(f"Loaded shape: {data.shape}, {data.nbytes / 1e9:.2f} GB")
    return data


def _slice_str(s: slice, dim_size: int) -> str:
    """Format slice for display."""
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else dim_size
    return f"{start}:{stop}"


def run_cellpose_3d(
    volume: np.ndarray,
    model_path: Path,
    anisotropy: float = 1.0,
    cellprob_threshold: float = -1.0,
    min_size: int = 0,
    flow3D_smooth: float = 0.0,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Run Cellpose 3D inference on a volume.

    Args:
        volume: 3D array (Z, Y, X)
        model_path: Path to trained Cellpose model
        anisotropy: Z spacing relative to XY (>1 means Z is coarser)
        cellprob_threshold: Cell probability threshold (lower = more permissive)
        min_size: Minimum object size in voxels
        flow3D_smooth: Smoothing for 3D flow field

    Returns:
        3D label array with instance segmentation
    """
    use_gpu = core.use_gpu()
    print(f"GPU available: {use_gpu}")

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

    print(f"Running 3D inference with model: {model_path.name}")
    print(f"  anisotropy={anisotropy}, cellprob_threshold={cellprob_threshold}, batch_size={batch_size}")

    masks, _flows, _styles = model.eval(
        x=volume,
        do_3D=True,
        z_axis=0,
        channel_axis=None,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        flow3D_smooth=flow3D_smooth,
        batch_size=batch_size,
    )

    masks = masks.astype(np.int32)
    n_objects = masks.max()
    print(f"  Found {n_objects} objects")

    return masks


def combine_masks(mask1: np.ndarray, mask2: np.ndarray,
                  mask_id1: int = 1, mask_id2: int = 2) -> np.ndarray:
    """
    Combine two segmentation masks with different class IDs.

    Each mask's instance labels are encoded as: class_id * 10000 + instance_id
    This allows distinguishing both class and instance from the combined mask.

    Args:
        mask1: First segmentation mask (e.g., cells)
        mask2: Second segmentation mask (e.g., blood vessels)
        mask_id1: Class ID for first mask
        mask_id2: Class ID for second mask

    Returns:
        Combined mask with encoded labels
    """
    combined = np.zeros_like(mask1, dtype=np.int32)

    # Encode mask1: class_id * 10000 + instance_id
    if mask1.max() > 0:
        m1 = mask1 > 0
        combined[m1] = mask_id1 * 10000 + mask1[m1]

    # Encode mask2 (overwrites where overlapping - mask2 takes priority)
    if mask2.max() > 0:
        m2 = mask2 > 0
        combined[m2] = mask_id2 * 10000 + mask2[m2]

    return combined


def save_outputs(
    outdir: Path,
    masks: np.ndarray,
    volume_shape: Tuple[int, int, int],
    roi: Tuple[slice, slice, slice] | None,
    meta: dict,
) -> None:
    """Save segmentation outputs."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Save masks as TIFF
    mask_path = outdir / "segmentation_3d.tif"
    tifffile.imwrite(str(mask_path), masks)
    print(f"Saved masks: {mask_path}")

    # Save as Zarr for Neuroglancer compatibility
    zarr_path = outdir / "segmentation_3d.zarr"
    z = zarr.open(zarr_path, mode="w", shape=masks.shape, dtype=masks.dtype, chunks=(64, 64, 64))
    z[:] = masks
    print(f"Saved Zarr: {zarr_path}")

    # Save metadata
    if roi is not None:
        meta["roi"] = {
            "z": [roi[0].start or 0, roi[0].stop or volume_shape[0]],
            "y": [roi[1].start or 0, roi[1].stop or volume_shape[1]],
            "x": [roi[2].start or 0, roi[2].stop or volume_shape[2]],
        }

    meta["output_shape"] = list(masks.shape)
    meta["n_objects_total"] = int(masks.max())

    meta_path = outdir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="3D segmentation of OME-Zarr volumes using Cellpose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to OME-Zarr directory",
    )

    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to primary Cellpose model",
    )

    parser.add_argument(
        "--model2",
        type=Path,
        help="Path to secondary Cellpose model (optional)",
    )

    parser.add_argument(
        "--outdir", "-o",
        type=Path,
        default=Path("segmentation_output"),
        help="Output directory (default: segmentation_output)",
    )

    parser.add_argument(
        "--roi", "-r",
        type=str,
        help="Region of interest: 'z_start:z_end,y_start:y_end,x_start:x_end'. "
             "Use empty for full range, e.g., ':100,:,:' for first 100 z-slices.",
    )

    parser.add_argument(
        "--level", "-l",
        type=int,
        default=0,
        help="Pyramid level to use (0=full resolution, default: 0)",
    )

    # Model 1 parameters
    parser.add_argument(
        "--mask-id1",
        type=int,
        default=1,
        help="Class ID for primary model masks (default: 1)",
    )

    parser.add_argument(
        "--cellprob-threshold1", "--cp1",
        type=float,
        default=-1.0,
        help="Cell probability threshold for model 1 (default: -1.0)",
    )

    # Model 2 parameters
    parser.add_argument(
        "--mask-id2",
        type=int,
        default=2,
        help="Class ID for secondary model masks (default: 2)",
    )

    parser.add_argument(
        "--cellprob-threshold2", "--cp2",
        type=float,
        default=-1.0,
        help="Cell probability threshold for model 2 (default: -1.0)",
    )

    # Common parameters
    parser.add_argument(
        "--anisotropy", "-a",
        type=float,
        default=1.0,
        help="Z anisotropy factor (default: 1.0 = isotropic)",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="Minimum object size in voxels (default: 0)",
    )

    parser.add_argument(
        "--flow3d-smooth",
        type=float,
        default=0.0,
        help="3D flow smoothing (default: 0.0)",
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="GPU batch size for network inference (default: 8)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.zarr_path.exists():
        print(f"Error: Zarr path not found: {args.zarr_path}")
        sys.exit(1)

    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    if args.model2 and not args.model2.exists():
        print(f"Error: Model 2 not found: {args.model2}")
        sys.exit(1)

    if not HAS_CELLPOSE:
        print("Error: cellpose is not installed. Install with: pip install cellpose")
        sys.exit(1)

    # Load volume
    print(f"\nLoading: {args.zarr_path}")

    # First, get shape for ROI parsing
    root = zarr.open_group(args.zarr_path, mode="r")
    multiscales = root.attrs.get("multiscales", [])
    datasets = multiscales[0].get("datasets", [])
    arr = root[datasets[args.level]["path"]]
    full_shape = arr.shape

    # Parse ROI if provided
    roi = None
    if args.roi:
        roi = parse_roi(args.roi, full_shape)

    # Load data
    volume = load_zarr_roi(args.zarr_path, roi=roi, level=args.level)

    # Run primary model
    print(f"\n--- Model 1: {args.model.name} ---")
    masks1 = run_cellpose_3d(
        volume,
        args.model,
        anisotropy=args.anisotropy,
        cellprob_threshold=args.cellprob_threshold1,
        min_size=args.min_size,
        flow3D_smooth=args.flow3d_smooth,
        batch_size=args.batch_size,
    )

    # Run secondary model if provided
    if args.model2:
        print(f"\n--- Model 2: {args.model2.name} ---")
        masks2 = run_cellpose_3d(
            volume,
            args.model2,
            anisotropy=args.anisotropy,
            cellprob_threshold=args.cellprob_threshold2,
            min_size=args.min_size,
            flow3D_smooth=args.flow3d_smooth,
            batch_size=args.batch_size,
        )

        print(f"\nCombining masks (id1={args.mask_id1}, id2={args.mask_id2})...")
        final_masks = combine_masks(masks1, masks2, args.mask_id1, args.mask_id2)
    else:
        final_masks = masks1

    # Build metadata
    meta = {
        "source_zarr": str(args.zarr_path),
        "source_shape": list(full_shape),
        "pyramid_level": args.level,
        "model1": str(args.model),
        "mask_id1": args.mask_id1,
        "cellprob_threshold1": args.cellprob_threshold1,
        "anisotropy": args.anisotropy,
        "min_size": args.min_size,
        "flow3d_smooth": args.flow3d_smooth,
        "batch_size": args.batch_size,
    }

    if args.model2:
        meta["model2"] = str(args.model2)
        meta["mask_id2"] = args.mask_id2
        meta["cellprob_threshold2"] = args.cellprob_threshold2

    # Save outputs
    print(f"\nSaving outputs to: {args.outdir}")
    save_outputs(args.outdir, final_masks, full_shape, roi, meta)

    print("\nDone!")


if __name__ == "__main__":
    main()
