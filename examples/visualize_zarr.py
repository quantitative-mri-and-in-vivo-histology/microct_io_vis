#!/usr/bin/env python3
"""
Neuroglancer visualization for OME-Zarr volumes.

Launches a local Neuroglancer viewer to visualize converted volumes.
Automatically configures shaders for uint16 data using stored metadata.

Usage:
    python examples/visualize_zarr.py data/processed/volume.zarr

    # Keep viewer open indefinitely (Ctrl+C to exit)
    python examples/visualize_zarr.py data/processed/volume.zarr --keep-open
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import neuroglancer
import zarr


def get_dtype_conversion_metadata(root: zarr.Group) -> dict | None:
    """Extract dtype conversion metadata if present."""
    try:
        multiscales = root.attrs.get("multiscales", [])
        if multiscales:
            metadata = multiscales[0].get("metadata", {})
            return metadata.get("dtype_conversion")
    except (KeyError, IndexError):
        pass
    return None




def visualize_zarr(zarr_path: Path, keep_open: bool = False) -> None:
    """
    Launch Neuroglancer viewer for an OME-Zarr volume.

    Args:
        zarr_path: Path to OME-Zarr directory
        keep_open: If True, keep viewer open until Ctrl+C
    """
    zarr_path = zarr_path.resolve()

    # Open zarr and read metadata
    root = zarr.open_group(zarr_path, mode="r")

    # Get multiscales metadata
    multiscales = root.attrs.get("multiscales", [])
    if not multiscales:
        print(f"Error: No multiscales metadata found in {zarr_path}")
        sys.exit(1)

    ms = multiscales[0]
    datasets = ms.get("datasets", [])
    axes = ms.get("axes", [])

    if not datasets:
        print(f"Error: No datasets found in multiscales metadata")
        sys.exit(1)

    # Get the full resolution array to determine shape and dtype
    level0_path = datasets[0]["path"]
    level0 = root[level0_path]

    print(f"Zarr path: {zarr_path}")
    print(f"Shape: {level0.shape}")
    print(f"Dtype: {level0.dtype}")
    print(f"Pyramid levels: {len(datasets)}")

    # Check for dtype conversion metadata
    dtype_conversion = get_dtype_conversion_metadata(root)
    if dtype_conversion:
        source_range = dtype_conversion.get("source_range", [0, 1])
        print(f"Dtype conversion: {dtype_conversion['source_dtype']} -> {dtype_conversion['output_dtype']}")
        print(f"Original value range: [{source_range[0]:.6f}, {source_range[1]:.6f}]")

    # Extract axis info
    axis_names = [ax.get("name", f"d{i}") for i, ax in enumerate(axes)]
    axis_units = [ax.get("unit", "") for ax in axes]

    # Get scale from first dataset's coordinate transformations
    scale = [1.0] * len(axes)
    coord_transforms = datasets[0].get("coordinateTransformations", [])
    for transform in coord_transforms:
        if transform.get("type") == "scale":
            scale = transform.get("scale", scale)
            break

    # Set up neuroglancer
    neuroglancer.set_server_bind_address("127.0.0.1")
    viewer = neuroglancer.Viewer()

    # Create coordinate space
    # Neuroglancer expects SI unit abbreviations (um, nm, etc.), not full names
    unit_map = {"micrometer": "um", "nanometer": "nm", "millimeter": "mm", "meter": "m"}
    normalized_units = [unit_map.get(u, u) if u else "um" for u in axis_units]

    dimensions = neuroglancer.CoordinateSpace(
        names=axis_names,
        units=normalized_units if all(normalized_units) else ["um"] * len(axes),
        scales=scale,
    )

    # Add the volume layer using LocalVolume
    # Note: This only uses level 0, not the full pyramid (zarr v3 not supported via HTTP yet)
    with viewer.txn() as s:
        s.dimensions = dimensions

        s.layers["volume"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=level0,
                dimensions=dimensions,
            ),
        )

        # Set initial position to center of volume
        s.position = [d // 2 for d in level0.shape]

        # Set zoom to fit the largest dimension in view
        max_dim = max(level0.shape[1], level0.shape[2])  # Y or X
        s.crossSectionScale = max_dim / 500

    print()
    print(f"Neuroglancer viewer URL: {viewer.get_viewer_url()}")
    print()

    # Open browser
    webbrowser.open(viewer.get_viewer_url())

    if keep_open:
        print("Viewer running. Press Ctrl+C to exit.")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down viewer.")
    else:
        print("Press Enter to close viewer...")
        input()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OME-Zarr volumes with Neuroglancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to OME-Zarr directory",
    )

    parser.add_argument(
        "--keep-open", "-k",
        action="store_true",
        help="Keep viewer open until Ctrl+C (default: wait for Enter)",
    )

    args = parser.parse_args()

    # Validate path exists
    if not args.zarr_path.exists():
        print(f"Error: Path not found: {args.zarr_path}")
        sys.exit(1)

    if not args.zarr_path.is_dir():
        print(f"Error: Not a directory: {args.zarr_path}")
        sys.exit(1)

    visualize_zarr(args.zarr_path, keep_open=args.keep_open)


if __name__ == "__main__":
    main()
