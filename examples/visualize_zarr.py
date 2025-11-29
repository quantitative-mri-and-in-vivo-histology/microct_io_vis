#!/usr/bin/env python3
"""
Neuroglancer visualization for OME-Zarr volumes.

Launches a local Neuroglancer viewer to visualize converted volumes.
Supports multi-resolution pyramids via HTTP serving.

Usage:
    python examples/visualize_zarr.py data/processed/volume.zarr

    # Keep viewer open indefinitely (Ctrl+C to exit)
    python examples/visualize_zarr.py data/processed/volume.zarr --keep-open

    # Use LocalVolume instead of HTTP (level 0 only)
    python examples/visualize_zarr.py data/processed/volume.zarr --local
"""

import argparse
import sys
import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import neuroglancer
import numpy as np
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


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS support for Neuroglancer."""

    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP access logs
        pass


def start_cors_server(directory: Path, port: int = 8080) -> tuple[HTTPServer, int]:
    """Start a CORS-enabled HTTP server for serving zarr data."""
    handler = partial(CORSRequestHandler, directory=str(directory))

    # Try to find an available port
    for attempt in range(10):
        try:
            server = HTTPServer(("localhost", port + attempt), handler)
            actual_port = port + attempt
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"Could not find available port starting from {port}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server, actual_port


def visualize_zarr(zarr_path: Path, keep_open: bool = False, local: bool = False, port: int = 9999) -> None:
    """
    Launch Neuroglancer viewer for an OME-Zarr volume.

    Args:
        zarr_path: Path to OME-Zarr directory
        keep_open: If True, keep viewer open until Ctrl+C
        local: If True, use LocalVolume (level 0 only). If False, use HTTP with pyramids.
        port: Port for Neuroglancer viewer (default: 9999)
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

    # Extract axis info for coordinate space
    axis_names = [ax.get("name", f"d{i}") for i, ax in enumerate(axes)]
    axis_units = [ax.get("unit", "") for ax in axes]

    # Get scale from first dataset's coordinate transformations
    scale = [1.0] * len(axes)
    coord_transforms = datasets[0].get("coordinateTransformations", [])
    for transform in coord_transforms:
        if transform.get("type") == "scale":
            scale = transform.get("scale", scale)
            break

    # Neuroglancer expects SI unit abbreviations (um, nm, etc.), not full names
    unit_map = {"micrometer": "um", "nanometer": "nm", "millimeter": "mm", "meter": "m"}
    normalized_units = [unit_map.get(u, u) if u else "um" for u in axis_units]

    dimensions = neuroglancer.CoordinateSpace(
        names=axis_names,
        units=normalized_units if all(normalized_units) else ["um"] * len(axes),
        scales=scale,
    )

    # Set up neuroglancer viewer on specified port
    neuroglancer.set_server_bind_address("127.0.0.1", port)
    viewer = neuroglancer.Viewer()

    server = None
    if local:
        # LocalVolume mode - level 0 only
        print("Mode: LocalVolume (level 0 only)")

        with viewer.txn() as s:
            s.dimensions = dimensions
            s.layers["volume"] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    data=level0,
                    dimensions=dimensions,
                ),
            )
            s.position = [d // 2 for d in level0.shape]
            max_dim = max(level0.shape[1], level0.shape[2])
            s.crossSectionScale = max_dim / 500
    else:
        # HTTP mode - serves all pyramid levels
        print("Mode: HTTP (multi-resolution pyramid)")

        # Start CORS server in zarr parent directory (port 8000+)
        server, http_port = start_cors_server(zarr_path.parent)

        zarr_url = f"zarr://http://localhost:{http_port}/{zarr_path.name}"

        # Get intensity range from lowest resolution level (fast to read)
        # Use 1st and 99th percentile for better contrast
        lowest_level = len(datasets) - 1
        lowest_arr = root[datasets[lowest_level]["path"]]
        sample_data = lowest_arr[:]
        data_min, data_max = int(np.percentile(sample_data, 1)), int(np.percentile(sample_data, 99))

        # Create shader with appropriate intensity window
        shader = f"""
#uicontrol invlerp normalized(range=[{data_min}, {data_max}])
void main() {{
  emitGrayscale(normalized());
}}
"""

        with viewer.txn() as s:
            s.layers["volume"] = neuroglancer.ImageLayer(
                source=zarr_url,
                shader=shader,
            )

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

    if server:
        server.shutdown()


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

    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Use LocalVolume (level 0 only) instead of HTTP pyramid serving",
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9999,
        help="Port for Neuroglancer viewer (default: 9999)",
    )

    args = parser.parse_args()

    # Validate path exists
    if not args.zarr_path.exists():
        print(f"Error: Path not found: {args.zarr_path}")
        sys.exit(1)

    if not args.zarr_path.is_dir():
        print(f"Error: Not a directory: {args.zarr_path}")
        sys.exit(1)

    visualize_zarr(args.zarr_path, keep_open=args.keep_open, local=args.local, port=args.port)


if __name__ == "__main__":
    main()
