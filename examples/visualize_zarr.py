#!/usr/bin/env python3
"""
Neuroglancer visualization for OME-Zarr volumes.

Launches a local Neuroglancer viewer to visualize converted volumes with
multi-resolution pyramid support via HTTP serving.

Usage:
    python examples/visualize_zarr.py data/processed/volume.zarr

    # Keep viewer open indefinitely (Ctrl+C to exit)
    python examples/visualize_zarr.py data/processed/volume.zarr --keep-open
"""

import argparse
import http.server
import socketserver
import sys
import threading
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


def start_http_server(zarr_path: Path, port: int = 0) -> tuple[socketserver.TCPServer, int]:
    """
    Start an HTTP server to serve the zarr directory.

    Args:
        zarr_path: Path to OME-Zarr directory to serve
        port: Port to use (0 = auto-select)

    Returns:
        Tuple of (server, actual_port)
    """
    # Serve the parent directory so the zarr folder name is in the URL
    serve_dir = zarr_path.parent

    class CORSHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def end_headers(self):
            # Add CORS headers for Neuroglancer
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            # Suppress HTTP request logging
            pass

    server = socketserver.TCPServer(("127.0.0.1", port), CORSHandler)
    actual_port = server.server_address[1]

    # Run server in background thread
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server, actual_port


def visualize_zarr(zarr_path: Path, keep_open: bool = False) -> None:
    """
    Launch Neuroglancer viewer for an OME-Zarr volume.

    Uses HTTP serving to enable multi-resolution pyramid support.

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

    # Start HTTP server for the zarr directory
    server, http_port = start_http_server(zarr_path)
    zarr_http_url = f"http://127.0.0.1:{http_port}/{zarr_path.name}"
    print(f"HTTP server: {zarr_http_url}")

    # Set up neuroglancer
    neuroglancer.set_server_bind_address("127.0.0.1")
    viewer = neuroglancer.Viewer()

    # Build zarr:// source URL for OME-Zarr with multiscale support
    zarr_source_url = f"zarr://{zarr_http_url}"

    with viewer.txn() as s:
        # Add the volume layer using HTTP-served zarr with pyramids
        s.layers["volume"] = neuroglancer.ImageLayer(source=zarr_source_url)

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

    # Clean up
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
