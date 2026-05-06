#!/usr/bin/env python3
"""
Neuroglancer visualization for OME-Zarr volumes.

Launches a local Neuroglancer viewer to visualize a grayscale volume and/or
one or more segmentation overlays. Multiple zarrs are symlinked into a temp
directory and served by a single CORS HTTP server.

Usage:
    # Grayscale only
    python examples/visualize_zarr.py data/processed/volume.zarr

    # Grayscale + segmentation overlay
    python examples/visualize_zarr.py data/processed/volume.zarr \
        --segmentation data/processed/segmentation/segmentation_3d.zarr

    # Multiple segmentations (e.g. cell + blood)
    python examples/visualize_zarr.py data/processed/volume.zarr \
        --segmentation cells.zarr blood.zarr

    # Segmentation only
    python examples/visualize_zarr.py --segmentation seg.zarr

    # Keep viewer open indefinitely (Ctrl+C to exit)
    python examples/visualize_zarr.py data/processed/volume.zarr --keep-open

    # Use LocalVolume instead of HTTP (level 0 only, grayscale only)
    python examples/visualize_zarr.py data/processed/volume.zarr --local
"""

import argparse
import sys
import tempfile
import threading
import time
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import neuroglancer
import numpy as np
import zarr


UNIT_MAP = {"micrometer": "um", "nanometer": "nm", "millimeter": "mm", "meter": "m"}


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


def read_zarr_meta(zarr_path: Path) -> dict:
    """Read OME-Zarr metadata needed for layer setup."""
    root = zarr.open_group(str(zarr_path), mode="r")
    multiscales = root.attrs.get("multiscales", [])
    if not multiscales:
        raise ValueError(f"No multiscales metadata in {zarr_path}")

    ms = multiscales[0]
    datasets = ms.get("datasets", [])
    axes = ms.get("axes", [])
    if not datasets:
        raise ValueError(f"No datasets in multiscales for {zarr_path}")

    level0 = root[datasets[0]["path"]]

    scale = [1.0] * len(axes)
    for t in datasets[0].get("coordinateTransformations", []):
        if t.get("type") == "scale":
            scale = list(t.get("scale", scale))
            break

    axis_names = [ax.get("name", f"d{i}") for i, ax in enumerate(axes)]
    axis_units = [UNIT_MAP.get(ax.get("unit", ""), ax.get("unit", "") or "um") for ax in axes]

    return {
        "root": root,
        "datasets": datasets,
        "level0": level0,
        "shape": tuple(level0.shape),
        "dtype": level0.dtype,
        "scale": scale,
        "axis_names": axis_names,
        "axis_units": axis_units,
        "dtype_conversion": get_dtype_conversion_metadata(root),
        "lowest_level_path": datasets[-1]["path"],
        "n_levels": len(datasets),
    }


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
        pass


def start_cors_server(directory: Path, port: int = 8080) -> tuple[HTTPServer, int]:
    handler = partial(CORSRequestHandler, directory=str(directory))
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


def add_grayscale_layer(viewer, name: str, url: str, meta: dict) -> None:
    lowest = meta["root"][meta["lowest_level_path"]][:]
    data_min = float(np.percentile(lowest, 1))
    data_max = float(np.percentile(lowest, 99))
    print(f"  [{name}] shader range (1st-99th pct): [{data_min:.6g}, {data_max:.6g}]")

    shader = f"""
#uicontrol invlerp normalized(range=[{data_min:.8g}, {data_max:.8g}])
void main() {{
  emitGrayscale(normalized());
}}
"""
    with viewer.txn() as s:
        s.layers[name] = neuroglancer.ImageLayer(source=url, shader=shader)


def add_segmentation_layer(viewer, name: str, url: str) -> None:
    with viewer.txn() as s:
        s.layers[name] = neuroglancer.SegmentationLayer(source=url)


def visualize(
    grayscale: Path | None,
    segmentations: list[Path],
    keep_open: bool = False,
    local: bool = False,
    port: int = 9999,
) -> None:
    if grayscale is None and not segmentations:
        print("Error: provide a grayscale path and/or --segmentation")
        sys.exit(1)

    if local and segmentations:
        print("Error: --local mode does not support segmentation overlays "
              "(use HTTP mode by omitting --local)")
        sys.exit(1)

    # Pre-read metadata for all inputs (validates and gives us coord space).
    gray_meta = read_zarr_meta(grayscale.resolve()) if grayscale else None
    seg_metas = [(p, read_zarr_meta(p.resolve())) for p in segmentations]

    primary_meta = gray_meta if gray_meta is not None else seg_metas[0][1]

    print("Inputs:")
    if grayscale:
        print(f"  grayscale: {grayscale}  shape={gray_meta['shape']} "
              f"dtype={gray_meta['dtype']} levels={gray_meta['n_levels']}")
        if gray_meta["dtype_conversion"]:
            dc = gray_meta["dtype_conversion"]
            sr = dc.get("source_range", [0, 1])
            print(f"    dtype_conversion: {dc['source_dtype']} -> {dc['output_dtype']}, "
                  f"source range [{sr[0]:.6g}, {sr[1]:.6g}]")
    for p, m in seg_metas:
        print(f"  segmentation: {p}  shape={m['shape']} dtype={m['dtype']} "
              f"levels={m['n_levels']}")

    dimensions = neuroglancer.CoordinateSpace(
        names=primary_meta["axis_names"],
        units=primary_meta["axis_units"],
        scales=primary_meta["scale"],
    )

    neuroglancer.set_server_bind_address("127.0.0.1", port)
    viewer = neuroglancer.Viewer()

    server = None
    tmp_ctx = None

    if local:
        print("Mode: LocalVolume (level 0 only)")
        with viewer.txn() as s:
            s.dimensions = dimensions
            s.layers[grayscale.stem] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    data=gray_meta["level0"],
                    dimensions=dimensions,
                ),
            )
            shape = gray_meta["shape"]
            scale = gray_meta["scale"]
            s.position = [d / 2 * sc for d, sc in zip(shape, scale)]
            max_dim = max(shape[1], shape[2])
            s.crossSectionScale = max_dim / 500
    else:
        print("Mode: HTTP (multi-resolution pyramid)")
        tmp_ctx = tempfile.TemporaryDirectory(prefix="neuroglancer_")
        tmp_dir = Path(tmp_ctx.name)

        layer_specs = []
        if gray_meta is not None:
            link_name = f"{grayscale.stem}_gray.zarr"
            (tmp_dir / link_name).symlink_to(grayscale.resolve())
            layer_specs.append(("grayscale", grayscale.stem, link_name, gray_meta))
        for p, m in seg_metas:
            link_name = f"{p.stem}_seg.zarr"
            target = tmp_dir / link_name
            if target.exists():
                link_name = f"{p.stem}_{len(layer_specs)}_seg.zarr"
                target = tmp_dir / link_name
            target.symlink_to(p.resolve())
            layer_specs.append(("segmentation", p.stem, link_name, m))

        server, http_port = start_cors_server(tmp_dir)
        print(f"HTTP server on port {http_port}")

        with viewer.txn() as s:
            s.dimensions = dimensions

        for layer_type, stem, link_name, m in layer_specs:
            url = f"zarr://http://localhost:{http_port}/{link_name}"
            if layer_type == "grayscale":
                add_grayscale_layer(viewer, stem, url, m)
            else:
                add_segmentation_layer(viewer, stem, url)

        with viewer.txn() as s:
            shape = primary_meta["shape"]
            scale = primary_meta["scale"]
            s.position = [d / 2 * sc for d, sc in zip(shape, scale)]

    print()
    print(f"Neuroglancer viewer URL: {viewer.get_viewer_url()}")
    print()

    webbrowser.open(viewer.get_viewer_url())

    if keep_open:
        print("Viewer running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down viewer.")
    else:
        print("Press Enter to close viewer...")
        input()

    if server:
        server.shutdown()
    if tmp_ctx:
        tmp_ctx.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OME-Zarr volumes (grayscale + segmentation) with Neuroglancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "zarr_path",
        type=Path,
        nargs="?",
        help="Path to grayscale OME-Zarr (optional if --segmentation is given)",
    )

    parser.add_argument(
        "--segmentation", "-s",
        type=Path,
        nargs="+",
        default=[],
        help="One or more segmentation OME-Zarr paths to overlay as label layers",
    )

    parser.add_argument(
        "--keep-open", "-k",
        action="store_true",
        help="Keep viewer open until Ctrl+C (default: wait for Enter)",
    )

    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Use LocalVolume (grayscale only, no segmentation overlays)",
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9999,
        help="Port for Neuroglancer viewer (default: 9999)",
    )

    args = parser.parse_args()

    if args.zarr_path is not None:
        if not args.zarr_path.exists():
            print(f"Error: Path not found: {args.zarr_path}")
            sys.exit(1)
        if not args.zarr_path.is_dir():
            print(f"Error: Not a directory: {args.zarr_path}")
            sys.exit(1)

    for p in args.segmentation:
        if not p.exists():
            print(f"Error: Segmentation path not found: {p}")
            sys.exit(1)
        if not p.is_dir():
            print(f"Error: Segmentation path is not a directory: {p}")
            sys.exit(1)

    visualize(
        grayscale=args.zarr_path,
        segmentations=list(args.segmentation),
        keep_open=args.keep_open,
        local=args.local,
        port=args.port,
    )


if __name__ == "__main__":
    main()
