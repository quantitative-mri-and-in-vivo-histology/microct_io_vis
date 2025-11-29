#!/usr/bin/env python3
"""
CLI for converting BigTIFF volumes to OME-Zarr with multi-resolution pyramids.

Usage:
    python examples/convert_tiff_to_zarr.py input.tif output.zarr --chunk-size 128 128 128

Examples:
    # Visualization-optimized (isotropic chunks for Neuroglancer)
    python examples/convert_tiff_to_zarr.py data/raw/volume.tif data/processed/volume.zarr \
        --chunk-size 128 128 128 --num-levels 4

    # Conversion-optimized (z-aligned chunks, faster writes)
    python examples/convert_tiff_to_zarr.py data/raw/volume.tif data/processed/volume.zarr \
        --chunk-size 32 256 256 --num-levels 4 --max-memory 8G

    # Show info about a TIFF without converting
    python examples/convert_tiff_to_zarr.py data/raw/volume.tif --info
"""

import argparse
import sys
from pathlib import Path

import tifffile

from src.streaming_converter import (
    StreamingPyramidConverter,
    convert_tiff_to_ome_zarr,
    parse_memory_string,
)


def show_tiff_info(tiff_path: Path) -> None:
    """Display information about a TIFF file without loading data."""
    print(f"TIFF Info: {tiff_path}")
    print("-" * 60)

    with tifffile.TiffFile(tiff_path) as tif:
        n_pages = len(tif.pages)
        page = tif.pages[0]

        print(f"  BigTIFF:      {tif.is_bigtiff}")
        print(f"  Pages:        {n_pages}")
        print(f"  Page shape:   {page.shape} (Y, X)")
        print(f"  Dtype:        {page.dtype}")
        print(f"  Compression:  {page.compression}")
        print(f"  Tiled:        {page.is_tiled}")

        # Calculate volume info
        z, y, x = n_pages, page.shape[0], page.shape[1]
        bytes_per_voxel = page.dtype.itemsize
        total_bytes = z * y * x * bytes_per_voxel
        total_gb = total_bytes / (1024**3)

        print()
        print(f"  Volume shape: ({z}, {y}, {x}) (Z, Y, X)")
        print(f"  Total size:   {total_gb:.2f} GB")

        # Suggest pyramid levels
        min_dim = min(y, x)
        max_levels = 1
        while min_dim // (2 ** max_levels) >= 32:
            max_levels += 1

        print()
        print(f"  Suggested num_levels: {min(max_levels, 5)} (max {max_levels} before spatial dims < 32)")

        # Check divisibility
        for levels in range(2, min(max_levels + 1, 6)):
            divisor = 2 ** (levels - 1)
            y_ok = y % divisor == 0
            x_ok = x % divisor == 0
            status = "OK" if (y_ok and x_ok) else f"FAIL (need Y,X divisible by {divisor})"
            print(f"    num_levels={levels}: {status}")


def estimate_conversion(tiff_path: Path, chunk_size: tuple, num_levels: int, max_memory_bytes: int | None) -> None:
    """Show conversion estimates without actually converting."""
    print("Conversion Estimate")
    print("-" * 60)

    try:
        converter = StreamingPyramidConverter(
            tiff_path=tiff_path,
            zarr_path=Path("/tmp/dummy.zarr"),  # Not actually created
            chunk_size=chunk_size,
            num_levels=num_levels,
            max_memory_bytes=max_memory_bytes,
        )

        print(f"  Source shape:   {converter.source_shape}")
        print(f"  Padded shape:   {converter.padded_shape}")
        print(f"  Batch size:     {converter.batch_size} slices")
        print(f"  Num batches:    {(converter.source_shape[0] + converter.batch_size - 1) // converter.batch_size}")
        print()

        print("  Pyramid levels:")
        for level, shape in enumerate(converter.pyramid_shapes):
            scale = 2 ** level
            print(f"    Level {level} ({scale}x): {shape}")
        print()

        mem = converter.estimate_memory_usage()
        if mem['using_default_budget']:
            print(f"  Memory budget:  {mem['memory_budget_mb']:.0f} MB (default, set --max-memory to customize)")
        else:
            print(f"  Memory budget:  {mem['memory_budget_mb']:.0f} MB")
        print(f"  Buffer threshold: {mem['buffer_threshold']} batches")
        print(f"  Estimated peak: {mem['estimated_peak_mb']:.0f} MB")

    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert BigTIFF to OME-Zarr with multi-resolution pyramids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("input", type=Path, help="Input BigTIFF file")
    parser.add_argument("output", type=Path, nargs="?", help="Output OME-Zarr directory")

    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Chunk size (Z Y X). Required for conversion. "
             "Use 128 128 128 for visualization, 32 256 256 for fast conversion.",
    )

    parser.add_argument(
        "--num-levels", "-n",
        type=int,
        default=4,
        help="Number of pyramid levels (default: 4 = 1x,2x,4x,8x)",
    )

    parser.add_argument(
        "--compression",
        choices=["blosc-lz4", "blosc-zstd", "none"],
        default="blosc-lz4",
        help="Compression codec (default: blosc-lz4)",
    )

    parser.add_argument(
        "--max-memory", "-m",
        type=str,
        default=None,
        metavar="SIZE",
        help="Maximum memory budget (e.g., '4G', '500M', '2GB'). "
             "Larger values buffer more data, reducing I/O. Default: 4GB",
    )

    parser.add_argument(
        "--dtype", "-d",
        choices=["float32", "uint16"],
        default="float32",
        help="Output data type. 'uint16' reduces storage by 50%% but requires "
             "normalization. Original values can be recovered from metadata. "
             "(default: float32)",
    )

    parser.add_argument(
        "--value-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Value range for uint16 normalization. If not provided, the TIFF "
             "will be pre-scanned to find the range (~10s for 56GB).",
    )

    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show TIFF info and exit (no conversion)",
    )

    parser.add_argument(
        "--estimate", "-e",
        action="store_true",
        help="Show conversion estimates and exit (no conversion)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate input exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Info mode
    if args.info:
        show_tiff_info(args.input)
        sys.exit(0)

    # Estimate mode
    if args.estimate:
        if args.chunk_size is None:
            print("Error: --chunk-size required for estimate")
            sys.exit(1)
        max_memory_bytes = parse_memory_string(args.max_memory) if args.max_memory else None
        estimate_conversion(
            args.input,
            tuple(args.chunk_size),
            args.num_levels,
            max_memory_bytes,
        )
        sys.exit(0)

    # Conversion mode - validate required args
    if args.output is None:
        print("Error: Output path required for conversion")
        parser.print_usage()
        sys.exit(1)

    if args.chunk_size is None:
        print("Error: --chunk-size required for conversion")
        print("  Visualization-optimized: --chunk-size 128 128 128")
        print("  Conversion-optimized:    --chunk-size 32 256 256")
        sys.exit(1)

    # Prepare value_range if provided
    value_range = tuple(args.value_range) if args.value_range else None

    # Run conversion
    try:
        convert_tiff_to_ome_zarr(
            tiff_path=args.input,
            zarr_path=args.output,
            chunk_size=tuple(args.chunk_size),
            num_levels=args.num_levels,
            compression=args.compression,
            max_memory=args.max_memory,
            output_dtype=args.dtype,
            value_range=value_range,
            progress=not args.quiet,
        )

        if not args.quiet:
            print(f"\nOutput: {args.output}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nConversion cancelled")
        sys.exit(130)


if __name__ == "__main__":
    main()
