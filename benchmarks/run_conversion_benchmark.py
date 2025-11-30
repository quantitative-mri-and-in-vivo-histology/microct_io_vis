#!/usr/bin/env python3
"""
Benchmark script for TIFF to OME-Zarr conversion.

Tests different combinations of:
- Pyramid levels: 1, 3, 5, 7
- Compression: blosc-lz4, blosc-zstd, none

Measures:
- Conversion runtime
- Output size
- Compression ratio vs original

Usage:
    python benchmarks/run_conversion_benchmark.py input.tif output_results.md
    python benchmarks/run_conversion_benchmark.py  # Uses default paths
"""

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import zarr

from src.streaming_converter import convert_tiff_to_ome_zarr


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    num_levels: int
    compression: str
    write_time_seconds: float
    read_time_seconds: float
    output_size_bytes: int
    input_size_bytes: int
    read_bytes: int
    num_chunks: int

    @property
    def output_size_gb(self) -> float:
        return self.output_size_bytes / (1024**3)

    @property
    def input_size_gb(self) -> float:
        return self.input_size_bytes / (1024**3)

    @property
    def compression_ratio(self) -> float:
        """Ratio of output size to input size."""
        if self.input_size_bytes == 0:
            return 0.0
        return self.output_size_bytes / self.input_size_bytes

    @property
    def space_savings_pct(self) -> float:
        """Percentage of space saved (negative means larger)."""
        return (1 - self.compression_ratio) * 100

    @property
    def read_throughput_gbps(self) -> float:
        """Read throughput in GB/s (decompressed data)."""
        if self.read_time_seconds == 0:
            return 0.0
        return (self.read_bytes / (1024**3)) / self.read_time_seconds


def get_directory_size(path: Path) -> int:
    """Calculate total size of a directory recursively."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def run_read_benchmark(zarr_path: Path) -> tuple[float, int, int]:
    """
    Read all chunks from level 0 sequentially.

    Returns:
        Tuple of (read_time_seconds, bytes_read, num_chunks)
    """
    root = zarr.open_group(zarr_path, mode="r")
    level0 = root["0"]

    # Get chunk grid dimensions
    chunks_per_dim = [
        (level0.shape[i] + level0.chunks[i] - 1) // level0.chunks[i]
        for i in range(3)
    ]

    start_time = time.perf_counter()
    bytes_read = 0
    num_chunks = 0

    # Read all chunks sequentially (z, y, x order)
    for z in range(chunks_per_dim[0]):
        for y in range(chunks_per_dim[1]):
            for x in range(chunks_per_dim[2]):
                # Calculate slice bounds
                z_start = z * level0.chunks[0]
                z_end = min((z + 1) * level0.chunks[0], level0.shape[0])
                y_start = y * level0.chunks[1]
                y_end = min((y + 1) * level0.chunks[1], level0.shape[1])
                x_start = x * level0.chunks[2]
                x_end = min((x + 1) * level0.chunks[2], level0.shape[2])

                # Read chunk (triggers decompression)
                chunk = level0[z_start:z_end, y_start:y_end, x_start:x_end]
                bytes_read += chunk.nbytes
                num_chunks += 1

    elapsed = time.perf_counter() - start_time
    return elapsed, bytes_read, num_chunks


def run_single_benchmark(
    input_path: Path,
    num_levels: int,
    compression: str,
    chunk_size: tuple[int, int, int],
) -> BenchmarkResult:
    """Run a single benchmark configuration (write + read)."""
    input_size = input_path.stat().st_size

    print(f"  Running: levels={num_levels}, compression={compression}")

    # Use temp directory for automatic cleanup
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "benchmark.zarr"

        # Time the conversion (write benchmark)
        start_time = time.perf_counter()

        convert_tiff_to_ome_zarr(
            tiff_path=input_path,
            zarr_path=output_path,
            chunk_size=chunk_size,
            num_levels=num_levels,
            compression=compression,
            max_memory="8G",  # Use 8GB memory budget
            progress=True,
        )

        write_elapsed = time.perf_counter() - start_time

        # Measure output size
        output_size = get_directory_size(output_path)

        print(f"    Write: {write_elapsed:.1f}s, Size: {output_size / 1024**3:.2f} GB")

        # Run read benchmark
        read_elapsed, read_bytes, num_chunks = run_read_benchmark(output_path)
        read_throughput = (read_bytes / (1024**3)) / read_elapsed

        print(f"    Read: {read_elapsed:.1f}s, Throughput: {read_throughput:.2f} GB/s")

    # Temp directory cleaned up automatically on exit

    return BenchmarkResult(
        num_levels=num_levels,
        compression=compression,
        write_time_seconds=write_elapsed,
        read_time_seconds=read_elapsed,
        output_size_bytes=output_size,
        input_size_bytes=input_size,
        read_bytes=read_bytes,
        num_chunks=num_chunks,
    )


def format_runtime(seconds: float) -> str:
    """Format runtime as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def generate_markdown_report(results: list[BenchmarkResult], input_path: Path) -> str:
    """Generate markdown report from benchmark results."""
    input_size_gb = results[0].input_size_gb if results else 0

    lines = [
        "# TIFF to OME-Zarr Conversion Benchmarks",
        "",
        "## Configuration",
        "",
        f"- **Input file:** `{input_path.name}`",
        f"- **Input size:** {input_size_gb:.2f} GB",
        "- **Chunk size:** 128 x 128 x 128",
        "- **Memory budget:** 8GB",
        "",
        "## Results",
        "",
        "| Levels | Compression | Write Time | Read Time | Read Throughput | Output Size | Compression Ratio |",
        "|--------|-------------|------------|-----------|-----------------|-------------|-------------------|",
    ]

    for r in results:
        write_str = format_runtime(r.write_time_seconds)
        read_str = format_runtime(r.read_time_seconds)
        throughput_str = f"{r.read_throughput_gbps:.2f} GB/s"
        size_str = f"{r.output_size_gb:.2f} GB"
        ratio_str = f"{r.compression_ratio:.2%}"

        lines.append(
            f"| {r.num_levels} | {r.compression} | {write_str} | {read_str} | {throughput_str} | {size_str} | {ratio_str} |"
        )

    # Add analysis section
    lines.extend([
        "",
        "## Analysis",
        "",
        "### Read Performance by Compression",
        "",
    ])

    # Group by compression for read performance
    compressions = ["blosc-lz4", "blosc-zstd", "none"]
    for comp in compressions:
        comp_results = [r for r in results if r.compression == comp]
        if comp_results:
            avg_throughput = sum(r.read_throughput_gbps for r in comp_results) / len(comp_results)
            avg_ratio = sum(r.compression_ratio for r in comp_results) / len(comp_results)
            lines.append(f"- **{comp}:** avg read throughput {avg_throughput:.2f} GB/s, avg compression ratio {avg_ratio:.2%}")

    lines.extend([
        "",
        "### Write Performance by Compression",
        "",
    ])

    for comp in compressions:
        comp_results = [r for r in results if r.compression == comp]
        if comp_results:
            avg_write = sum(r.write_time_seconds for r in comp_results) / len(comp_results)
            lines.append(f"- **{comp}:** avg write time {format_runtime(avg_write)}")

    lines.extend([
        "",
        "### By Pyramid Levels",
        "",
    ])

    # Group by levels
    levels_list = [1, 3, 5, 7]
    for level in levels_list:
        level_results = [r for r in results if r.num_levels == level]
        if level_results:
            avg_size = sum(r.output_size_gb for r in level_results) / len(level_results)
            lines.append(f"- **{level} levels:** avg output size {avg_size:.2f} GB")

    lines.extend([
        "",
        "### Notes",
        "",
        "- **Read throughput** = decompressed bytes / read time (higher is better for viewer performance)",
        "- **Compression ratio** = output size / input size (lower is better for storage)",
        "- More pyramid levels increase output size due to additional downsampled copies",
        "- blosc-lz4 typically offers best read throughput (fastest decompression)",
        "- blosc-zstd achieves better compression but may have slower decompression",
        "",
    ])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark TIFF to OME-Zarr conversion with various configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s input.tif results.md
    %(prog)s data/raw/volume.tif data/results/benchmark.md
    %(prog)s  # Uses default paths
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/raw/tomo_reco_id0004_t0008.tif",
        help="Path to input TIFF file (default: data/raw/tomo_reco_id0004_t0008.tif)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="data/results/conversion_benchmarks.md",
        help="Path to output markdown report (default: data/results/conversion_benchmarks.md)",
    )
    return parser.parse_args()


def main():
    """Run the benchmark suite."""
    args = parse_args()

    input_path = Path(args.input)
    results_path = Path(args.output)

    chunk_size = (128, 128, 128)
    levels_to_test = [1, 3, 5, 7]
    compressions_to_test = ["blosc-lz4", "blosc-zstd", "none"]

    # Validate input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Create output directory if needed
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Input size: {input_path.stat().st_size / 1024**3:.2f} GB")
    print(f"Output: {results_path}")
    print()

    # Run all benchmarks
    results = []
    total_runs = len(levels_to_test) * len(compressions_to_test)
    current_run = 0

    for num_levels in levels_to_test:
        for compression in compressions_to_test:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Benchmarking...")

            result = run_single_benchmark(
                input_path=input_path,
                num_levels=num_levels,
                compression=compression,
                chunk_size=chunk_size,
            )
            results.append(result)
            print()

    # Generate and save report
    report = generate_markdown_report(results, input_path)

    results_path.write_text(report)

    print(f"Results saved to: {results_path}")
    print()
    print("=" * 60)
    print(report)

    return 0


if __name__ == "__main__":
    exit(main())
