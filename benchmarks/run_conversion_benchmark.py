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

Results are saved to data/results/conversion_benchmarks.md
"""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from src.streaming_converter import convert_tiff_to_ome_zarr


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    num_levels: int
    compression: str
    runtime_seconds: float
    output_size_bytes: int
    input_size_bytes: int

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


def get_directory_size(path: Path) -> int:
    """Calculate total size of a directory recursively."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def run_single_benchmark(
    input_path: Path,
    output_path: Path,
    num_levels: int,
    compression: str,
    chunk_size: tuple[int, int, int],
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    input_size = input_path.stat().st_size

    # Clean up any existing output
    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"  Running: levels={num_levels}, compression={compression}")

    # Time the conversion
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

    elapsed = time.perf_counter() - start_time

    # Measure output size
    output_size = get_directory_size(output_path)

    print(f"    Runtime: {elapsed:.1f}s, Size: {output_size / 1024**3:.2f} GB")

    # Clean up output
    shutil.rmtree(output_path)

    return BenchmarkResult(
        num_levels=num_levels,
        compression=compression,
        runtime_seconds=elapsed,
        output_size_bytes=output_size,
        input_size_bytes=input_size,
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
        "| Levels | Compression | Runtime | Output Size | vs Original | Space Savings |",
        "|--------|-------------|---------|-------------|-------------|---------------|",
    ]

    for r in results:
        runtime_str = format_runtime(r.runtime_seconds)
        size_str = f"{r.output_size_gb:.2f} GB"
        ratio_str = f"{r.compression_ratio:.2%}"
        savings_str = f"{r.space_savings_pct:+.1f}%"

        lines.append(
            f"| {r.num_levels} | {r.compression} | {runtime_str} | {size_str} | {ratio_str} | {savings_str} |"
        )

    # Add analysis section
    lines.extend([
        "",
        "## Analysis",
        "",
        "### By Compression Method",
        "",
    ])

    # Group by compression
    compressions = ["blosc-lz4", "blosc-zstd", "none"]
    for comp in compressions:
        comp_results = [r for r in results if r.compression == comp]
        if comp_results:
            avg_ratio = sum(r.compression_ratio for r in comp_results) / len(comp_results)
            avg_runtime = sum(r.runtime_seconds for r in comp_results) / len(comp_results)
            lines.append(f"- **{comp}:** avg ratio {avg_ratio:.2%}, avg runtime {format_runtime(avg_runtime)}")

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
        "- **Compression ratio** = output size / input size (lower is better)",
        "- **Space savings** = percentage reduction from original (positive is smaller)",
        "- More pyramid levels increase output size due to additional downsampled copies",
        "- blosc-zstd typically achieves better compression but may be slower",
        "- blosc-lz4 offers good balance of speed and compression",
        "",
    ])

    return "\n".join(lines)


def main():
    """Run the benchmark suite."""
    # Configuration
    input_path = Path("data/raw/tomo_reco_id0004_t0008.tif")
    output_base = Path("data/processed/benchmark_temp.zarr")
    results_path = Path("data/results/conversion_benchmarks.md")

    chunk_size = (128, 128, 128)
    levels_to_test = [1, 3, 5, 7]
    compressions_to_test = ["blosc-lz4", "blosc-zstd", "none"]

    # Validate input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Input: {input_path}")
    print(f"Input size: {input_path.stat().st_size / 1024**3:.2f} GB")
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
                output_path=output_base,
                num_levels=num_levels,
                compression=compression,
                chunk_size=chunk_size,
            )
            results.append(result)
            print()

    # Generate and save report
    report = generate_markdown_report(results, input_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(report)

    print(f"Results saved to: {results_path}")
    print()
    print("=" * 60)
    print(report)

    return 0


if __name__ == "__main__":
    exit(main())
