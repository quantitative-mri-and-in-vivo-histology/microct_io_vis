"""
Streaming BigTIFF to OME-Zarr converter with multi-resolution pyramid support.

This module converts large BigTIFF volumes (stored as stacked 2D pages) to
OME-Zarr format with multi-resolution pyramids, using a memory-efficient
streaming approach that never loads the full volume into RAM.

Key insight: The largest downsampling level determines the batch size.
For max_level=3 (8x downsample), we read 8 source slices per batch.
"""

from pathlib import Path
import sys
from typing import Callable, Literal, Optional

import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec
from skimage.transform import downscale_local_mean
from tqdm import tqdm


# Default memory budget: 4GB
DEFAULT_MEMORY_BYTES = 4 * 1024**3

# Supported output dtypes
OutputDtype = Literal["float32", "uint16"]


def parse_memory_string(mem_str: str) -> int:
    """
    Parse human-readable memory string to bytes.

    Args:
        mem_str: Memory string like '4G', '500M', '2.5G', '4096'

    Returns:
        Memory size in bytes

    Examples:
        >>> parse_memory_string('4G')
        4294967296
        >>> parse_memory_string('500M')
        524288000
    """
    mem_str = mem_str.strip().upper()
    # Remove optional 'B' suffix (e.g., '4GB' -> '4G')
    if mem_str.endswith('B') and len(mem_str) > 1 and mem_str[-2] in 'KMGT':
        mem_str = mem_str[:-1]

    multipliers = {
        'K': 1024,
        'M': 1024**2,
        'G': 1024**3,
        'T': 1024**4,
    }

    for suffix, mult in multipliers.items():
        if mem_str.endswith(suffix):
            return int(float(mem_str[:-1]) * mult)

    return int(mem_str)


def downsample_3d(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downsample a 3D volume by a factor using local mean (box filter).

    Args:
        volume: 3D array of shape (Z, Y, X)
        factor: Downsampling factor (applied to all axes)

    Returns:
        Downsampled array of shape (Z//factor, Y//factor, X//factor)
    """
    return downscale_local_mean(volume, (factor, factor, factor))


def compute_padded_shape(
    shape: tuple[int, ...],
    batch_size: int,
    spatial_divisor: int = 1
) -> tuple[int, ...]:
    """
    Compute the shape after padding all dimensions for pyramid alignment.

    Args:
        shape: Original (Z, Y, X) shape
        batch_size: Number of slices per batch (must be power of 2)
        spatial_divisor: Y and X must be divisible by this (2^(num_levels-1))

    Returns:
        Padded shape where:
        - Z is divisible by batch_size
        - Y and X are divisible by spatial_divisor
    """
    z, y, x = shape
    padded_z = ((z + batch_size - 1) // batch_size) * batch_size
    padded_y = ((y + spatial_divisor - 1) // spatial_divisor) * spatial_divisor
    padded_x = ((x + spatial_divisor - 1) // spatial_divisor) * spatial_divisor
    return (padded_z, padded_y, padded_x)


def compute_pyramid_shapes(
    base_shape: tuple[int, ...],
    num_levels: int
) -> list[tuple[int, ...]]:
    """
    Compute shapes for all pyramid levels.

    Args:
        base_shape: Shape of level 0 (Z, Y, X)
        num_levels: Number of pyramid levels

    Returns:
        List of shapes, one per level
    """
    shapes = [base_shape]
    for level in range(1, num_levels):
        prev = shapes[-1]
        # Each level is 2x smaller in all dimensions
        shapes.append((prev[0] // 2, prev[1] // 2, prev[2] // 2))
    return shapes


class StreamingPyramidConverter:
    """
    Memory-efficient BigTIFF to OME-Zarr converter with pyramid generation.

    Uses batch-aligned processing where the batch size is determined by the
    largest downsampling level (2^(num_levels-1) slices per batch).

    Example:
        converter = StreamingPyramidConverter(
            tiff_path="volume.tif",
            zarr_path="volume.zarr",
            num_levels=4,
            chunk_size=(128, 128, 128),
        )
        converter.convert()
    """

    def __init__(
        self,
        tiff_path: str | Path,
        zarr_path: str | Path,
        chunk_size: tuple[int, int, int],
        num_levels: int = 4,
        compression: Literal["blosc-lz4", "blosc-zstd", "none"] = "blosc-lz4",
        max_memory_bytes: int | None = None,
        output_dtype: OutputDtype = "float32",
        value_range: tuple[float, float] | None = None,
    ):
        """
        Initialize the converter.

        Args:
            tiff_path: Path to input BigTIFF file
            zarr_path: Path for output OME-Zarr directory
            chunk_size: Zarr chunk size (Z, Y, X). Recommendations:
                - Visualization-optimized: (128, 128, 128) - isotropic for Neuroglancer
                - Conversion-optimized: (32, 256, 256) - aligned with z-slices
            num_levels: Number of pyramid levels (1=full only, 4=full+2x+4x+8x)
            compression: Compression codec ("blosc-lz4", "blosc-zstd", "none")
            max_memory_bytes: Maximum memory budget in bytes. If None, defaults
                to 4GB with an informational message.
            output_dtype: Output data type. "float32" preserves original values,
                "uint16" normalizes to 0-65535 for 50% storage reduction.
            value_range: (min, max) range for uint16 normalization. If None and
                output_dtype is "uint16", will pre-scan the TIFF to find range.
        """
        # Validate inputs
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if output_dtype not in ("float32", "uint16"):
            raise ValueError(f"output_dtype must be 'float32' or 'uint16', got {output_dtype}")

        self.tiff_path = Path(tiff_path)
        self.zarr_path = Path(zarr_path)
        self.num_levels = num_levels
        self.chunk_size = chunk_size
        self.compression = compression
        self.output_dtype = output_dtype
        self.value_range = value_range

        # Memory budget
        self._using_default_memory = max_memory_bytes is None
        self.max_memory_bytes = max_memory_bytes if max_memory_bytes is not None else DEFAULT_MEMORY_BYTES

        # Batch size is determined by largest downsampling factor
        # For num_levels=4: levels are 1x, 2x, 4x, 8x, so batch_size=8
        self.batch_size = 2 ** (num_levels - 1)

        # Read source metadata (without loading data)
        with tifffile.TiffFile(self.tiff_path) as tif:
            # Get number of z-slices from pages
            n_pages = len(tif.pages)
            if n_pages == 0:
                raise ValueError(f"TIFF file has no pages: {tiff_path}")

            # Get Y, X dimensions from first page
            page_shape = tif.pages[0].shape
            self.source_shape = (n_pages, page_shape[0], page_shape[1])  # (Z, Y, X)
            self.source_dtype = tif.pages[0].dtype

        # Compute spatial divisor for Y/X padding
        self.spatial_divisor = 2 ** (num_levels - 1)

        # Warn if batch_size exceeds source slices
        if self.batch_size > self.source_shape[0]:
            import warnings
            warnings.warn(
                f"Batch size ({self.batch_size}) exceeds number of slices "
                f"({self.source_shape[0]}). Consider using fewer pyramid levels."
            )

        # Compute padded shape and pyramid shapes
        self.padded_shape = compute_padded_shape(
            self.source_shape, self.batch_size, self.spatial_divisor
        )
        self.pyramid_shapes = compute_pyramid_shapes(self.padded_shape, num_levels)

    def _get_compressors(self) -> list | None:
        """Get the configured compressor list for zarr v3."""
        if self.compression == "none":
            return None
        elif self.compression == "blosc-lz4":
            return [BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")]
        elif self.compression == "blosc-zstd":
            return [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]
        else:
            raise ValueError(f"Unknown compression: {self.compression}")

    def _compute_value_range(self, show_progress: bool = True) -> tuple[float, float]:
        """
        Pre-scan the TIFF to find global min/max values.

        Args:
            show_progress: Whether to show tqdm progress bar

        Returns:
            Tuple of (min_value, max_value)
        """
        global_min = float('inf')
        global_max = float('-inf')

        with tifffile.TiffFile(self.tiff_path) as tif:
            pages = tif.pages
            iterator = tqdm(
                pages,
                desc="Pre-scanning for value range",
                unit="slice",
                disable=not show_progress,
            )
            for page in iterator:
                data = page.asarray()
                global_min = min(global_min, float(data.min()))
                global_max = max(global_max, float(data.max()))

        return (global_min, global_max)

    def _convert_to_output_dtype(self, data: np.ndarray) -> np.ndarray:
        """
        Convert float32 data to output dtype.

        Args:
            data: Input array (float32)

        Returns:
            Array converted to output_dtype
        """
        if self.output_dtype == "float32":
            return data

        # uint16: normalize to [0, 65535]
        vmin, vmax = self.value_range
        if vmax == vmin:
            # Avoid division by zero - all values are the same
            return np.zeros(data.shape, dtype=np.uint16)

        # Normalize to [0, 1], then scale to [0, 65535]
        normalized = (data - vmin) / (vmax - vmin)
        scaled = np.clip(normalized, 0, 1) * 65535
        return scaled.astype(np.uint16)

    def _calculate_memory_per_batch(self) -> dict[str, int]:
        """
        Calculate memory requirements per batch in bytes.

        Returns:
            Dictionary with:
            - 'working_memory': Fixed overhead for one batch (all pyramid levels)
            - 'buffer_per_batch': Memory added to buffer per batch (all levels combined)
        """
        bytes_per_element = 4  # float32
        _, padded_y, padded_x = self.padded_shape

        total_batch_memory = 0
        for level in range(self.num_levels):
            level_batch_z = max(1, self.batch_size // (2 ** level))
            level_y = padded_y // (2 ** level)
            level_x = padded_x // (2 ** level)
            level_bytes = level_batch_z * level_y * level_x * bytes_per_element
            total_batch_memory += level_bytes

        return {
            'working_memory': total_batch_memory,  # One batch in memory during processing
            'buffer_per_batch': total_batch_memory,  # Same amount added to buffer
        }

    def _compute_buffer_threshold(self) -> int:
        """
        Compute how many batches to buffer before flushing.

        Uses the memory budget to determine optimal buffering.

        Returns:
            Number of batches to accumulate before flushing (minimum 1).

        Raises:
            ValueError: If memory budget is too small for even one batch.
        """
        mem = self._calculate_memory_per_batch()
        min_required = mem['working_memory'] + mem['buffer_per_batch']

        if self.max_memory_bytes < min_required:
            raise ValueError(
                f"Memory budget ({self.max_memory_bytes / 1e9:.2f}GB) too small. "
                f"Minimum required: {min_required / 1e9:.2f}GB. "
                f"Increase --max-memory or reduce num_levels."
            )

        # Available memory for buffers = total budget - working memory
        available_for_buffers = self.max_memory_bytes - mem['working_memory']

        # How many batches can we buffer?
        batches_that_fit = max(1, available_for_buffers // mem['buffer_per_batch'])

        return batches_that_fit

    def _create_zarr_arrays(self) -> zarr.Group:
        """Create the OME-Zarr structure with all pyramid levels."""
        # Use zarr v3 API - pass path directly to open_group
        root = zarr.open_group(self.zarr_path, mode="w")

        compressors = self._get_compressors()

        # Create array for each pyramid level
        for level, shape in enumerate(self.pyramid_shapes):
            # Adjust chunk size for smaller levels
            level_chunks = tuple(
                min(c, s) for c, s in zip(self.chunk_size, shape)
            )

            # Use create_array with zarr v3 API
            dtype = np.uint16 if self.output_dtype == "uint16" else np.float32
            create_kwargs = {
                "name": str(level),
                "shape": shape,
                "chunks": level_chunks,
                "dtype": dtype,
                "fill_value": 0,
            }
            if compressors is not None:
                create_kwargs["compressors"] = compressors

            root.create_array(**create_kwargs)

        # Write OME-NGFF metadata
        self._write_ome_metadata(root)

        return root

    def _write_ome_metadata(self, root: zarr.Group) -> None:
        """Write OME-NGFF v0.4 compliant metadata."""
        # Build datasets list with coordinate transformations
        datasets = []
        for level in range(self.num_levels):
            scale_factor = 2 ** level
            datasets.append({
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        # Scale factors for Z, Y, X (assuming isotropic voxels)
                        "scale": [float(scale_factor)] * 3
                    }
                ]
            })

        # Build metadata dict
        metadata = {
            "description": "Converted from BigTIFF with streaming pyramid generation",
            "original_shape": list(self.source_shape),
        }

        # Add dtype conversion info for uint16
        if self.output_dtype == "uint16" and self.value_range is not None:
            metadata["dtype_conversion"] = {
                "source_dtype": "float32",
                "output_dtype": "uint16",
                "source_range": list(self.value_range),
                "transform": "linear",
            }

        # OME-NGFF multiscales metadata
        root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": self.tiff_path.stem,
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
            "type": "local_mean",  # Downsampling method used
            "metadata": metadata,
        }]

    def _read_batch(
        self,
        tif: tifffile.TiffFile,
        batch_idx: int
    ) -> np.ndarray:
        """
        Read a batch of slices from the TIFF file.

        Args:
            tif: Open TiffFile object
            batch_idx: Index of the batch (0, 1, 2, ...)

        Returns:
            Array of shape (batch_size, padded_Y, padded_X), zero-padded as needed
        """
        z_start = batch_idx * self.batch_size
        z_end = min(z_start + self.batch_size, self.source_shape[0])

        # Target padded spatial dimensions
        _, padded_y, padded_x = self.padded_shape
        source_y, source_x = self.source_shape[1], self.source_shape[2]

        # Read available slices
        slices = []
        for z in range(z_start, z_end):
            slice_data = tif.pages[z].asarray()
            slices.append(slice_data)

        # Stack into 3D array
        batch = np.stack(slices, axis=0)

        # Pad Z if necessary (last batch may be incomplete)
        if batch.shape[0] < self.batch_size:
            z_pad_size = self.batch_size - batch.shape[0]
            z_padding = np.zeros(
                (z_pad_size, batch.shape[1], batch.shape[2]),
                dtype=batch.dtype
            )
            batch = np.concatenate([batch, z_padding], axis=0)

        # Pad Y and X if necessary
        if padded_y != source_y or padded_x != source_x:
            padded_batch = np.zeros(
                (self.batch_size, padded_y, padded_x),
                dtype=batch.dtype
            )
            padded_batch[:, :source_y, :source_x] = batch
            batch = padded_batch

        return batch.astype(np.float32)

    def _build_pyramid_batch(self, source_batch: np.ndarray) -> list[np.ndarray]:
        """
        Build all pyramid levels from a source batch.

        Args:
            source_batch: Array of shape (batch_size, Y, X)

        Returns:
            List of arrays, one per pyramid level
            Level 0: (batch_size, Y, X)
            Level 1: (batch_size/2, Y/2, X/2)
            Level 2: (batch_size/4, Y/4, X/4)
            ...
        """
        levels = [source_batch]

        for level in range(1, self.num_levels):
            # Downsample from previous level
            downsampled = downsample_3d(levels[-1], factor=2)
            levels.append(downsampled.astype(np.float32))

        return levels

    def _write_batch_direct(
        self,
        root: zarr.Group,
        batch_idx: int,
        pyramid_batch: list[np.ndarray]
    ) -> None:
        """
        Write batch directly to Zarr (immediate writes, more I/O).

        Args:
            root: Zarr group containing pyramid arrays
            batch_idx: Index of the current batch
            pyramid_batch: List of arrays for each pyramid level
        """
        for level, data in enumerate(pyramid_batch):
            level_batch_size = self.batch_size // (2 ** level)
            z_start = batch_idx * level_batch_size
            z_end = z_start + data.shape[0]

            # Clip to actual array bounds (handles padding at edges)
            actual_z_end = min(z_end, self.pyramid_shapes[level][0])
            actual_data = data[:actual_z_end - z_start]

            if actual_data.shape[0] > 0:
                root[str(level)][z_start:actual_z_end] = actual_data

    def convert(
        self,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
        show_progress: bool = True,
    ) -> Path:
        """
        Run the conversion with memory-aware buffering.

        Args:
            progress_callback: Optional callback(current_batch, total_batches, info_dict)
                where info_dict contains:
                - 'buffer_mb': Current estimated buffer memory in MB
                - 'buffer_threshold': Number of batches before flush
                - 'batches_buffered': Current number of batches in buffer
                - 'flushing': True if currently flushing buffers to disk
            show_progress: Whether to show progress bars (for pre-scan)

        Returns:
            Path to the created OME-Zarr directory
        """
        # Pre-scan for value range if needed
        if self.output_dtype == "uint16" and self.value_range is None:
            self.value_range = self._compute_value_range(show_progress=show_progress)

        # Create output structure
        root = self._create_zarr_arrays()

        # Calculate number of batches
        num_batches = (self.source_shape[0] + self.batch_size - 1) // self.batch_size

        # Compute buffer threshold based on memory budget
        buffer_threshold = self._compute_buffer_threshold()

        # Initialize buffers for each level
        buffers: list[list[np.ndarray]] = [[] for _ in range(self.num_levels)]
        buffer_z_starts: list[int] = [0] * self.num_levels
        batches_buffered = 0

        def current_buffer_bytes() -> int:
            """Calculate current buffer memory usage."""
            return sum(
                sum(b.nbytes for b in level_buf)
                for level_buf in buffers
            )

        def flush_all_buffers(batch_idx: int) -> None:
            """Flush all level buffers to disk."""
            nonlocal batches_buffered

            # Notify that we're flushing
            if progress_callback:
                progress_callback(batch_idx, num_batches, {
                    'buffer_mb': current_buffer_bytes() / 1e6,
                    'buffer_threshold': buffer_threshold,
                    'batches_buffered': batches_buffered,
                    'flushing': True,
                })

            for level in range(self.num_levels):
                if not buffers[level]:
                    continue

                buffered_data = np.concatenate(buffers[level], axis=0)
                z_start = buffer_z_starts[level]
                z_end = z_start + buffered_data.shape[0]

                # Clip to actual array bounds
                actual_z_end = min(z_end, self.pyramid_shapes[level][0])
                actual_data = buffered_data[:actual_z_end - z_start]

                if actual_data.shape[0] > 0:
                    # Convert dtype at flush time (keeps processing in float32)
                    converted_data = self._convert_to_output_dtype(actual_data)
                    root[str(level)][z_start:actual_z_end] = converted_data

                # Reset buffer
                buffers[level] = []
                buffer_z_starts[level] = z_end

            batches_buffered = 0

        with tifffile.TiffFile(self.tiff_path) as tif:
            for batch_idx in range(num_batches):
                # Read batch from TIFF
                source_batch = self._read_batch(tif, batch_idx)

                # Build pyramid levels
                pyramid_batch = self._build_pyramid_batch(source_batch)

                # Add to buffers
                for level, data in enumerate(pyramid_batch):
                    buffers[level].append(data)

                batches_buffered += 1

                # Flush when we hit threshold
                if batches_buffered >= buffer_threshold:
                    flush_all_buffers(batch_idx + 1)

                if progress_callback:
                    progress_callback(batch_idx + 1, num_batches, {
                        'buffer_mb': current_buffer_bytes() / 1e6,
                        'buffer_threshold': buffer_threshold,
                        'batches_buffered': batches_buffered,
                        'flushing': False,
                    })

        # Flush remaining data
        flush_all_buffers(num_batches)

        return self.zarr_path

    def estimate_memory_usage(self) -> dict[str, float]:
        """
        Estimate memory usage for the conversion.

        Returns:
            Dictionary with memory estimates in MB:
            - 'memory_budget_mb': Configured memory budget
            - 'working_memory_mb': Memory for processing one batch
            - 'buffer_per_batch_mb': Memory added per buffered batch
            - 'buffer_threshold': Number of batches before flush
            - 'estimated_peak_mb': Expected peak memory usage
        """
        mem = self._calculate_memory_per_batch()
        buffer_threshold = self._compute_buffer_threshold()
        estimated_peak = mem['working_memory'] + (buffer_threshold * mem['buffer_per_batch'])

        return {
            "memory_budget_mb": self.max_memory_bytes / 1e6,
            "working_memory_mb": mem['working_memory'] / 1e6,
            "buffer_per_batch_mb": mem['buffer_per_batch'] / 1e6,
            "buffer_threshold": buffer_threshold,
            "estimated_peak_mb": estimated_peak / 1e6,
            "using_default_budget": self._using_default_memory,
        }


def convert_tiff_to_ome_zarr(
    tiff_path: str | Path,
    zarr_path: str | Path,
    chunk_size: tuple[int, int, int],
    num_levels: int = 4,
    compression: Literal["blosc-lz4", "blosc-zstd", "none"] = "blosc-lz4",
    max_memory: str | int | None = None,
    output_dtype: OutputDtype = "float32",
    value_range: tuple[float, float] | None = None,
    progress: bool = True,
) -> Path:
    """
    Convenience function to convert a BigTIFF to OME-Zarr.

    Args:
        tiff_path: Path to input BigTIFF file
        zarr_path: Path for output OME-Zarr directory
        chunk_size: Zarr chunk size (Z, Y, X). Recommendations:
            - Visualization-optimized: (128, 128, 128) - isotropic for Neuroglancer
            - Conversion-optimized: (32, 256, 256) - aligned with z-slices
        num_levels: Number of pyramid levels
        compression: Compression codec
        max_memory: Memory budget as string ('4G', '500M') or bytes (int).
            If None, defaults to 4GB.
        output_dtype: Output data type. "float32" preserves original values,
            "uint16" normalizes to 0-65535 for 50% storage reduction.
        value_range: (min, max) range for uint16 normalization. If None and
            output_dtype is "uint16", will pre-scan the TIFF to find range.
        progress: Whether to print progress

    Returns:
        Path to the created OME-Zarr directory
    """
    # Parse memory budget
    if isinstance(max_memory, str):
        max_memory_bytes = parse_memory_string(max_memory)
    else:
        max_memory_bytes = max_memory

    converter = StreamingPyramidConverter(
        tiff_path=tiff_path,
        zarr_path=zarr_path,
        chunk_size=chunk_size,
        num_levels=num_levels,
        compression=compression,
        max_memory_bytes=max_memory_bytes,
        output_dtype=output_dtype,
        value_range=value_range,
    )

    # Print info
    if progress:
        mem = converter.estimate_memory_usage()

        # Show default memory message
        if mem['using_default_budget']:
            print(f"Using default memory budget: {mem['memory_budget_mb']:.0f}MB "
                  f"(set --max-memory to customize)", file=sys.stderr)

        print(f"Converting: {tiff_path}")
        print(f"  Source shape: {converter.source_shape}")
        print(f"  Padded shape: {converter.padded_shape}")
        print(f"  Pyramid levels: {num_levels}")
        print(f"  Batch size: {converter.batch_size} slices")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Output dtype: {output_dtype}")
        if output_dtype == "uint16" and value_range is not None:
            print(f"  Value range: [{value_range[0]:.6f}, {value_range[1]:.6f}]")
        print(f"  Memory budget: {mem['memory_budget_mb']:.0f} MB")
        print(f"  Buffer threshold: {mem['buffer_threshold']} batches")
        print(f"  Estimated peak: {mem['estimated_peak_mb']:.0f} MB")
        print()

    # Set up progress bar
    num_batches = (converter.source_shape[0] + converter.batch_size - 1) // converter.batch_size
    pbar = tqdm(
        total=num_batches,
        desc="Converting",
        unit="batch",
        disable=not progress,
    )

    def progress_callback(current: int, total: int, info: dict) -> None:
        buffer_mb = info.get('buffer_mb', 0)
        batches_buffered = info.get('batches_buffered', 0)
        buffer_threshold = info.get('buffer_threshold', 1)
        flushing = info.get('flushing', False)

        if flushing:
            pbar.set_postfix_str(f"Writing {buffer_mb:.0f}MB to disk...")
        else:
            pbar.set_postfix_str(f"Buffer: {batches_buffered}/{buffer_threshold} ({buffer_mb:.0f}MB)")

        # Only update progress on non-flush callbacks to avoid double counting
        if not flushing:
            pbar.n = current
            pbar.refresh()

    result = converter.convert(
        progress_callback=progress_callback if progress else None,
        show_progress=progress,
    )

    pbar.close()

    if progress:
        print("Done!")

    return result
