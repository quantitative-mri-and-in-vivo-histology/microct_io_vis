"""Tests for the streaming pyramid converter."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from src.streaming_converter import (
    StreamingPyramidConverter,
    compute_padded_shape,
    compute_pyramid_shapes,
    convert_tiff_to_ome_zarr,
    downsample_3d,
    parse_memory_string,
)


class TestDownsample3D:
    """Tests for the downsample_3d function."""

    def test_downsample_factor_2(self):
        """Test 2x downsampling with known values."""
        # 4x4x4 volume with constant value
        volume = np.ones((4, 4, 4), dtype=np.float32) * 8.0
        result = downsample_3d(volume, factor=2)

        assert result.shape == (2, 2, 2)
        assert np.allclose(result, 8.0)

    def test_downsample_averaging(self):
        """Test that downsampling correctly averages values."""
        # 2x2x2 volume with different values
        volume = np.array([
            [[0, 2], [4, 6]],
            [[8, 10], [12, 14]]
        ], dtype=np.float32)

        result = downsample_3d(volume, factor=2)

        # Should be a single voxel with mean of all values
        expected_mean = volume.mean()
        assert result.shape == (1, 1, 1)
        assert np.isclose(result[0, 0, 0], expected_mean)

    def test_downsample_preserves_dtype(self):
        """Test that downsampling returns float32 for float32 input."""
        volume = np.random.rand(8, 8, 8).astype(np.float32)
        result = downsample_3d(volume, factor=2)

        # scikit-image preserves float32 input dtype
        assert result.dtype == np.float32


class TestComputeShapes:
    """Tests for shape computation functions."""

    def test_padded_shape_exact(self):
        """Test when shape is already aligned with batch size."""
        shape = (16, 512, 512)
        result = compute_padded_shape(shape, batch_size=8)
        assert result == (16, 512, 512)

    def test_padded_shape_needs_padding(self):
        """Test when shape needs padding."""
        shape = (17, 512, 512)
        result = compute_padded_shape(shape, batch_size=8)
        assert result == (24, 512, 512)

    def test_padded_shape_single_slice(self):
        """Test edge case with fewer slices than batch size."""
        shape = (3, 512, 512)
        result = compute_padded_shape(shape, batch_size=8)
        assert result == (8, 512, 512)

    def test_pyramid_shapes(self):
        """Test pyramid shape computation."""
        base_shape = (64, 256, 256)
        shapes = compute_pyramid_shapes(base_shape, num_levels=4)

        assert len(shapes) == 4
        assert shapes[0] == (64, 256, 256)   # Level 0: 1x
        assert shapes[1] == (32, 128, 128)   # Level 1: 2x
        assert shapes[2] == (16, 64, 64)     # Level 2: 4x
        assert shapes[3] == (8, 32, 32)      # Level 3: 8x


class TestStreamingPyramidConverter:
    """Tests for the StreamingPyramidConverter class."""

    @pytest.fixture
    def small_tiff(self, tmp_path):
        """Create a small test TIFF file."""
        tiff_path = tmp_path / "test.tif"

        # Create 16 slices of 64x64 with gradient values
        n_slices = 16
        height, width = 64, 64

        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(n_slices):
                # Each slice has value = z to verify ordering
                slice_data = np.full((height, width), z, dtype=np.float32)
                tif.write(slice_data)

        return tiff_path

    @pytest.fixture
    def unaligned_tiff(self, tmp_path):
        """Create a TIFF with non-power-of-2 slices."""
        tiff_path = tmp_path / "unaligned.tif"

        # 11 slices (not divisible by 8)
        n_slices = 11
        height, width = 32, 32

        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(n_slices):
                slice_data = np.full((height, width), z, dtype=np.float32)
                tif.write(slice_data)

        return tiff_path

    def test_converter_initialization(self, small_tiff, tmp_path):
        """Test converter initializes correctly."""
        zarr_path = tmp_path / "output.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=small_tiff,
            zarr_path=zarr_path,
            num_levels=3,
            chunk_size=(8, 32, 32),
        )

        assert converter.source_shape == (16, 64, 64)
        assert converter.batch_size == 4  # 2^(3-1) = 4
        assert converter.num_levels == 3

    def test_convert_small_memory(self, small_tiff, tmp_path):
        """Test conversion with small memory budget (frequent flushing)."""
        zarr_path = tmp_path / "output_small_mem.zarr"

        # Use small memory to force frequent flushing (like old direct strategy)
        result = convert_tiff_to_ome_zarr(
            tiff_path=small_tiff,
            zarr_path=zarr_path,
            num_levels=3,
            chunk_size=(8, 32, 32),
            max_memory="1M",  # Very small, forces buffer_threshold=1
            progress=False,
        )

        assert result.exists()

        # Verify structure
        root = zarr.open(str(zarr_path), mode="r")
        assert "0" in root
        assert "1" in root
        assert "2" in root

        # Check shapes
        assert root["0"].shape == (16, 64, 64)
        assert root["1"].shape == (8, 32, 32)
        assert root["2"].shape == (4, 16, 16)

    def test_convert_large_memory(self, small_tiff, tmp_path):
        """Test conversion with large memory budget (buffered writes)."""
        zarr_path = tmp_path / "output_large_mem.zarr"

        result = convert_tiff_to_ome_zarr(
            tiff_path=small_tiff,
            zarr_path=zarr_path,
            num_levels=3,
            chunk_size=(8, 32, 32),
            max_memory="1G",  # Plenty of memory for buffering
            progress=False,
        )

        assert result.exists()

        # Verify data integrity by checking level 0 values
        root = zarr.open(str(zarr_path), mode="r")

        # Each original slice had value = z
        for z in range(16):
            slice_data = root["0"][z]
            assert np.allclose(slice_data, z), f"Slice {z} has wrong values"

    def test_convert_unaligned_slices(self, unaligned_tiff, tmp_path):
        """Test conversion handles non-power-of-2 slice counts."""
        zarr_path = tmp_path / "output_unaligned.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=unaligned_tiff,
            zarr_path=zarr_path,
            num_levels=3,  # batch_size = 4
            chunk_size=(4, 16, 16),
        )

        # Should pad 11 slices to 12 (next multiple of 4)
        assert converter.padded_shape == (12, 32, 32)
        assert converter.pyramid_shapes[0] == (12, 32, 32)
        assert converter.pyramid_shapes[1] == (6, 16, 16)
        assert converter.pyramid_shapes[2] == (3, 8, 8)

        converter.convert()

        root = zarr.open(str(zarr_path), mode="r")
        assert root["0"].shape == (12, 32, 32)

    def test_ome_metadata(self, small_tiff, tmp_path):
        """Test that OME-NGFF metadata is written correctly."""
        zarr_path = tmp_path / "output_meta.zarr"

        convert_tiff_to_ome_zarr(
            tiff_path=small_tiff,
            zarr_path=zarr_path,
            chunk_size=(8, 32, 32),
            num_levels=3,
            progress=False,
        )

        root = zarr.open(str(zarr_path), mode="r")

        # Check multiscales metadata
        assert "multiscales" in root.attrs
        multiscales = root.attrs["multiscales"]
        assert len(multiscales) == 1

        ms = multiscales[0]
        assert ms["version"] == "0.4"
        assert len(ms["axes"]) == 3
        assert ms["axes"][0]["name"] == "z"
        assert ms["axes"][1]["name"] == "y"
        assert ms["axes"][2]["name"] == "x"

        # Check datasets
        assert len(ms["datasets"]) == 3
        assert ms["datasets"][0]["path"] == "0"
        assert ms["datasets"][1]["path"] == "1"
        assert ms["datasets"][2]["path"] == "2"

        # Check scale factors
        assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]
        assert ms["datasets"][1]["coordinateTransformations"][0]["scale"] == [2.0, 2.0, 2.0]
        assert ms["datasets"][2]["coordinateTransformations"][0]["scale"] == [4.0, 4.0, 4.0]

    def test_memory_estimation(self, small_tiff, tmp_path):
        """Test memory usage estimation."""
        zarr_path = tmp_path / "output.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=small_tiff,
            zarr_path=zarr_path,
            chunk_size=(8, 32, 32),
            num_levels=3,
        )

        mem = converter.estimate_memory_usage()

        # Check new memory estimation fields
        assert "memory_budget_mb" in mem
        assert "working_memory_mb" in mem
        assert "buffer_per_batch_mb" in mem
        assert "buffer_threshold" in mem
        assert "estimated_peak_mb" in mem
        assert "using_default_budget" in mem

        # Should use default budget when not specified
        assert mem["using_default_budget"] is True
        # 4GB = 4 * 1024^3 bytes / 1e6 = ~4295 MB
        assert abs(mem["memory_budget_mb"] - 4 * 1024**3 / 1e6) < 1

        # Buffer threshold should be at least 1
        assert mem["buffer_threshold"] >= 1

        # Estimated peak should be reasonable
        assert mem["estimated_peak_mb"] > 0
        assert mem["estimated_peak_mb"] <= mem["memory_budget_mb"]

    def test_memory_budget_affects_threshold(self, small_tiff, tmp_path):
        """Test that memory budget affects buffer threshold."""
        zarr_path = tmp_path / "output.zarr"

        # Large memory budget
        converter_large = StreamingPyramidConverter(
            tiff_path=small_tiff,
            zarr_path=zarr_path,
            chunk_size=(8, 32, 32),
            num_levels=3,
            max_memory_bytes=1 * 1024**3,  # 1GB
        )
        mem_large = converter_large.estimate_memory_usage()

        # Small memory budget
        converter_small = StreamingPyramidConverter(
            tiff_path=small_tiff,
            zarr_path=zarr_path,
            chunk_size=(8, 32, 32),
            num_levels=3,
            max_memory_bytes=10 * 1024**2,  # 10MB
        )
        mem_small = converter_small.estimate_memory_usage()

        # Large budget should allow more batches to be buffered
        assert mem_large["buffer_threshold"] > mem_small["buffer_threshold"]

    def test_compression_options(self, small_tiff, tmp_path):
        """Test different compression options."""
        for compression in ["blosc-lz4", "blosc-zstd", "none"]:
            zarr_path = tmp_path / f"output_{compression}.zarr"

            convert_tiff_to_ome_zarr(
                tiff_path=small_tiff,
                zarr_path=zarr_path,
                chunk_size=(8, 32, 32),
                num_levels=2,
                compression=compression,
                progress=False,
            )

            root = zarr.open(str(zarr_path), mode="r")
            assert root["0"].shape == (16, 64, 64)


class TestValidation:
    """Tests for input validation."""

    def test_num_levels_must_be_positive(self, tmp_path):
        """Test that num_levels < 1 raises ValueError."""
        tiff_path = tmp_path / "test.tif"

        # Create minimal TIFF
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            tif.write(np.zeros((64, 64), dtype=np.float32))

        zarr_path = tmp_path / "output.zarr"

        with pytest.raises(ValueError, match="num_levels must be >= 1"):
            StreamingPyramidConverter(
                tiff_path=tiff_path,
                zarr_path=zarr_path,
                chunk_size=(8, 32, 32),
                num_levels=0,
            )

    def test_empty_tiff_raises_error(self, tmp_path):
        """Test that empty TIFF raises ValueError."""
        tiff_path = tmp_path / "empty.tif"

        # Create empty TIFF (no pages)
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            pass  # No pages written

        zarr_path = tmp_path / "output.zarr"

        with pytest.raises(ValueError, match="has no pages"):
            StreamingPyramidConverter(
                tiff_path=tiff_path,
                zarr_path=zarr_path,
                chunk_size=(8, 32, 32),
                num_levels=2,
            )

    def test_non_divisible_dimensions_get_padded(self, tmp_path):
        """Test that Y/X dimensions not divisible by downscale factor are padded."""
        tiff_path = tmp_path / "odd.tif"

        # Create TIFF with 65x65 dimensions (not divisible by 4 for num_levels=3)
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(8):
                tif.write(np.ones((65, 65), dtype=np.float32) * z)

        zarr_path = tmp_path / "output.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=tiff_path,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=3,  # requires divisibility by 4
        )

        # Should pad 65 to 68 (next multiple of 4)
        assert converter.source_shape == (8, 65, 65)
        assert converter.padded_shape == (8, 68, 68)

        # Verify conversion works
        converter.convert()

        root = zarr.open(str(zarr_path), mode="r")
        assert root["0"].shape == (8, 68, 68)
        assert root["1"].shape == (4, 34, 34)
        assert root["2"].shape == (2, 17, 17)

        # Original data should be preserved in top-left corner
        assert np.allclose(root["0"][0, :65, :65], 0.0)
        assert np.allclose(root["0"][1, :65, :65], 1.0)

        # Padded region should be zeros
        assert np.allclose(root["0"][0, 65:, :], 0.0)
        assert np.allclose(root["0"][0, :, 65:], 0.0)

    def test_batch_size_warning(self, tmp_path):
        """Test that warning is issued when batch_size > num_slices."""
        tiff_path = tmp_path / "small.tif"

        # Create TIFF with only 2 slices
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(2):
                tif.write(np.zeros((64, 64), dtype=np.float32))

        zarr_path = tmp_path / "output.zarr"

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StreamingPyramidConverter(
                tiff_path=tiff_path,
                zarr_path=zarr_path,
                chunk_size=(4, 32, 32),
                num_levels=4,  # batch_size = 8 > 2 slices
            )
            assert len(w) == 1
            assert "exceeds number of slices" in str(w[0].message)

    def test_memory_budget_too_small(self, tmp_path):
        """Test that too-small memory budget raises informative error."""
        tiff_path = tmp_path / "test.tif"

        # Create a larger TIFF to require more memory
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(8):
                tif.write(np.zeros((256, 256), dtype=np.float32))

        zarr_path = tmp_path / "output.zarr"

        # Memory budget way too small for this volume
        converter = StreamingPyramidConverter(
            tiff_path=tiff_path,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=3,
            max_memory_bytes=1000,  # Only 1KB - way too small
        )

        # Error is raised when trying to convert (which calls _compute_buffer_threshold)
        with pytest.raises(ValueError, match="Memory budget.*too small"):
            converter.convert()


class TestParseMemoryString:
    """Tests for parse_memory_string function."""

    def test_parse_gigabytes(self):
        """Test parsing gigabyte values."""
        assert parse_memory_string("4G") == 4 * 1024**3
        assert parse_memory_string("4GB") == 4 * 1024**3
        assert parse_memory_string("2.5G") == int(2.5 * 1024**3)

    def test_parse_megabytes(self):
        """Test parsing megabyte values."""
        assert parse_memory_string("500M") == 500 * 1024**2
        assert parse_memory_string("500MB") == 500 * 1024**2
        assert parse_memory_string("1024M") == 1024 * 1024**2

    def test_parse_kilobytes(self):
        """Test parsing kilobyte values."""
        assert parse_memory_string("1024K") == 1024 * 1024
        assert parse_memory_string("1024KB") == 1024 * 1024

    def test_parse_terabytes(self):
        """Test parsing terabyte values."""
        assert parse_memory_string("1T") == 1 * 1024**4
        assert parse_memory_string("1TB") == 1 * 1024**4

    def test_parse_plain_bytes(self):
        """Test parsing plain byte values."""
        assert parse_memory_string("4294967296") == 4294967296

    def test_parse_case_insensitive(self):
        """Test case insensitivity."""
        assert parse_memory_string("4g") == parse_memory_string("4G")
        assert parse_memory_string("500m") == parse_memory_string("500M")
        assert parse_memory_string("4gb") == parse_memory_string("4GB")

    def test_parse_with_whitespace(self):
        """Test handling of whitespace."""
        assert parse_memory_string("  4G  ") == 4 * 1024**3


class TestDownsamplingAccuracy:
    """Tests to verify downsampling produces correct results."""

    @pytest.fixture
    def gradient_tiff(self, tmp_path):
        """Create a TIFF with known gradient for verification."""
        tiff_path = tmp_path / "gradient.tif"

        # 8 slices of 16x16 with gradient
        n_slices = 8
        size = 16

        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(n_slices):
                # Create gradient: value = z * 100 + y * 10 + x
                y_coords, x_coords = np.mgrid[0:size, 0:size]
                slice_data = (z * 100 + y_coords * 10 + x_coords).astype(np.float32)
                tif.write(slice_data)

        return tiff_path

    def test_downsampling_preserves_mean(self, gradient_tiff, tmp_path):
        """Test that downsampling preserves overall mean."""
        zarr_path = tmp_path / "gradient_out.zarr"

        convert_tiff_to_ome_zarr(
            tiff_path=gradient_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 8, 8),
            num_levels=2,
            progress=False,
        )

        root = zarr.open(str(zarr_path), mode="r")

        # Mean should be approximately preserved across levels
        mean_0 = root["0"][:].mean()
        mean_1 = root["1"][:].mean()

        # Allow some tolerance due to edge effects
        assert abs(mean_0 - mean_1) / mean_0 < 0.01


class TestSpatialPadding:
    """Tests for Y/X spatial padding functionality."""

    def test_compute_padded_shape_with_spatial_divisor(self):
        """Test compute_padded_shape with spatial_divisor parameter."""
        # Z needs padding, Y/X already aligned
        shape = (17, 64, 64)
        result = compute_padded_shape(shape, batch_size=8, spatial_divisor=16)
        assert result == (24, 64, 64)

        # Y/X need padding
        shape = (16, 65, 70)
        result = compute_padded_shape(shape, batch_size=8, spatial_divisor=16)
        assert result == (16, 80, 80)

        # All dimensions need padding
        shape = (11, 33, 47)
        result = compute_padded_shape(shape, batch_size=8, spatial_divisor=8)
        assert result == (16, 40, 48)

    def test_compute_padded_shape_no_spatial_padding_needed(self):
        """Test when spatial dimensions are already aligned."""
        shape = (16, 128, 256)
        result = compute_padded_shape(shape, batch_size=8, spatial_divisor=64)
        assert result == (16, 128, 256)

    @pytest.fixture
    def asymmetric_tiff(self, tmp_path):
        """Create a TIFF with asymmetric non-divisible dimensions."""
        tiff_path = tmp_path / "asymmetric.tif"

        # 10 slices of 33x47 (both not divisible by 8)
        n_slices = 10
        height, width = 33, 47

        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(n_slices):
                # Use distinct values to verify data placement
                slice_data = np.full((height, width), z * 10.0, dtype=np.float32)
                # Add marker in corner
                slice_data[0, 0] = z * 100.0
                slice_data[height-1, width-1] = z * 1000.0
                tif.write(slice_data)

        return tiff_path

    def test_asymmetric_padding_small_memory(self, asymmetric_tiff, tmp_path):
        """Test conversion with asymmetric Y/X padding using small memory budget."""
        zarr_path = tmp_path / "asymmetric_small_mem.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=asymmetric_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 16, 16),
            num_levels=4,  # spatial_divisor = 8
            max_memory_bytes=1 * 1024 * 1024,  # 1MB - forces frequent flushing
        )

        # 33 -> 40 (next multiple of 8), 47 -> 48
        assert converter.source_shape == (10, 33, 47)
        assert converter.padded_shape == (16, 40, 48)  # Z: 10->16, Y: 33->40, X: 47->48

        converter.convert()

        root = zarr.open(str(zarr_path), mode="r")

        # Check all pyramid levels exist with correct shapes
        assert root["0"].shape == (16, 40, 48)
        assert root["1"].shape == (8, 20, 24)
        assert root["2"].shape == (4, 10, 12)
        assert root["3"].shape == (2, 5, 6)

        # Verify original data is preserved
        for z in range(10):
            assert root["0"][z, 0, 0] == z * 100.0, f"Corner marker wrong at z={z}"
            assert root["0"][z, 32, 46] == z * 1000.0, f"Far corner marker wrong at z={z}"

        # Verify padded regions are zeros
        assert np.allclose(root["0"][0, 33:, :], 0.0)  # Y padding
        assert np.allclose(root["0"][0, :, 47:], 0.0)  # X padding
        assert np.allclose(root["0"][10:, :, :], 0.0)  # Z padding

    def test_asymmetric_padding_large_memory(self, asymmetric_tiff, tmp_path):
        """Test conversion with asymmetric Y/X padding using large memory budget."""
        zarr_path = tmp_path / "asymmetric_large_mem.zarr"

        result = convert_tiff_to_ome_zarr(
            tiff_path=asymmetric_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 16, 16),
            num_levels=4,
            max_memory="1G",  # Plenty of memory for buffering
            progress=False,
        )

        assert result.exists()

        root = zarr.open(str(zarr_path), mode="r")

        # Same shape checks
        assert root["0"].shape == (16, 40, 48)

        # Verify data integrity at corners
        assert root["0"][5, 0, 0] == 500.0
        assert root["0"][9, 32, 46] == 9000.0

    def test_large_spatial_padding(self, tmp_path):
        """Test with many pyramid levels requiring significant spatial padding."""
        tiff_path = tmp_path / "small_xy.tif"

        # 4 slices of 17x17 with 5 levels (divisor=16)
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for z in range(4):
                data = np.ones((17, 17), dtype=np.float32) * (z + 1)
                tif.write(data)

        zarr_path = tmp_path / "small_xy.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=tiff_path,
            zarr_path=zarr_path,
            chunk_size=(4, 8, 8),
            num_levels=5,  # spatial_divisor = 16
        )

        # 17 -> 32 (next multiple of 16)
        assert converter.padded_shape == (16, 32, 32)

        converter.convert()

        root = zarr.open(str(zarr_path), mode="r")
        assert root["0"].shape == (16, 32, 32)
        assert root["4"].shape == (1, 2, 2)

        # Data in original region
        assert np.allclose(root["0"][0, :17, :17], 1.0)
        assert np.allclose(root["0"][3, :17, :17], 4.0)

        # Zeros in padded regions
        assert np.allclose(root["0"][0, 17:, :], 0.0)
        assert np.allclose(root["0"][0, :, 17:], 0.0)

    def test_metadata_contains_original_shape(self, asymmetric_tiff, tmp_path):
        """Test that original shape is stored in OME metadata."""
        zarr_path = tmp_path / "meta_check.zarr"

        convert_tiff_to_ome_zarr(
            tiff_path=asymmetric_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 16, 16),
            num_levels=3,
            progress=False,
        )

        root = zarr.open(str(zarr_path), mode="r")
        multiscales = root.attrs["multiscales"][0]

        # Original shape should be recorded
        assert multiscales["metadata"]["original_shape"] == [10, 33, 47]


class TestDtypeConversion:
    """Tests for output dtype conversion (uint16)."""

    @pytest.fixture
    def float32_tiff(self, tmp_path):
        """Create a TIFF with float32 values in a known range."""
        tiff_path = tmp_path / "float32_data.tif"

        # Create data with known range: [-0.5, 1.5]
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            for _ in range(4):
                # Linearly varying values from -0.5 to 1.5 across the slice
                data = np.linspace(-0.5, 1.5, 64 * 64, dtype=np.float32).reshape(64, 64)
                tif.write(data)

        return tiff_path

    def test_uint16_conversion_and_normalization(self, float32_tiff, tmp_path):
        """Test conversion to uint16 with correct normalization."""
        zarr_path = tmp_path / "uint16_output.zarr"

        convert_tiff_to_ome_zarr(
            tiff_path=float32_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=2,
            output_dtype="uint16",
            value_range=(-0.5, 1.5),
            progress=False,
        )

        root = zarr.open(str(zarr_path), mode="r")

        # Check dtype is uint16
        assert root["0"].dtype == np.uint16
        assert root["1"].dtype == np.uint16

        # Check value range is [0, 65535]
        data = root["0"][:]
        assert data.min() >= 0
        assert data.max() <= 65535

        # Verify normalization: min value (-0.5) should map to 0
        # max value (1.5) should map to 65535
        assert data.min() == 0
        assert data.max() == 65535

    def test_explicit_value_range_skips_prescan(self, float32_tiff, tmp_path):
        """Test that providing value_range skips the pre-scan."""
        zarr_path = tmp_path / "no_prescan.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=float32_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=2,
            output_dtype="uint16",
            value_range=(-1.0, 2.0),  # Explicit range
        )

        # value_range should be set from init
        assert converter.value_range == (-1.0, 2.0)

        # Run conversion
        converter.convert(show_progress=False)

        # Should use the provided range, not auto-detected
        root = zarr.open(str(zarr_path), mode="r")
        multiscales = root.attrs["multiscales"][0]
        assert multiscales["metadata"]["dtype_conversion"]["source_range"] == [-1.0, 2.0]

    def test_metadata_contains_conversion_info(self, float32_tiff, tmp_path):
        """Test that dtype conversion metadata is stored correctly."""
        zarr_path = tmp_path / "meta_dtype.zarr"

        convert_tiff_to_ome_zarr(
            tiff_path=float32_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=2,
            output_dtype="uint16",
            value_range=(-0.5, 1.5),
            progress=False,
        )

        root = zarr.open(str(zarr_path), mode="r")
        multiscales = root.attrs["multiscales"][0]

        # Check conversion metadata
        conv = multiscales["metadata"]["dtype_conversion"]
        assert conv["source_dtype"] == "float32"
        assert conv["output_dtype"] == "uint16"
        assert conv["source_range"] == [-0.5, 1.5]
        assert conv["transform"] == "linear"

    def test_float32_default_unchanged(self, float32_tiff, tmp_path):
        """Test that float32 output (default) preserves original values."""
        zarr_path = tmp_path / "float32_output.zarr"

        convert_tiff_to_ome_zarr(
            tiff_path=float32_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=2,
            output_dtype="float32",  # Default
            progress=False,
        )

        root = zarr.open(str(zarr_path), mode="r")

        # Check dtype is float32
        assert root["0"].dtype == np.float32

        # Check values are preserved (not normalized)
        data = root["0"][:]
        assert np.isclose(data.min(), -0.5, atol=1e-5)
        assert np.isclose(data.max(), 1.5, atol=1e-5)

        # No dtype_conversion metadata for float32
        multiscales = root.attrs["multiscales"][0]
        assert "dtype_conversion" not in multiscales["metadata"]

    def test_uint16_auto_range_detection(self, float32_tiff, tmp_path):
        """Test automatic value range detection when not provided."""
        zarr_path = tmp_path / "auto_range.zarr"

        converter = StreamingPyramidConverter(
            tiff_path=float32_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=2,
            output_dtype="uint16",
            value_range=None,  # Should trigger pre-scan
        )

        # value_range is None before convert()
        assert converter.value_range is None

        # Run conversion (triggers pre-scan)
        converter.convert(show_progress=False)

        # value_range should be auto-detected
        assert converter.value_range is not None
        assert np.isclose(converter.value_range[0], -0.5, atol=1e-5)
        assert np.isclose(converter.value_range[1], 1.5, atol=1e-5)

    def test_invalid_output_dtype_raises(self, tmp_path):
        """Test that invalid output_dtype raises ValueError."""
        tiff_path = tmp_path / "dummy.tif"
        with tifffile.TiffWriter(tiff_path, bigtiff=True) as tif:
            tif.write(np.zeros((32, 32), dtype=np.float32))

        zarr_path = tmp_path / "output.zarr"

        with pytest.raises(ValueError, match="output_dtype must be"):
            StreamingPyramidConverter(
                tiff_path=tiff_path,
                zarr_path=zarr_path,
                chunk_size=(4, 32, 32),
                num_levels=2,
                output_dtype="int8",  # Invalid
            )

    def test_value_recovery_formula(self, float32_tiff, tmp_path):
        """Test that original values can be recovered from uint16 using metadata."""
        zarr_path = tmp_path / "recoverable.zarr"

        value_range = (-0.5, 1.5)

        convert_tiff_to_ome_zarr(
            tiff_path=float32_tiff,
            zarr_path=zarr_path,
            chunk_size=(4, 32, 32),
            num_levels=2,
            output_dtype="uint16",
            value_range=value_range,
            progress=False,
        )

        root = zarr.open(str(zarr_path), mode="r")
        uint16_data = root["0"][:]

        # Recover original values using the formula:
        # original = source_range[0] + (uint16_value / 65535) * (source_range[1] - source_range[0])
        vmin, vmax = value_range
        recovered = vmin + (uint16_data.astype(np.float64) / 65535) * (vmax - vmin)

        # Read original data
        with tifffile.TiffFile(float32_tiff) as tif:
            original_slice = tif.pages[0].asarray()

        # Compare recovered values to original (allowing for quantization error)
        # uint16 has 65536 levels, so max error is (vmax-vmin)/65535/2
        max_quantization_error = (vmax - vmin) / 65535 / 2
        assert np.allclose(recovered[0, :64, :64], original_slice, atol=max_quantization_error * 2)
