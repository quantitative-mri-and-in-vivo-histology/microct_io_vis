# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository focuses on finding an optimal data structure for very large micro-CT imaging data (grayscale float32 volumes). The primary goals are:

1. **Data format conversion**: Convert BigTIFF stacked pages (~56GB files) to a memory-efficient format
2. **Multi-resolution pyramids**: Enable visualization on memory-constrained systems (32GB laptops) via Neuroglancer
3. **Streaming processing**: Handle volumes that cannot fit in RAM

## Source Data Characteristics

- **Format**: BigTIFF with separate pages (one page per z-slice)
- **Size**: ~56GB per volume (e.g., `data/raw/tomo_reco_id0004_t0008.tif`)
- **Dtype**: float32 grayscale
- **Structure**: 3D volumes stored as 2D slice stacks

## Target Format: OME-Zarr (NGFF v0.4)

**Why OME-Zarr over alternatives:**
- **Neuroglancer native support** - No conversion needed for visualization
- **Broad ecosystem** - napari, FIJI, ITK, dask, xarray all support it
- **Cloud-ready** - Works on S3/GCS or local filesystem identically
- **Built-in pyramid spec** - Multi-resolution is part of NGFF standard
- **Python-native** - zarr library with chunked-by-design architecture

**Alternatives considered:**
- *Neuroglancer Precomputed*: Faster in Neuroglancer but ecosystem-locked
- *N5*: Java-centric (BigDataViewer), less Python support
- *HDF5*: Broad support but no standard pyramid spec

### OME-Zarr Directory Structure
```
volume.zarr/
├── .zattrs          # OME-NGFF metadata (multiscales, axes)
├── .zgroup
├── 0/               # Full resolution
│   └── .zarray      # Chunked array (e.g., 128x128x128 chunks)
├── 1/               # 2x downsampled
├── 2/               # 4x downsampled
└── ...
```

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run single test
pytest path/to/test_file.py::test_name -v

# Inspect TIFF file headers (without loading data)
python -c "import tifffile; print(tifffile.TiffFile('path/to/file.tif').pages[0].shape)"
```

## Key Libraries

- **tifffile**: Reading BigTIFF files with memory-mapped access
- **zarr**: Chunked array storage with compression
- **ome-zarr**: OME-NGFF metadata and pyramid utilities
- **numpy**: Array operations
- **neuroglancer**: Visualization of multi-resolution volumes

## Memory-Efficient Processing Patterns

When working with these large volumes:

1. **Never load full volumes into RAM** - Use streaming/chunked access
2. **Use tifffile's memory mapping** for reading: `tif.pages[z].asarray()` loads one slice
3. **Configure Zarr chunks** to match access patterns (typically z-slice oriented for conversion, isotropic for visualization)
4. **Process slice-by-slice** rather than full volume operations

## Streaming Pyramid Construction Strategy

Building multi-resolution pyramids without loading the full volume into memory.

### Algorithm: Batch-Aligned Processing

The largest downsampling level determines the batch size. For `num_levels=4` (8x max downsample), process 8 source slices per batch:

```
batch_size = 2^(num_levels - 1)  # e.g., 8 for 4 levels

For each batch of batch_size slices:
  level_0_data = read_batch_from_tiff()           # (8, H, W)
  level_1_data = downsample_3d(level_0_data, 2)   # (4, H/2, W/2)
  level_2_data = downsample_3d(level_1_data, 2)   # (2, H/4, W/4)
  level_3_data = downsample_3d(level_2_data, 2)   # (1, H/8, W/8)

  Write all levels to zarr
```

**Key insight**: Each pyramid level is built from the previous level (not re-reading source), keeping memory bounded regardless of pyramid depth.

### Memory Budget (2560×2560 float32, 4 levels)
```
Source batch:  8 × 2560 × 2560 × 4 = 210 MB
Level 1:       4 × 1280 × 1280 × 4 =  26 MB
Level 2:       2 × 640 × 640 × 4   =   3 MB
Level 3:       1 × 320 × 320 × 4   = 0.4 MB
─────────────────────────────────────────────
Total per batch:                    ~240 MB
```

With buffered writes (128 z-chunk), multiply by ~16 batches = ~3.8 GB peak memory.

### Downsampling Method
Uses **local mean** (box filter) via `skimage.transform.downscale_local_mean`:
- Applies 2x downsampling to all three dimensions simultaneously
- Preserves mean intensity across pyramid levels

### Chunk Size Options

The user must specify chunk size. Recommendations:

| Use Case | Chunk Size | Trade-off |
|----------|------------|-----------|
| **Visualization-optimized** | `(128, 128, 128)` | Isotropic for Neuroglancer arbitrary slicing |
| **Conversion-optimized** | `(32, 256, 256)` | Fewer partial chunk writes during conversion |

### Memory-Aware Buffering

The converter uses a `--max-memory` parameter to intelligently manage write buffering:

```bash
# Default: 4GB memory budget
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128

# Custom memory budget
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128 --max-memory 8G

# Smaller budget for memory-constrained systems
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128 --max-memory 2G
```

**How it works:**
- Calculates memory needed per batch (working memory + buffer)
- Determines how many batches can be buffered within the budget
- All pyramid levels flush together when threshold is reached
- Larger budgets = more buffering = fewer I/O operations = faster conversion
- Smaller budgets = frequent flushing = lower memory usage

**Memory format:** Accepts human-readable strings like `4G`, `500M`, `2GB`, `4096` (bytes)

### Padding Behavior

Z dimension is padded to align with batch_size (filled with zeros). Original shape is stored in OME-NGFF metadata under `metadata.original_shape`.

### Validation Requirements

- `num_levels >= 1`
- Y and X dimensions must be divisible by `2^(num_levels-1)`
- Warning issued if batch_size exceeds number of source slices

## Devcontainer Configuration

The project uses a devcontainer with 70GB shared memory (`--shm-size=70g`) to handle large temporary arrays. PYTHONPATH is set to `/workspace`.

## Relevant Skills

The `3d-morphometry-analyzer` skill provides detailed patterns for:
- Streaming HDF5 data processing
- Parallel slice-wise operations
- Histogram-based memory-efficient aggregation
- Axis permutations without full loading
