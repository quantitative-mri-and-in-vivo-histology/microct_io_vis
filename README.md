# microct_io_vis

Convert large micro-CT volumes (BigTIFF) to OME-Zarr with multi-resolution pyramids for visualization in Neuroglancer.

## Features

- **Streaming conversion**: Process 56GB+ volumes without loading into RAM
- **Multi-resolution pyramids**: Automatic 2x downsampling at each level
- **Memory-aware buffering**: Configurable memory budget for I/O optimization
- **Neuroglancer visualization**: Built-in viewer with HTTP pyramid serving

## Installation

```bash
pip install -r requirements.txt
```

Or open the devcontainer in VS Code for a ready-to-use environment.

## Usage

### Convert BigTIFF to OME-Zarr

The converter streams 2D slices from BigTIFF and writes them as chunked 3D arrays in OME-Zarr format. It simultaneously builds a multi-resolution pyramid by progressively downsampling, enabling efficient visualization at any zoom level.

![Conversion pipeline](docs/images/conversion_pipeline.png)

**Key options:**
- `--max-memory`: Memory budget for streaming. Lower values flush to disk more frequently, reducing RAM usage but increasing conversion time.
- `--compression`: Compression algorithm (`blosc-zstd`, `blosc-lz4`, `none`). Saves disk space at the cost of slightly slower viewer performance.
- `--dtype uint16`: Halves output size by converting from float32, but reduces dynamic range.

```bash
# Basic conversion with isotropic chunks (best for visualization)
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128

# Custom memory budget and pyramid levels
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128 -n 4 --max-memory 8G

# Inspect TIFF without converting
python examples/convert_tiff_to_zarr.py input.tif --info

# Estimate memory usage before conversion
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128 --estimate

# Convert to uint16 (50% smaller output, good for visualization)
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128 --dtype uint16
```

### Segment with Cellpose

`segment_zarr.py` runs a 3D Cellpose model on the volume (or a sub-ROI) and writes the labels as an OME-Zarr pyramid (uint32, nearest-NN downsampling) plus a TIFF and `meta.json`. The segmentation's coordinate transform mirrors the source so it overlays correctly in physical units.

```bash
# Single model (cell)
python examples/segment_zarr.py output.zarr \
    --model data/models/best_model_cell \
    --outdir output_seg

# Dual model (cell + blood, distinct mask IDs) on a sub-ROI
python examples/segment_zarr.py output.zarr \
    --model data/models/best_model_cell \
    --model2 data/models/best_model_blood --mask-id2 2 \
    --roi 0:256,256:768,256:768 \
    --outdir output_seg
```

### Visualize in Neuroglancer

```bash
# Grayscale only
python examples/visualize_zarr.py output.zarr

# Grayscale + segmentation overlay
python examples/visualize_zarr.py output.zarr \
    --segmentation output_seg/segmentation_3d.zarr --keep-open
```

![Neuroglancer visualization example](docs/images/neuroglancer_example.png)

## Benchmarks

Tested on a 52.64 GB BigTIFF volume with 128x128x128 chunks and 8GB memory budget:

| Levels | Compression | Write Time | Read Throughput | Output Size | Ratio |
|:------:|:-----------:|:----------:|:---------------:|:-----------:|:-----:|
|   1    | blosc-lz4   |    57.5s   |    3.13 GB/s    |   35.03 GB  |  66%  |
|   1    | blosc-zstd  |    45.3s   |    3.13 GB/s    |   34.58 GB  |  66%  |
|   1    | none        |    54.5s   |    4.59 GB/s    |   53.13 GB  | 101%  |
|   3    | blosc-lz4   |   2m 18s   |    3.10 GB/s    |   40.07 GB  |  76%  |
|   3    | blosc-zstd  |   2m 05s   |    3.12 GB/s    |   39.55 GB  |  75%  |
|   3    | none        |   2m 17s   |    4.21 GB/s    |   61.13 GB  | 116%  |
|   5    | blosc-lz4   |   2m 05s   |    2.94 GB/s    |   40.16 GB  |  76%  |
|   5    | blosc-zstd  |   1m 51s   |    3.19 GB/s    |   39.63 GB  |  75%  |
|   5    | none        |   2m 05s   |    3.72 GB/s    |   61.41 GB  | 117%  |
|   7    | blosc-lz4   |   2m 18s   |    3.14 GB/s    |   40.16 GB  |  76%  |
|   7    | blosc-zstd  |   2m 06s   |    3.13 GB/s    |   39.63 GB  |  75%  |
|   7    | none        |   2m 21s   |    3.83 GB/s    |   61.41 GB  | 117%  |

**Recommendations:**
- Use `blosc-zstd` for best compression with good read speed
- Use `blosc-lz4` for fastest decompression
- 3-5 pyramid levels is the sweet spot (minimal size increase beyond that)

## Why OME-Zarr?

- Native Neuroglancer support
- Works with napari, FIJI, dask, xarray
- Cloud-ready (S3/GCS or local filesystem)
- Built-in multi-resolution pyramid spec (NGFF v0.4)

## Alternative Tools

Other OME-Zarr conversion tools worth evaluating:

- [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) - Java-based, supports all Bio-Formats file formats
- [BatchConvert](https://github.com/Euro-BioImaging/BatchConvert) - Nextflow wrapper for parallelized batch conversion

## Requirements

- Python 3.10+
- numpy, tifffile, zarr, scikit-image, tqdm, neuroglancer
