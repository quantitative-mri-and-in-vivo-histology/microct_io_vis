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

## Usage

### Convert BigTIFF to OME-Zarr

```bash
# Basic conversion with isotropic chunks (best for visualization)
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128

# Custom memory budget and pyramid levels
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128 -n 4 --max-memory 8G

# Inspect TIFF without converting
python examples/convert_tiff_to_zarr.py input.tif --info

# Estimate memory usage before conversion
python examples/convert_tiff_to_zarr.py input.tif output.zarr -c 128 128 128 --estimate
```

### Visualize in Neuroglancer

```bash
python examples/visualize_zarr.py output.zarr
```

## Why OME-Zarr?

- Native Neuroglancer support
- Works with napari, FIJI, dask, xarray
- Cloud-ready (S3/GCS or local filesystem)
- Built-in multi-resolution pyramid spec (NGFF v0.4)

## Requirements

- Python 3.10+
- numpy, tifffile, zarr, scikit-image, tqdm, neuroglancer
