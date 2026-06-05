# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:50:26 2026

@author: cuel001
"""

import rasterio
import zarr
import numpy as np
from numcodecs import Blosc
from pathlib import Path
from rasterio.windows import Window
from zarr.codecs import BloscCodec
from tqdm import tqdm
from zarr.codecs import BloscCodec

def convert_tiff_to_zarr(tiff_path, zarr_path, chunk_size=512):

    with rasterio.open(tiff_path) as src:

        # ✅ only first 3 channels
        bands = (1, 2, 3)
        
        dtype = src.dtypes[0]

        H = src.height
        W = src.width
        C = len(bands)

        codec = BloscCodec(
            cname="zstd",
            clevel=3,
            shuffle="bitshuffle",
        )
        
        z = zarr.create_array(
            store=zarr_path,
            shape=(C, H, W),
            chunks=(C, chunk_size, chunk_size),
            dtype=dtype,
            compressors=[codec],   
            overwrite=True,
            write_empty_chunks=False, 
        )

        for y in range(0, H, chunk_size):
            for x in range(0, W, chunk_size):

                h = min(chunk_size, H - y)
                w = min(chunk_size, W - x)

                window = Window(x, y, w, h)

                data = src.read(bands, window=window)

                z[:, y:y+h, x:x+w] = data

    print(f"Saved: {zarr_path}")


RASTERS_FOLDER = ["C:/PostDoc/Seminar_Jefferson/data/2023/ortos_2023-11",
                  "C:/PostDoc/Seminar_Jefferson/data/2025/ortos_2025-11",
                  "C:/PostDoc/Seminar_Jefferson/data/2025/ortos_2025-12"]

raster_files = []
for pth in RASTERS_FOLDER:
    raster_files.extend(Path(pth).glob("*.tif"))

print(f"Found {len(raster_files)} rasters")
    
    
output_dir = Path("C:/PostDoc/Seminar_Jefferson/data/zarr_dataset/rasters_zarr")


output_dir.mkdir(parents=True, exist_ok=True)

for i, path in enumerate(tqdm(raster_files)):

    out_zarr = output_dir / (path.stem + ".zarr")

    convert_tiff_to_zarr(str(path), str(out_zarr))
