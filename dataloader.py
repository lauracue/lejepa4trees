# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:06:06 2026

@author: cuel001
"""
import os
import rasterio
import numpy as np
import random
from rasterio.windows import Window
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import ConcatDataset


class UnifiedDataset(Dataset):
    def __init__(self, ssl_ds, labeled_ds):
        self.data = ConcatDataset([ssl_ds, labeled_ds])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class JPGDataset(Dataset):

    def __init__(
        self,
        jpg_files,
        crop_size=128):
        
        self.jpg_files = jpg_files
        self.crop_size = crop_size

        self.test = v2.Compose(
            [
                v2.Resize(crop_size),
                v2.CenterCrop(crop_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.jpg_files)


    def __getitem__(self, idx):

        path = self.jpg_files[idx]

        image = Image.open(path).convert("RGB")
        
        label = int(Path(path).stem.split('cls')[1])
    
        
        return torch.stack([self.test(image)]), label
        
        
        

class RandomRasterCropDataset(Dataset):

    def __init__(
        self,
        raster_files,
        crop_size=128,
        views = 2,
        bands=(1, 2, 3),
        max_tries=10,
        nodata_value=0,
        samples=1000,
    ):
        self.raster_files = raster_files
        self.crop_size = crop_size
        self.bands = bands
        self.max_tries = max_tries
        self.nodata_value = nodata_value
        self.samples = samples
        self.views = views
        
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(crop_size, scale=(0.4, 1.0), ratio=(0.8, 1.2)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomChoice(
                                    [
                                        v2.Identity(),
                                        v2.RandomRotation((90, 90)),
                                        v2.RandomRotation((180, 180)),
                                        v2.RandomRotation((270, 270)),
                                    ]),
                v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.2, 0.05)], p=0.5),
                v2.RandomGrayscale(p=0.05),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return self.samples

    def _is_bad_crop(self, image):
        img = np.array(image)

        total_pixels = img.size
        bad_pixels = np.sum((img == 0) | (img == 255))

        ratio = bad_pixels / total_pixels
        return ratio > 0.95

    def __getitem__(self, idx):

        for _ in range(self.max_tries):

            path = random.choice(self.raster_files)

            with rasterio.open(path) as src:

                H, W = src.height, src.width

                if H <= self.crop_size or W <= self.crop_size:
                    continue

                y = random.randint(0, H - self.crop_size)
                x = random.randint(0, W - self.crop_size)

                window = Window(x, y, self.crop_size, self.crop_size)

                crop = src.read(
                    self.bands,
                    window=window,
                    out_dtype=np.uint8,
                )


                if not self._is_bad_crop(crop):
                    break  # valid crop found
                    
        crop = torch.from_numpy(crop)
        return torch.stack([self.aug(crop) for _ in range(self.views)])
