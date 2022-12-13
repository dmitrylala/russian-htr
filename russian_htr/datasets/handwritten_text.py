import json
from pathlib import Path
from typing import Optional, Union

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

import torch
from torchok.data.datasets import ImageDataset
from torchok.constructor import DATASETS


@DATASETS.register_class
class HandwrittenDataset(ImageDataset):
    data_modes = ['all', 'train', 'val', 'test']

    def __init__(
        self,
        data_folder: str,
        mode: str,
        transform: Union[BasicTransform, BaseCompose],
        augment: Optional[Union[BasicTransform, BaseCompose]] = None,
        input_dtype: str = 'float32'
    ):
        """
        Dataset with pairs (image, text).
        Args:
            data_folder: Directory with subfolders 'images' and 'labels',
                        also with .json's with markup.
            mode: 'train', 'val' or 'test'
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Type of output image tensor.
        Raises:
            RuntimeError: if dataset or metadata file not found or corrupted.
        """
        super().__init__(transform=transform, augment=augment, input_dtype=input_dtype)

        if mode not in self.data_modes:
            raise ValueError(f"Mode {mode} is invalid")
        self.mode = mode

        data_folder = Path(data_folder)
        with open(data_folder / f"{mode}.json") as f:
            markup = json.load(f)

        self.images = sorted(map(lambda x: str(data_folder / 'images' / x), markup.keys()))

        if mode != 'test':
            label_paths = sorted(map(lambda x: str(data_folder / 'labels' / x), markup.values()))
            self.labels = []
            for label_path in label_paths:
                with open(label_path) as f:
                    label = json.load(f)
                self.labels.append(label)

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.images)

    def get_raw(self, idx: int) -> dict:
        """
        Get item sample.
        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations.
            sample['target'] - Text on image.
            sample['index'] - Index.
        """
        image = self._read_image(self.images[idx])
        sample = {"image": image, 'index': idx}
        if self.mode != 'test':
            sample['target'] = self.labels[idx]

        sample = self._apply_transform(self.augment, sample)

        return sample

    def __getitem__(self, idx: int) -> dict:
        """
        Get item sample.
        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['target'] - Text on image.
            sample['index'] - Index.
        """
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)

        # converting to torch.FloatTensor; self.input_dtype='float32' by default
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        return sample
