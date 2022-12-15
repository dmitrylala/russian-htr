import pickle
from typing import Optional, Union

import numpy as np
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

import torch
from torchok.data.datasets import ImageDataset
from torchok.constructor import DATASETS


@DATASETS.register_class
class HandwrittenDataset(ImageDataset):
    data_modes = ['train', 'test']

    def __init__(
        self,
        ds_pickle: str,
        mode: str,
        transform: Union[BasicTransform, BaseCompose],
        augment: Optional[Union[BasicTransform, BaseCompose]] = None,
        input_dtype: str = 'float32'
    ):
        """
        Dataset class for loading IAM or CVL dataset in .pickle with following format:

        {
        'train': [{writer_1:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},
                    {writer_2:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...],
        'test': [{writer_3:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},
                    {writer_4:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...],
        }

        Pairs (image, text) should be divided by writers.

        Args:
            ds_pickle: Path to .pickle file with python dict in described format.
            mode: 'train' or 'test'.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Type of output image tensor.
        """
        super().__init__(transform=transform, augment=augment, input_dtype=input_dtype)

        if mode not in self.data_modes:
            raise ValueError(f"Mode {mode} is invalid")

        with open(ds_pickle, 'rb') as f:
            ds = pickle.load(f)[mode]

        writer_start = 0 if mode == 'train' else 339  # num_writers in train part of IAM dataset

        # encode writers ids
        writer_ids = sorted(ds.keys())
        self.writer_ids = dict(zip(writer_ids, np.arange(len(writer_ids) + writer_start)))

        samples = []
        writer2idxs = {}
        for writer_id, writer_samples in ds.items():
            start_idx = len(samples)
            idxs = np.arange(start_idx, start_idx + len(writer_samples))
            writer2idxs[self.writer_ids[writer_id]] = idxs

            samples.extend(writer_samples)

        # renaming keys
        for sample in samples:
            sample['image'] = np.array(sample.pop('img').convert('L'))
            sample['target'] = sample.pop('label')
        self.samples = samples

        # attribute for GroupSampler:
        # batches need to contain samples from only one writer
        self.group2idxs = writer2idxs

        self.idx2writer = {
            idx: writer_id
            for writer_id, idxs in writer2idxs.items()
            for idx in idxs
        }

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.samples)

    def get_raw(self, idx: int) -> dict:
        """
        Get item sample.
        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations.
            sample['target'] - Text on image.
            sample['index'] - Index.
        """
        sample = self.samples[idx]
        sample['index'] = idx
        sample['width'] = sample['image'].shape[1]
        sample['writer_id'] = self.idx2writer[idx]

        sample = self._apply_transform(self.augment, sample)

        return sample

    def random_sample_by_wid(self, writer_id):
        idx = np.random.choice(self.group2idxs[writer_id])
        return self[idx]

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
        sample['writer_id'] = torch.from_numpy(np.array([sample['writer_id']])).type(torch.LongTensor)

        return sample
