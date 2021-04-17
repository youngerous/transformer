import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils import SequentialDistributedSampler


class CustomDataset(Dataset):
    def __init__(self, root_path: str, mode: str):
        super(CustomDataset, self).__init__()
        assert mode in ["train", "dev", "test"]
        self.dset = None

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, idx: int):
        return


def get_loader(
    batch_size: int, path: str, workers: int, mode: str, distributed: bool = False
) -> DataLoader:
    """
    :param batch_size: Mini-batch size
    :param path: Root path of dataset
    :param workers: Number of dataloader workers
    :param mode: Choose 'train', 'dev', or 'test'
    :param distributed: Whether to use ddp

    :return: dataloader
    """
    assert mode in ["train", "dev", "test"]

    dset = CustomDataset(root_path=path, mode=mode, tok=tok)
    shuffle_flag = mode == "train"
    sampler = None
    if distributed:
        sampler = (
            DistributedSampler(dset)
            if mode == "train"
            else SequentialDistributedSampler(dset)
        )
        shuffle_flag = False

    return DataLoader(
        dataset=dset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_flag,
        num_workers=workers,
        pin_memory=True,
        drop_last=(mode == "train"),
    )
