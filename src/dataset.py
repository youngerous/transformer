import glob
import json
import os
from typing import List, Tuple

import jsonlines
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils import SequentialDistributedSampler


class TranslationDataset(Dataset):
    """Kor-Eng translation dataset

    Args:
        tok (BertTokenizer):
        cached_path (str): Cached(Tokenized) path of dataset
        mode (str): Choose 'train', 'valid', or 'test
        max_len (int): Maximum length of sequence
    """

    def __init__(self, tok, cached_path, mode, max_len):
        super(TranslationDataset, self).__init__()
        assert mode in ["train", "valid", "test"]
        self.max_len = max_len
        self.pad_idx = tok.pad_token_id
        self.eos_idx = tok.sep_token_id  # [SEP] means <eos> in this implementation

        self.dset = []
        with open(cached_path, "r", encoding="utf-8") as f:
            jsonl = list(f)
        for json_str in jsonl:
            self.dset.append(json.loads(json_str))
        print(f"Load {len(self.dset)} {mode} sample")

    def add_pad(self, indice: List[int]) -> List[int]:
        diff = self.max_len - len(indice)
        if diff > 0:
            indice += [self.pad_idx] * diff
        else:
            indice = indice[: self.max_len - 1] + [self.eos_idx]
        return indice

    def get_src_mask(self, indice: torch.Tensor) -> torch.Tensor:
        return (indice != self.pad_idx).unsqueeze(-2)

    def get_tgt_mask(self, indice: torch.Tensor) -> torch.Tensor:
        mask = (indice != self.pad_idx).unsqueeze(-2)
        mask = mask & self.subsequent_mask(indice.shape[-1]).type_as(mask.data)
        return mask

    def subsequent_mask(self, size) -> torch.Tensor:
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        return torch.from_numpy(subsequent_mask) == 0

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        # there are special tokens, which are used in BERT, so we remove them in src
        # [SEP] token in tgt will be used as <eos> token
        src = self.dset[idx]["src"][1:-1]
        tgt = self.dset[idx]["tgt"][1:]

        # add pad token
        src = torch.tensor(self.add_pad(src))
        tgt = torch.tensor(self.add_pad(tgt))

        # get masking vector
        src_mask = self.get_src_mask(src)
        tgt_mask = self.get_tgt_mask(tgt)

        assert len(src) == self.max_len
        assert len(tgt) == self.max_len
        assert len(src_mask[0]) == self.max_len
        assert len(tgt_mask[0]) == self.max_len

        return src, tgt, src_mask, tgt_mask


def get_loader(tok, batch_size, root_path, workers, max_len, mode, distributed=False):
    """
    Args:
        tok (BertTokenizer): BERT tokenizer to use
        batch_size (int): Mini-batch size
        root_path (str): Root path of dataset
        workers (int): Number of dataloader workers
        max_len (int): Maximum length of sequence
        mode (str): Choose 'train', 'valid', or 'test
        distributed (bool): Whether to use ddp

    Returns:
        DataLoader
    """
    assert mode in ["train", "valid", "test"]

    # check if cached
    cached_dir = os.path.join(root_path, f"cached/cached_{mode}.jsonl")
    if not os.path.isfile(cached_dir):
        print(f"There is no cached(tokenized) {mode} file. Start processing...")
        if distributed:
            if rank != 0:
                dist.barrier()
            cache_processed_data(tok, root_path, cached_dir, mode)
            if rank == 0:
                dist.barrier()
        else:
            cache_processed_data(tok, root_path, cached_dir, mode)
        print("Done!")

    # build Dataset and Dataloader
    dset = TranslationDataset(
        tok=tok, cached_path=cached_dir, mode=mode, max_len=max_len
    )
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


def cache_processed_data(tokenizer, root_pth, cached_pth, mode):
    """Convert csv into jsonl"""
    os.makedirs(os.path.join(root_pth, "cached/"), exist_ok=True)

    # load raw data
    df = pd.read_csv(
        glob.glob(os.path.join(root_pth, f"{mode}_*.csv"))[0],
        index_col=False,
    )

    # tokenize and save cached jsonl file
    with jsonlines.open(cached_pth, "w") as f:
        for idx in tqdm(range(len(df))):
            f.write(
                {
                    "src": tokenizer.encode(df.iloc[idx][0]),
                    "tgt": tokenizer.encode(df.iloc[idx][1]),
                }
            )
