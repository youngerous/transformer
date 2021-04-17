import glob
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import load_config
from dataset import get_loader
from trainer import Trainer
from utils import ResultWriter, fix_seed


def main(rank, hparams, ngpus_per_node: int):
    fix_seed(hparams.seed)
    resultwriter = ResultWriter(hparams.result_path)
    if hparams.distributed:
        hparams.rank = hparams.rank * ngpus_per_node + rank
        print(f"Use GPU {hparams.rank} for training")
        dist.init_process_group(
            backend=hparams.dist_backend,
            init_method=hparams.dist_url,
            world_size=hparams.world_size,
            rank=hparams.rank,
        )

    # get dataloaders
    loaders = [
        get_loader(
            batch_size=hparams.batch_size,
            path=hparams.root_path,
            workers=hparams.workers,
            mode=mode,
            distributed=hparams.distributed,
        )
        for mode in ["train", "dev", "test"]
    ]

    # get model
    if hparams.distributed:
        if rank != 0:
            dist.barrier()
        model = None
        if rank == 0:
            dist.barrier()
    else:
        model = None

    # training phase
    trainer = Trainer(hparams, loaders, model, resultwriter)
    best_result = trainer.fit()

    # testing phase
    if rank in [-1, 0]:
        version = best_result["version"]
        state_dict = torch.load(
            glob.glob(
                os.path.join(hparams.ckpt_path, f"version-{version}/best_model_*.pt")
            )[0]
        )
        test_result = trainer.test(state_dict)

        # save result
        best_result.update(test_result)
        resultwriter.update(hparams, **best_result)


if __name__ == "__main__":
    hparams = load_config()
    ngpus_per_node = torch.cuda.device_count()

    if hparams.distributed:
        hparams.rank = 0
        hparams.world_size = ngpus_per_node * hparams.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(hparams, ngpus_per_node))
    else:
        main(hparams.rank, hparams, ngpus_per_node)
