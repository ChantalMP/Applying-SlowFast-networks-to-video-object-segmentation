#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

import torch

import slowfast.utils.multiprocessing as mpu
from slowfast.utils.parser import load_config, parse_args

from tools.test_net import test
from tools.train_net import train
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg.device = 'cpu'

    device = torch.device(cfg.device)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    train,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            train(cfg=cfg,device=device)



    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    test,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            test(cfg=cfg,device=device)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
