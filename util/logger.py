# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

logger_initialized = {}

def get_logger(name, save_dir, distributed_rank, filename="log.log"):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.propagate = False
    # don't log results for the non-master process
    if distributed_rank > 0:
        logger.setLevel(logging.ERROR)
        return logger
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.setLevel(logging.INFO)

    logger_initialized[name] = True

    return logger
