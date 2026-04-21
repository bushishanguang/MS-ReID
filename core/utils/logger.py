# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys


def get_experiment_name(save_dir, default_name="reid_baseline"):
    if not save_dir:
        return default_name

    normalized_dir = os.path.normpath(os.path.abspath(save_dir))
    parts = normalized_dir.split(os.sep)

    outputs_index = -1
    for i, part in enumerate(parts):
        if part == "outputs":
            outputs_index = i

    if outputs_index >= 0 and outputs_index + 1 < len(parts):
        return parts[outputs_index + 1]

    return os.path.basename(normalized_dir) or default_name


def setup_logger(name, save_dir, distributed_rank):
    logger_name = get_experiment_name(save_dir, name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "{}.log".format(logger_name)), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
