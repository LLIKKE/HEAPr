# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import argparse
import logging
import os
from datetime import datetime

from termcolor import colored



def build_arg_parser():
    """Build an argument parser for the CLI."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/like/deepseek-moe-16b-base",
        help="model path",
    )
    parser.add_argument(
        "--global_pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="global_pruning.",
    )

    parser.add_argument(
        "--compress_ratio",
        "--compress_radio",
        dest="compress_ratio",
        type=float,
        default=0.75,
        help="compression ratio for column clustering",
    )
    parser.add_argument(
        "--eval_batch_size",
        "--eval_batchsize",
        dest="eval_batch_size",
        type=int,
        default=64,
        help="eval batch size",
    )
    parser.add_argument(
        "--cali_data",
        type=str,
        default="wiki",
        help="cali data",
    )
    parser.add_argument(
        "--cali_nsamples",
        type=int,
        default=128,
        help="cali nsamples number",
    )

    parser.add_argument(
        "--cali_batch_size",
        "--cali_batchsize",
        dest="cali_batch_size",
        type=int,
        default=1,
        help="calibration batch size",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "arc_challenge",
            "arc_easy",
            #"boolq",
            #"hellaswag",
            #"lambada_openai",
            #"openbookqa",
            #"piqa",
            #"social_iqa",
            #"winogrande",
        ],
    )

    parser.add_argument(
        "--zero_shot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="eval zero shot.",
    )
    parser.add_argument(
        "--generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate example.",
    )
    parser.add_argument("--log_dir", type=str, default="./", help="log path")
    return parser


def parser():
    return build_arg_parser().parse_known_args()

def create_logger(exp_dir, dist_rank=0, name=""):
    """Create a configured logger for console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(console_handler)

    log_file = os.path.join(
        exp_dir, f"log_rank{dist_rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


def parse_args_and_setup_logging():
    """Parse arguments, prepare experiment directory, and create logger."""
    args, unknown_args = parser()
    args.compress_radio = args.compress_ratio
    args.eval_batchsize = args.eval_batch_size
    args.cali_batchsize = args.cali_batch_size
    args.model_name = args.model_path.split("/")[-1]
    log = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.exp_dir = os.path.join(args.log_dir, args.model_name, log)
    os.makedirs(args.exp_dir, exist_ok=True)
    logger = create_logger(args.exp_dir)
    return args, logger


def process_args():
    return parse_args_and_setup_logging()
