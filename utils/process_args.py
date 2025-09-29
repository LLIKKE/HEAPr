import argparse
import transformers
import os
from datetime import datetime
import logging
from termcolor import colored
import pprint
import shutil



def parser():
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
        "--compress_radio",
        type=float,
        default=0.75,
        help="compress radio for coloum cluster",
    )
    parser.add_argument(
        "--eval_batchsize",
        type=int,
        default=64,
        help="eval_batchsize",
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
        "--cali_batchsize",
        type=int,
        default=1,
        help="cali_batchsize",
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["arc_challenge",
                 "arc_easy",
                 #"boolq",
                 #"hellaswag",
                 #"lambada_openai",
                 #"openbookqa",
                 #"piqa",
                 #"social_iqa",
                 #"winogrande",
                 ])

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
    parser.add_argument(
        "--log_dir", type=str, default="./", help="log path"
    )
    args, unknown = parser.parse_known_args()

    return args, unknown

def create_logger(exp_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def process_args():
    args, unknown_args = parser()
    args.model_name = args.model_path.split("/")[-1]
    log = f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    args.exp_dir = os.path.join(args.log_dir,args.model_name, log)
    os.makedirs(args.exp_dir, exist_ok=True)
    logger = create_logger(args.exp_dir)
    return args, logger
