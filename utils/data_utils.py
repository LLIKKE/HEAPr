# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Any, Dict

import datasets
import torch
import transformers


def get_wikitext2(
    nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, args=None
):
    """Load and sample Wikitext-2 (or C4 fallback) for calibration.

    Args:
        nsamples: Number of samples to draw.
        seed: Random seed for sampling.
        seqlen: Sequence length for each sample.
        model: Model name or path for tokenizer fallback.
        tokenizer: Optional tokenizer instance.
        args: Parsed CLI arguments with cali_data selection.

    Returns:
        Tokenized dataset containing input_ids and attention_mask.
    """
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if args.cali_data == "c4":
        try:
            traindata = datasets.load_dataset(
                "json",
                data_files={"train": "c4-train.00001-of-01024.json"},
                trust_remote_code=True,
            )["train"]
            text = "\n\n".join(traindata["text"])
            text = text[:int(len(text) * 0.1)]
        except FileNotFoundError:
            print("C4 data file not found. Using Wikitext-2 as a fallback.")
            traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
            text = "\n\n".join(traindata["text"])
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
        text = "\n\n".join(traindata["text"])

    trainenc = tokenizer(text, return_tensors="pt")

    random.seed(seed)

    total_sequences = trainenc.input_ids.size(1) // seqlen

    all_input_ids = trainenc.input_ids[:, : total_sequences * seqlen].view(-1, seqlen)
    all_attention_masks = trainenc.attention_mask[
        :, : total_sequences * seqlen
    ].view(-1, seqlen)

    num_to_sample = min(nsamples, total_sequences)
    sample_indices = random.sample(range(total_sequences), num_to_sample)

    trainenc["input_ids"] = all_input_ids[sample_indices]
    trainenc["attention_mask"] = all_attention_masks[sample_indices]
    print(trainenc["input_ids"].shape)
    return trainenc
