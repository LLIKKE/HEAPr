import torch
import torch.nn as nn
import numpy as np
import random
import os
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

@torch.no_grad()
def ppl_eval(
    model,
    tokenizer,
    datasets=["wikitext2", "ptb", "c4"],
    model_seq_len=2048,
    batch_size=2,
    seed=0,
    loaders=None,
):
    def _perplexity(total_nll, total_tokens):
        return torch.exp(total_nll / total_tokens)

    model.eval()
    ppls = {}

    # main_device = next(model.parameters()).device
    #main_device = list(model.parameters())[-1].device

    for dataset in datasets:
        if loaders is not None and dataset in loaders:
            data = loaders[dataset]
        else:
            data = get_test_data(
                dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size
            )

        main_device = model.device
        total_nll = torch.tensor(0.0, device=main_device)
        total_tokens = 0

        with tqdm(data, total=len(data), desc=f"Evaluating {dataset} - Perplexity") as progress_bar:
            for batch in progress_bar:
                batch = batch.to(main_device)
                output = model(batch)
                logits = output.logits if hasattr(output, "logits") else output[0]
                logits = logits.to(main_device)

                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = batch[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                nll = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_nll += nll
                total_tokens += shift_labels.numel()

        if total_tokens == 0:
            raise ValueError(f"Empty evaluation data for dataset={dataset!r}.")
        ppl = _perplexity(total_nll, total_tokens)
        ppls[dataset] = ppl.item()
    print(ppls)
    return ppls

def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len
        if nsamples == 0:
            return IndexDataset(tensors=torch.empty((0, seq_len), dtype=test_ids.dtype))

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test',trust_remote_code=True)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
