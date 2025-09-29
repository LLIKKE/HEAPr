import tqdm
import torch
import torch.nn as nn
from models.deepseek.modeling_deepseek import DeepseekMoE
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from utils.utils import count_parameters


def reduce_expert(experts, grouped_indices, logger=None):

    for group, expert in zip(grouped_indices, experts):
        if isinstance(expert, MLP_identity):
            continue
        sorted_group = sorted(list(set(group)))

        expert.gate_proj.weight.data = expert.gate_proj.weight.data[sorted_group, :]
        expert.up_proj.weight.data = expert.up_proj.weight.data[sorted_group, :]
        expert.down_proj.weight.data = expert.down_proj.weight.data[:, sorted_group]

class MLP_identity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.identity = nn.Identity()

    def forward(self, x):
        return x

def pruning_global(model, config, args, logger=None):

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    logger.info(f"🚀 Starting {args.model_name} GLOBAL pruning...")
    save_path = f"./Sk_{args.model_name}_{args.cali_data}_{args.cali_nsamples}.pt"
    saved_S_k = torch.load(save_path, map_location="cpu")
    all_sensitivities = []
    all_indices_map = []
    layers = [layer for layer in model.model.layers]
    logger.info("Collecting all sensitivity scores...")
    for idx, layer in enumerate(layers):
        if isinstance(layer.mlp, (DeepseekMoE, Qwen2MoeSparseMoeBlock)):
            if idx in saved_S_k:
                S_k = saved_S_k[idx]
                all_sensitivities.append(S_k)
                indices = torch.stack([
                    torch.full_like(S_k, fill_value=idx),
                    torch.arange(len(S_k))
                ], dim=1)
                all_indices_map.append(indices)

    if not all_sensitivities:
        logger.warning("No MoE layers with sensitivity scores found to prune.")
        return

    global_S_k = torch.cat(all_sensitivities, dim=0)
    global_indices_map = torch.cat(all_indices_map, dim=0).long()

    total_neurons = global_S_k.shape[0]
    k = int(total_neurons * (1 - args.compress_radio))

    logger.info(f"Globally ranking {total_neurons} neurons and keeping top {k}...")
    _, top_k_indices = torch.topk(global_S_k, k, largest=True)

    retained_neurons_info = global_indices_map[top_k_indices]

    indices_to_keep_by_layer = {int(idx): [] for idx in saved_S_k.keys()}
    for info in retained_neurons_info:
        layer_idx, neuron_idx_in_layer = info[0].item(), info[1].item()
        indices_to_keep_by_layer[layer_idx].append(neuron_idx_in_layer)

    logger.info("Applying pruning based on global ranking...")
    subexpert_num = config.moe_intermediate_size

    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Pruning")):
        if idx in indices_to_keep_by_layer:
            num_experts = len(layer.mlp.experts)

            indices_for_this_layer = indices_to_keep_by_layer[idx]

            grouped_indices_per_expert = [[] for _ in range(num_experts)]
            for flat_index in indices_for_this_layer:
                expert_idx = flat_index // subexpert_num
                neuron_idx_in_expert = flat_index % subexpert_num
                if 0 <= expert_idx < num_experts:
                    grouped_indices_per_expert[expert_idx].append(neuron_idx_in_expert)

            reduce_expert(layer.mlp.experts, grouped_indices_per_expert)
            torch.cuda.empty_cache()
    logger.info("✅ Global pruning completed.")