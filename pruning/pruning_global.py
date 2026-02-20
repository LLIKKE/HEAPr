import gc
import os

import tqdm
import torch
import torch.nn as nn


from utils.utils import count_parameters, find_named_layers


def build_expert_sensitivity_hook(name, layer_infos):
    """Build a forward hook to record gradients and sensitivity statistics."""
    if name not in layer_infos:
        layer_infos[name] = {}

    def forward_hook(module, inp, out):
        device = inp[0].device

        def tensor_backward_hook(grad):
            if "token" not in layer_infos[name]:
                layer_infos[name] = {"token": 0, "mean": 0}
            g = grad.detach().to(torch.float32)
            if "grad" not in layer_infos[name]:
                layer_infos[name]["grad"] = torch.zeros_like(g.T @ g, device="cpu")
            layer_infos[name]["grad"].add_((g.T @ g).cpu())

            layer_infos[name]["token"] += g.shape[0]
            layer_infos[name]["mean"] = (
                layer_infos[name]["grad"] / layer_infos[name]["token"]
            ).cpu()

        if out.requires_grad:
            out.register_hook(tensor_backward_hook)
        else:
            if "sen" not in layer_infos[name]:
                layer_infos[name].update({"sen": 0, "mean_sen": 0, "token": 0})
            n, d = inp[0].shape
            X = inp[0].to(torch.float32).T
            W = module.weight.to(torch.float32).T
            M = layer_infos[name]["mean"].to(device)

            X_squared_sum = torch.sum(X ** 2, dim=1)
            WM = torch.matmul(W, M)
            WMW_sum = torch.sum(WM * W, dim=1)
            sen_k = X_squared_sum * WMW_sum
            layer_infos[name]["token"] += n
            layer_infos[name]["sen"] += sen_k.detach().cpu()
            layer_infos[name]["mean_sen"] = (
                layer_infos[name]["sen"] / layer_infos[name]["token"]
            )

    return forward_hook


def clear_param_grads_backward_hook(module, grad_input, grad_output):
    """Clear parameter gradients to reduce memory usage."""
    if hasattr(module, "weight") and module.weight.grad is not None:
        module.weight.grad = None
    if (
        hasattr(module, "bias")
        and module.bias is not None
        and module.bias.grad is not None
    ):
        module.bias.grad = None


def prune_expert_neurons_inplace(experts, grouped_indices):
    """Prune expert weights in-place using provided indices."""
    for group, expert in zip(grouped_indices, experts):
        if isinstance(expert, IdentityExpertMLP):
            continue
        sorted_group = sorted(list(set(group)))

        expert.gate_proj.weight.data = expert.gate_proj.weight.data[sorted_group, :]
        expert.up_proj.weight.data = expert.up_proj.weight.data[sorted_group, :]
        expert.down_proj.weight.data = expert.down_proj.weight.data[:, sorted_group]


class IdentityExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.identity = nn.Identity()

    def forward(self, x):
        return x


def pruning_global(model, cali_data, config, args, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    logger.info(f"🚀 Starting {args.model_name} GLOBAL pruning...")

    save_path = f"./Sk_{args.model_name}_{args.cali_data}_{args.cali_nsamples}.pt"
    if "deepseek" in args.model_name.lower():
        from models.deepseek.modeling_deepseek import DeepseekMoE
    elif "qwen" in args.model_name.lower():
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    # Stages 1 & 2: Calculate and save sensitivity scores if they don't exist
    if not os.path.exists(save_path):
        wikidata = cali_data
        model.use_cache = False
        nsamples = args.cali_nsamples
        bs = args.cali_batch_size
        num_batch = nsamples // bs
        layer_infos = {}
        hook_handles = []

        linear_layers = find_named_layers(model, layers=[torch.nn.Linear])
        logger.info("Registering hooks for experts...")
        for name, module in linear_layers.items():
            if "down_proj" in name and ".experts" in name:
                fwd_hook = build_expert_sensitivity_hook(name, layer_infos)
                hook_handles.append(module.register_forward_hook(fwd_hook))
            hook_handles.append(
                module.register_full_backward_hook(clear_param_grads_backward_hook)
            )
        logger.info(f"✅ Registered {len(hook_handles)} hooks.")

        # Stage 1: Collect gradients
        logger.info("Stage 1: Collecting gradients...")
        model.train()
        with torch.enable_grad():
            for i in tqdm.tqdm(range(num_batch), unit="batch", desc="Stage 1"):
                inputs = wikidata["input_ids"][i * bs : (i + 1) * bs, :].to(
                    model.device
                )
                masks = wikidata["attention_mask"][i * bs : (i + 1) * bs, :].to(
                    model.device
                )
                labels = inputs.clone()
                outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                loss.backward()
                model.zero_grad()
        torch.cuda.empty_cache()
        logger.info("✅ Gradients collected.")

        # Stage 2: Hook activations & compute sensitivities
        logger.info("Stage 2: Hooking activations & computing sensitivities...")
        model.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(num_batch), unit="batch", desc="Stage 2"):
                inputs = wikidata["input_ids"][i * bs : (i + 1) * bs, :].to(
                    model.device
                )
                masks = wikidata["attention_mask"][i * bs : (i + 1) * bs, :].to(
                    model.device
                )
                labels = inputs.clone()
                model(input_ids=inputs, attention_mask=masks, labels=labels)
        torch.cuda.empty_cache()
        logger.info("✅ Activations collected.")

        logger.info("Removing all hooks...")
        for handle in hook_handles:
            handle.remove()
        hook_handles.clear()
        logger.info("✅ All hooks have been removed.")

        # Consolidate and save sensitivity scores
        save_Sk = {}
        layers = [layer for layer in model.model.layers]
        for idx, layer in enumerate(layers):
            if (
                ("deepseek" in args.model_name.lower() and isinstance(layer.mlp, DeepseekMoE)) or
                ("qwen" in args.model_name.lower() and isinstance(layer.mlp, (Qwen2MoeSparseMoeBlock, Qwen3MoeSparseMoeBlock)))
            ):
                S_k_layer = []
                for i in range(len(layer.mlp.experts)):
                    name = f"model.layers.{idx}.mlp.experts.{i}.down_proj"
                    if name not in layer_infos or "mean_sen" not in layer_infos[name]:
                        S_k_layer.append(torch.full((config.moe_intermediate_size,), -1.0))
                    else:
                        S_k_layer.append(layer_infos[name]["mean_sen"])
                save_Sk[idx] = torch.cat(S_k_layer, dim=0)

        torch.save(save_Sk, save_path)
        logger.info(f"✅ Sensitivity results calculated and saved at {save_path}")
        del layer_infos
        gc.collect()
        torch.cuda.empty_cache()
    # --- Stage 3: Global Pruning ---
    logger.info("Stage 3: Performing GLOBAL expert pruning...")

    # Load all sensitivity scores
    saved_S_k = torch.load(save_path, map_location="cpu")
    print(save_path)
    all_sensitivities = []
    # (layer_idx, neuron_idx_within_layer) to map back after global sort
    all_indices_map = []

    layers = [layer for layer in model.model.layers]
    original_param = count_parameters(model, logger)
    logger.info("Collecting all sensitivity scores...")
    for idx, layer in enumerate(layers):
        if (
            ("deepseek" in args.model_name.lower() and isinstance(layer.mlp, DeepseekMoE)) or
            ("qwen" in args.model_name.lower() and isinstance(layer.mlp, (Qwen2MoeSparseMoeBlock, Qwen3MoeSparseMoeBlock)))
        ):
            if idx in saved_S_k:
                S_k = saved_S_k[idx]
                all_sensitivities.append(S_k)
                indices = torch.stack(
                    [torch.full_like(S_k, fill_value=idx), torch.arange(len(S_k))],
                    dim=1,
                )
                all_indices_map.append(indices)

    if not all_sensitivities:
        logger.warning("No MoE layers with sensitivity scores found to prune.")
        return

    # Concatenate all scores and their mapping info into global tensors
    global_S_k = torch.cat(all_sensitivities, dim=0)
    global_indices_map = torch.cat(all_indices_map, dim=0).long()

    # Determine the total number of neurons to keep globally
    total_neurons = global_S_k.shape[0]
    k = int(total_neurons * (1 - args.compress_ratio))

    logger.info(f"Globally ranking {total_neurons} neurons and keeping top {k}...")
    _, top_k_indices = torch.topk(global_S_k, k, largest=True)

    # Get the original (layer_idx, neuron_idx) pairs for the neurons to be kept
    retained_neurons_info = global_indices_map[top_k_indices]

    # Group the retained neuron indices by their original layer
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

            prune_expert_neurons_inplace(layer.mlp.experts, grouped_indices_per_expert)
            torch.cuda.empty_cache()
    pruned_param = count_parameters(model, logger)
    logger.info(f"Actual compress ratio: {1 - pruned_param / original_param}")

    logger.info("✅ Global pruning completed.")


def create_hooks(name, layer_infos):
    return build_expert_sensitivity_hook(name, layer_infos)


def memory_release_backward_hook(module, grad_input, grad_output):
    return clear_param_grads_backward_hook(module, grad_input, grad_output)


def reduce_expert(experts, grouped_indices, logger=None):
    return prune_expert_neurons_inplace(experts, grouped_indices)


