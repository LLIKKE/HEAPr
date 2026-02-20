import os
import pprint

import torch
import transformers
from transformers import AutoTokenizer

from pruning.pruning_global import pruning_global
from utils.data_utils import get_wikitext2
from utils.process_args import parse_args_and_setup_logging
from utils.utils import set_seed
from utils.eval_utils import ppl_eval

def main():
    args, logger = parse_args_and_setup_logging()
    logger.info("Arguments: ")
    logger.info(pprint.pformat(vars(args)))
    logger.info("--" * 30)
    set_seed(args.seed)
    logger.info("seed set to {}".format(args.seed))

    config = transformers.AutoConfig.from_pretrained(
        args.model_path, token=None, trust_remote_code=True
    )

    logger.info(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if "deepseek" in args.model_path:
        from models.deepseek.modeling_deepseek import DeepseekForCausalLM

        model = DeepseekForCausalLM.from_pretrained(
            args.model_path, config=config, torch_dtype="auto", device_map="auto"
        )
    elif "Qwen" in args.model_path:
        from transformers import Qwen2MoeForCausalLM, Qwen3MoeForCausalLM

        config.rope_scaling = None
        if "Qwen3" in args.model_path:
            model = Qwen3MoeForCausalLM.from_pretrained(
                args.model_path, config=config, torch_dtype="auto", device_map="auto"
            )
        else:
            model = Qwen2MoeForCausalLM.from_pretrained(
                args.model_path, config=config, torch_dtype="auto", device_map="auto"
            )
    model.eval()

    if os.path.exists(f"{args.log_dir}/{args.model_name}/{args.cali_data}.pt"):
        cali_data = torch.load(
            f"{args.log_dir}/{args.model_name}/{args.cali_data}.pt",
            weights_only=False,
        )
    else:
        cali_data = get_wikitext2(
            nsamples=args.cali_nsamples, tokenizer=tokenizer, args=args
        )
        print(f"{args.log_dir}/{args.model_name}/{args.cali_data}.pt")
        torch.save(
            cali_data, f"{args.log_dir}/{args.model_name}/{args.cali_data}.pt"
        )
    model.use_cache = False

    pruning_global(model, cali_data, config, args, logger)

    ppl = ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb'],batch_size=12)
    logger.info(f"PPL: {ppl}")

    if args.zero_shot:
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import make_table
        from lm_eval.models.huggingface import HFLM
        import lm_eval

        task_manager = TaskManager()
        tasks = task_manager.match_tasks(args.tasks)
        hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_batch_size="auto",
            trust_remote_code=True
        )
        results = lm_eval.simple_evaluate(
            hflm, tasks=tasks, batch_size="auto", max_batch_size=256
        )

        metric_vals = {}

        for task, result in results["results"].items():
            task_metrics = {}
            if "acc_norm,none" in result:
                task_metrics["acc_norm,none"] = round(result["acc_norm,none"], 4)
                task_metrics["acc_norm_stderr,none"] = round(
                    result.get("acc_norm_stderr,none", 0.0), 4
                )

            if "acc,none" in result:
                task_metrics["acc,none"] = round(result["acc,none"], 4)
                task_metrics["acc_stderr,none"] = round(
                    result.get("acc_stderr,none", 0.0), 4
                )

            metric_vals[task] = task_metrics

        logger.info("\n" + make_table(results))


if __name__ == "__main__":
    main()
