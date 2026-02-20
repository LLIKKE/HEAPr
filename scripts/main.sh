python main.py \
--model_path "/mnt/bn/global-comprice-xuyong/mlx/users/like.2248/playground/hf_model/deepseek-moe-16b-base" \
--compress_ratio 0.21 \
--cali_data "wiki" \
--cali_nsamples 128 \
--cali_batch_size 8 \
--eval_batch_size 256 \
--zero_shot \
--tasks openbookqa arc_easy winogrande hellaswag arc_challenge piqa mathqa \
--log_dir "./log_pruning"
