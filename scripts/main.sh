#!/bin/bash

python main.py \
  --model_path "deepseek-moe-16b-base" \
  --compress_radio 0.20 \
  --cali_data "wiki" \
  --eval_batchsize 32 \
  --zero_shot \
  --tasks openbookqa arc_easy winogrande hellaswag arc_challenge piqa mathqa \
  --log_dir "log_pruning"
