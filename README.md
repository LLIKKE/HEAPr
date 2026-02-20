
# HEAPr

<div align="center">

<a href="https://openreview.net/forum?id=JAbMgS7gl6" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/ICLR-2026-b31b1b.svg" alt="ICLR 2026">
</a>

<a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License: CC BY-NC 4.0">
</a>

<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">

</div>



Official implementation of the **ICLR 2026** paper: **[HEAPr: Hessian-based Efficient Atomic Expert Pruning in Output Space](https://openreview.net/forum?id=JAbMgS7gl6)**

---

### Overview

**HEAPr** is a structured pruning method for Mixture-of-Experts models that prunes at a finer granularity than experts by decomposing each expert into **atomic experts**. This enables:

1. **Flexible pruning granularity**: structured pruning at the atomic-expert level, yielding practical speedups across different hardware.
2. **Second-order pruning criterion**: inspired by Optimal Brain Surgery, leveraging a second-order information matrix to achieve state-of-the-art performance under pruning.
3. **Low calibration cost**: pruning can be completed with two forward passes + one backward pass on a small calibration set.

<div align="center">
  <img src="pruning/20260220-153611.png" alt="HEAPr overview" width="70%" />
</div>


---

### Installation

#### Setup

```bash
conda create -n heapr python=3.10 -y
conda activate heapr
pip install -r requirements.txt
```

---

### Usage

`--model_path` can be set to:
- `deepseek-ai/deepseek-moe-16b-base`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- `Qwen/Qwen3-30B-A3B`
- `Qwen/Qwen2-57B-A14B`

Example command:

```bash
python main.py \
  --model_path "deepseek-ai/deepseek-moe-16b-base" \
  --compress_ratio 0.2 \
  --cali_data "wiki" \
  --cali_nsamples 128 \
  --cali_batch_size 8 \
  --eval_batch_size 128 \
  --zero_shot \
  --tasks openbookqa arc_easy winogrande hellaswag arc_challenge piqa mathqa \
  --log_dir "./log_pruning"
```

---

### Main Results

Comparison of perplexity on WikiText2 / PTB and average accuracy across selected zero-shot tasks.

| Model | Pruning Ratio | Wiki (ppl) | PTB (ppl) | OBQA | ARC-e | WinoG | HellaS | ARC-c | PIQA | MathQA | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeekMoE-16B-Base | 20% | 6.54 | 9.88 | 33.40 | 75.59 | 70.56 | 57.70 | 44.03 | 78.51 | 31.16 | 55.85 |
|  | 40% | 6.80 | 10.86 | 31.20 | 73.40 | 69.69 | 53.61 | 41.98 | 76.71 | 29.45 | 53.72 |
| Qwen1.5-MoE-A2.7B-Chat | 25% | 8.31 | 14.12 | 30.40 | 68.43 | 66.30 | 55.22 | 37.88 | 76.06 | 35.01 | 52.76 |
|  | 50% | 9.24 | 17.58 | 26.00 | 63.13 | 64.01 | 46.17 | 34.13 | 69.80 | 33.74 | 48.14 |
| Qwen2-57B-A14B | 40% | 5.92 | 9.34 | 33.20 | 75.25 | 74.43 | 62.88 | 46.33 | 80.74 | 38.49 | 58.76 |
| Qwen3-30B-A3B | 25% | 9.10 | 16.80 | 33.40 | 76.01 | 69.77 | 54.67 | 49.32 | 77.37 | 49.41 | 58.56 |
|  | 50% | 11.22 | 26.29 | 23.60 | 67.21 | 61.80 | 38.19 | 38.82 | 66.59 | 35.88 | 47.44 |

---

### Citation

If you use this codebase or results in your research or product, please cite:

```bibtex
@inproceedings{
  li2026heapr,
  title={{HEAP}r: Hessian-based Efficient Atomic Expert Pruning in Output Space},
  author={Ke Li and Zheng Yang and Zhongbin Zhou and Xuefeng and Zhonglin Jiang and Wenxiao Wang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=JAbMgS7gl6}
}
```

---

### License

This project is licensed under **CC BY-NC 4.0**.
