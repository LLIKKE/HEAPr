# HEAPr

Official implementation for the paper **HEAPr: Hessian-based Efficient Atomic Expert Pruning in Output Space**.[(Paper Link)](https://arxiv.org/abs/2509.22299v1)

🚧 Code is coming soon. Currently, `DeepSeekMoE-16B-base` checkpoint loading and pruning evaluation are available for testing.

---

## Getting Started


### 1. Download Model Weights

We recommend downloading pretrained model weights locally from [Hugging Face](https://huggingface.co/meta-llama).
Make sure to update the path in the script accordingly.

---

### 2. Run

You can start testing checkpoints and pruning results with:

```bash
bash scripts/main.sh
```

---

### 3. Results

| Model                  | Pruning Rate | Wiki (ppl) | ptb (ppl) | obqa  | ARC_e | WinoG | HellaS | ARC_C | PIQA  | MathQA | Average |
|------------------------|--------------|------------|-----------|-------|-------|-------|--------|-------|-------|--------|---------|
| DeepSeekMoE-16B-Base   | 20%          | 6.64       | 10.51     | 31.54 | 75.88 | 71.43 | 57.39  | 44.62 | 79.05 | 31.51  | 55.92   |
|                        | 40%          | 6.91       | 11.56     | 30.00 | 73.78 | 69.06 | 52.29  | 40.61 | 76.50 | 30.12  | 53.19   |
| Qwen1.5-MoE-A2.7B-Chat | 25%          | 8.14       | 14.76     | 31.80 | 67.22 | 67.82 | 55.67  | 37.56 | 76.39 | 34.87  | 53.05   |
|                        | 50%          | 9.23       | 18.73     | 27.01 | 63.89 | 64.32 | 46.35  | 34.22 | 70.86 | 33.37  | 48.57   |
| Qwen2-57B-A14B         | 40%          | 5.75       | 9.59      | 32.60 | 74.87 | 74.03 | 62.88  | 46.33 | 81.01 | 38.93  | 58.66   |
| Qwen3-30B-A3B          | 25%          | 8.41       | 18.06     | 32.20 | 77.02 | 70.48 | 54.64  | 49.06 | 77.58 | 49.75  | 58.68   |


## Citation

If you find our work useful, please cite:

```
@article{li2025heapr
  title={HEAPr: Hessian-based Efficient Atomic Expert Pruning in Output Space}, 
  author={Ke Li and Zheng Yang and Zhongbin Zhou and Feng Xue and Zhonglin Jiang and Wenxiao Wang},
  journal={arXiv preprint arXiv:2509.22299},
  year={2025},
}
```

---

## License

This project is licensed under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

---