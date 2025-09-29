# HEAPr

Official implementation for the paper **HEAPr: Hessian-based Efficient Atomic Expert Pruning in Output Space**.

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