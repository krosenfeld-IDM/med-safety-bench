# med-safety-bench

This repository contains the ```MedSafetyBench``` benchmark dataset from the paper [MedSafetyBench: Evaluating and Improving the
Medical Safety of Large Language Models](https://arxiv.org/abs/2403.03744). This benchmark dataset consists of harmful medical requests and medical safety demonstrations and is designed to evaluate and improve the medical safety of large language models.

**Note**: This dataset contains content that may be used for harmful purposes. It should be used for research only. By using this dataset, you agree to use it for research only.

## Dataset information

900 harmful medical requests
- ```datasets/med_harm_gpt4``` (n = 450)
- ```datasets/med_harm_llama2``` (n = 450)

900 medical safety demonstrations
- ```datasets/training/med_safety/ft_safety_med_n900.json```
- Other files in the folder are subsets of the above

900 general safety demonstrations (from [Bianchi et al. (2023)](https://arxiv.org/abs/2309.07875))
- ```datasets/training/gen_safety/ft_safety_gen_n900.json```
- Other files in the folder are subsets of the above

The 900 medical safety demonstrations and 900 general safety demonstrations are pooled together in ```datasets/training/both_safety/ft_safety_both_n1800.json```

## Citation

```
@article{han2024towards,
  title={Towards safe large language models for medicine},
  author={Han, Tessa and Kumar, Aounon and Agarwal, Chirag and Lakkaraju, Himabindu},
  journal={arXiv preprint arXiv:2403.03744},
  year={2024}
}
```
