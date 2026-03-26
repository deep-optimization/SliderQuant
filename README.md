# SliderQuant: Accurate Post-Training Quantization for LLMs

<div align="center">
<a href="https://deep-optimization.github.io/sliderquant/"><img src="https://img.shields.io/badge/Project%20Page-SliderQuant-lightblue?logo=github" alt="Project Page"></a>&nbsp;<a href="#"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv" alt="arXiv"></a>&nbsp;<a href="https://openreview.net/forum?id=YNqZqw4fLT"><img src="https://img.shields.io/badge/OpenReview-Discussion-8A2BE2?logo=OpenReview" alt="OpenReview"></a>&nbsp;<a href="https://huggingface.co/IntelLabsChina/SliderQuant"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IntelLabsChina%2FSliderQuant-yellow" alt="Hugging Face"></a>

</div>

By Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Zhonghong Ou and Anbang Yao.

This repository is the official PyTorch implementation of "SliderQuant: Accurate Post-Training Quantization for LLMs", accepted to ICLR 2026.

![SliderQuant overview](asserts/main-fig.png)

SliderQuant (Sliding-layer Quantization) is a new learnable post-training quantization framework for LLMs, which consists of two key components:

- Inter-layer sliding quantization couples three types of sliding window designs to address the varying quantization sensitivity of shallow, intermediate and deep layers of any pre-trained LLMs.
- Intra-layer sliding quantization quantizes layers inside the current slidning window in an incremental manner.

## Table Of Contents

- [Main Results](#main-results)
- [Model Zoo](#model-zoo)
- [Install](#install)
- [How To Train](#how-to-train)
- [How To Test](#how-to-test)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Main Results

#### Language Generation

![Table 1](asserts/table-1.png)

#### Zero-Shot Commonsense Reasoning

![Table 2](asserts/table-2.png)

#### Methods With Extra Inference-Time Cost

![Table 3](asserts/table-3.png)

#### MoE Model Results

![Table 4](asserts/table-4.png)

#### Math Resoning and Code Generation

![Table 5](asserts/table-5.png)

## Model Zoo

The following checkpoints are planned for public release on Hugging Face:

| Model | Quantization | Hugging Face |
| --- | --- | --- |
| Llama2-13B | W4A4 | [SliderQuant-Llama2-13B-W4A4](https://huggingface.co/IntelLabsChina/SliderQuant/blob/main/llama2-13b-w4a4-slider_parameters.pth) |
| Llama2-13B | W2A16 | [SliderQuant-Llama2-13B-W2A16](https://huggingface.co/IntelLabsChina/SliderQuant/blob/main/llama2-13b-w2a16-slider_parameters.pth) |
| Qwen2.5-14B | W4A4 | [SliderQuant-Qwen2.5-14B-W4A4](https://huggingface.co/IntelLabsChina/SliderQuant/blob/main/qwen2.5-14b-w4a4-slider_parameters.pth) |
| Qwen2.5-14B | W2A16 | [SliderQuant-Qwen2.5-14B-W2A16](https://huggingface.co/IntelLabsChina/SliderQuant/blob/main/qwen2.5-14b-w2a16-slider_parameters.pth) |

All checkpoints are available under [IntelLabsChina/SliderQuant](https://huggingface.co/IntelLabsChina/SliderQuant).

## Install

```bash
git clone https://github.com/genggng/sliderquant

mamba create -n sliderquant python=3.10 -y
mamba activate sliderquant

cd sliderquant
pip install -e .

```

## How To Train

1. Create a folder and place the experimental configuration file inside, following this structure:

```text
sliderquant/
├── log-llama2
│   └── llama2-w4a4
│       └── config.yaml
```

1. Edit `task_list.conf` to specify the `result_dir`.

```bash
result_dir=configs/llama2-7b-w2a16

result_dir=${exp_id}
GPU_NUM=1
port=29507
THRESHOLD=0.05
WAIT_MODE=true
WAIT_INTERVAL=60
```

1. Start training:

```bash
./auto_train_ddp.sh
```

## How To Test

1. Edit `task_list.conf` to specify the `result_dir`.

```bash
result_dir=configs/llama2-7b-w2a16

GPU_NUM=1
port=29507
THRESHOLD=0.05
WAIT_MODE=true
WAIT_INTERVAL=60
```

1. Run evaluation:

```bash
./auto_test_one.sh
```

## Citation

If SliderQuant is useful in your research, please cite:

```bibtex
@inproceedings{wang2026sliderquant,
  title={SliderQuant: Accurate Post-Training Quantization for LLMs},
  author={Wang, Shigeng and Li, Chao and Kang, Yangyuxuan and Fan, Jiawei and Ou, Zhonghong and Yao, Anbang},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## Acknowledgement

SliderQuant builds code from:

- [OmniQuant](https://github.com/OpenGVLab/OmniQuant)
- [QuaRot](https://github.com/spcl/QuaRot)

Thank you to the authors and maintainers of both projects for making their code public.
