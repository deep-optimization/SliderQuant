<div align="center">
  <h1>SliderQuant: Accurate Post-Training Quantization for LLMs</h1>
  <p>Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, Zhonghong Ou, and Anbang Yao</p>
  <p>
    <a href="https://deep-optimization.github.io/sliderquant/"><img src="https://img.shields.io/badge/Project%20Page-SliderQuant-lightblue?logo=github" alt="Project Page"></a>
    <a href="https://arxiv.org/abs/2603.25284"><img src="https://img.shields.io/badge/arXiv-2603.25284-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
    <a href="https://openreview.net/forum?id=YNqZqw4fLT"><img src="https://img.shields.io/badge/OpenReview-Discussion-8A2BE2?logo=OpenReview" alt="OpenReview"></a>
    <a href="https://huggingface.co/IntelLabsChina/SliderQuant"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IntelLabsChina%2FSliderQuant-yellow" alt="Hugging Face"></a>
  </p>
</div>

---

This codebase extends the official PyTorch implementation of "SliderQuant:
Accurate Post-Training Quantization for LLMs" with a Qwen3 integration that
combines Attend to Your Own Thoughts (AYOT) calibration and public CAT-Q
ternary refinement.

### 📢 News
* **[March 2026]** 🎉 We release the official codebase and model checkpoints of SliderQuant. See [Model Zoo](#model-zoo) for available weights.
* **[January 2026]** 🎉 Our paper SliderQuant: Accurate Post-Training Quantization for LLMs has been accepted to **ICLR 2026**.

![SliderQuant overview](asserts/main-fig.png)

SliderQuant (**Slid**ing-lay**er** **Quant**ization) is a new learnable post-training quantization framework for LLMs, which consists of two key components:

- Inter-layer sliding quantization couples three types of sliding window designs to address the varying quantization sensitivity of shallow, intermediate and deep layers of any pre-trained LLMs.
- Intra-layer sliding quantization quantizes layers inside the current slidning window in an incremental manner.

## Table Of Contents

- [Table Of Contents](#table-of-contents)
- [Main Results](#main-results)
    - [Language Generation](#language-generation)
    - [Zero-Shot Commonsense Reasoning](#zero-shot-commonsense-reasoning)
    - [Methods With Extra Inference-Time Cost](#methods-with-extra-inference-time-cost)
    - [MoE Model Results](#moe-model-results)
    - [Math Resoning and Code Generation](#math-resoning-and-code-generation)
- [Model Zoo](#model-zoo)
- [Install](#install)
- [How To Train](#how-to-train)
- [How To Test](#how-to-test)
- [ScaleQ and CAT-Q for Qwen3](#scaleq-and-cat-q-for-qwen3)
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
git clone https://github.com/deep-optimization/SliderQuant.git

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

2. Edit `task_list.conf` to specify the `result_dir`.

```bash
result_dir=configs/llama2-7b-w2a16

GPU_NUM=1
port=29507
THRESHOLD=0.05
WAIT_MODE=true
WAIT_INTERVAL=60
```

3. Start training:

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

2. Run evaluation:

```bash
./auto_test_one.sh
```

## ScaleQ and CAT-Q for Qwen3

The Qwen3 recipe adds the following components to SliderQuant:

- immutable AYOT calibration artifacts built from pinned public datasets;
- progressive CAT-Q ternarization with 128-weight groups;
- rank-64 LoRA refinement applied before hard ternary materialization;
- masked windowed optimization with strict checkpoint validation;
- seeded generation for public math and code benchmarks; and
- parameterized workflow entry points for local or scheduled execution.

Install PyTorch for the target CUDA environment, then install the source and
the ScaleQ-specific dependencies:

```bash
python -m pip install --no-deps -e .
python -m pip install -r requirements-scaleq.txt
```

Build a versioned AYOT artifact:

```bash
python scripts/build_ayot_calibration.py \
    --out-root artifacts/calibration \
    --version v1
```

Optimize Qwen3-1.7B with the public recipe:

```bash
export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:?set the local worker count}"

torchrun --standalone --nproc_per_node="$LOCAL_WORLD_SIZE" main.py \
    --config configs/scaleq-qwen3-1p7b/config.yaml \
    --model Qwen/Qwen3-1.7B \
    --calib_manifest artifacts/calibration/v1/manifest.json \
    --output_dir artifacts/runs/full-v1 \
    --use_ddp
```

For scheduled execution, `cluster/render_jobs.py` keeps deployment-specific
values outside the repository. Set the source and code revisions plus the
target scheduler, storage, and worktree values before rendering a stage:

```bash
export SCALEQ_SHA="$(git rev-parse HEAD)"
export SCALEQ_CODE_SHA="$SCALEQ_SHA"
: "${SCALEQ_NAMESPACE:?set the scheduler namespace}"
: "${SCALEQ_QUEUE:?set the scheduler queue}"
: "${SCALEQ_PVC:?set the artifact volume claim}"
: "${SCALEQ_MOUNT_PATH:?set the artifact mount path}"
: "${SCALEQ_SHARED_REPO:?set the shared source checkout}"
: "${SCALEQ_ROOT:?set the artifact root}"
: "${SCALEQ_WORK_REPO:?set the detached worktree path}"
: "${SCALEQ_CONTAINER_IMAGE:?set the container image}"
: "${SCALEQ_JOB_PREFIX:?set the job-name prefix}"
: "${SCALEQ_RUN_ID:?set the artifact identifier}"
: "${SCALEQ_PRIORITY_CLASS:?set the scheduler priority class}"
: "${SCALEQ_CPU:?set the CPU request}"
: "${SCALEQ_MEMORY:?set the memory request}"
: "${SCALEQ_GPU_COUNT:?set the accelerator request}"
: "${SCALEQ_SHM_SIZE:?set the shared-memory size}"
: "${SCALEQ_DEADLINE_SECONDS:?set the stage deadline}"
: "${SCALEQ_EVAL_BATCH_SIZE:?set the evaluation batch size}"

python cluster/render_jobs.py full > cluster/rendered-full.yaml
```

All configured runtime paths must be absolute container paths without parent
traversal. `SCALEQ_ROOT` must be contained by `SCALEQ_MOUNT_PATH` and must not
overlap `SCALEQ_WORK_REPO`; the shared checkout and detached worktree must also
be distinct and non-overlapping. Optional `SCALEQ_MODEL_DIR` and
`SCALEQ_AYOT_DIR` settings select alternate model and calibration directories
under `SCALEQ_ROOT`; the renderer validates and forwards both values.

The executable configuration is in
`configs/scaleq-qwen3-1p7b/config.yaml`. Method choices that are not defined
by the public papers are recorded in
`configs/scaleq-qwen3-1p7b/assumptions.yaml`. Parameters without a public
definition are not inferred.

### Public foundations

- [SliderQuant](https://github.com/deep-optimization/SliderQuant) provides the
  sliding-layer quantization framework and upstream codebase.
- [Attend to Your Own Thoughts](https://arxiv.org/abs/2608.01078) introduces
  AYOT calibration and ScaleQ-1.58.
- [CAT-Q](https://arxiv.org/abs/2606.26650) introduces learnable modulation
  and softened ternarization for post-training ternary quantization.
- [BitTern CAT-Q](https://github.com/IntelChina-AI/BitTern/tree/main/projects/cat-q)
  provides the public CAT-Q configuration and checkpoint conventions.
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)
  provides the pinned base model and tokenizer.
- [EvalPlus](https://github.com/evalplus/evalplus) provides the public code
  benchmark datasets and execution-compatible sample format.
- [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) and
  [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
  provide the public AYOT question sources.

## Citation

If SliderQuant is useful in your research, please cite:

```bibtex
@inproceedings{wang2026sliderquant,
  title={SliderQuant: Accurate Post-Training Quantization for LLMs},
  author={Wang, Shigeng and Li, Chao and Kang, Yangyuxuan and Fan, Jiawei and Ou, Zhonghong and Yao, Anbang},
  booktitle={International Conference on Learning Representations},
  year={2026}
}

@article{wang2026ayot,
  title={Attend to Your Own Thoughts: Breaking the Barrier for Post-Training Quantization of Reasoning LLMs through the Lens of 1.58-Bit Quantization},
  author={Wang, Shigeng and Li, Chao and Kang, Yangyuxuan and Fan, Jiawei and Yao, Anbang},
  journal={arXiv preprint arXiv:2608.01078},
  year={2026}
}

@article{wang2026catq,
  title={CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs},
  author={Wang, Shigeng and Li, Chao and Kang, Yangyuxuan and Fan, Jiawei and Yao, Anbang},
  journal={arXiv preprint arXiv:2606.26650},
  year={2026}
}
```

## Acknowledgement

SliderQuant builds code from:

- [OmniQuant](https://github.com/OpenGVLab/OmniQuant)
- [QuaRot](https://github.com/spcl/QuaRot)

The ScaleQ/CAT-Q integration also builds on the public AYOT, CAT-Q, BitTern,
Qwen3, EvalPlus, MetaMathQA, and OpenCodeInstruct resources linked above.

We are grateful to the authors and maintainers of both projects for making their amazing code public.
