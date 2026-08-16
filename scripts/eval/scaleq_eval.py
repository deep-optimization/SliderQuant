#!/usr/bin/env python3
import argparse
import json
import os
import re
import unicodedata
import uuid
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASETS = {
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "config": "default",
        "split": "test",
        "question": "problem",
        "answer": "answer",
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "config": "main",
        "split": "test",
        "question": "question",
        "answer": "answer",
    },
    "omnimath": {
        "repo": "KbsdJames/Omni-MATH",
        "revision": "40ba231d8f16e29ecd40e6407e2c8640145a8f62",
        "config": "default",
        "split": "test",
        "question": "problem",
        "answer": "answer",
    },
}


def write_once(path, text):
    if path.exists():
        assert path.read_text() == text
        return
    path.write_text(text)


def read_jsonl(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    records = []
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        records.append(json.loads(line))
    return records


def balanced_braced(text, start):
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return None


def extract_answer(text):
    boxed = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(boxed):
        answer = balanced_braced(text, match.end() - 1)
        if answer is not None:
            return answer
    final = re.findall(
        r"(?:final answer(?: is)?|answer is)\s*[:=]?\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if final:
        return final[-1].splitlines()[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(answer):
    answer = unicodedata.normalize("NFKC", answer)
    answer = answer.replace("$", "").replace(",", "")
    answer = re.sub(r"\\(?:left|right)", "", answer)
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\s+", "", answer).strip(".")
    return answer.casefold()


def numeric_equal(left, right):
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def score_answer(generated, reference):
    prediction = normalize_answer(extract_answer(generated))
    if "####" in reference:
        reference = reference.rsplit("####", 1)[1]
    target = normalize_answer(reference)
    return prediction, target, prediction == target or numeric_equal(prediction, target)


def strip_code_fence(text):
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    return (fenced[-1] if fenced else text).strip()


def assigned(index, rank, world):
    return index % world == rank


def math_examples(task):
    spec = DATASETS[task]
    dataset = load_dataset(
        spec["repo"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
    )
    for index, row in enumerate(dataset):
        yield {
            "task_id": f"{task}-{index:06d}",
            "question": row[spec["question"]],
            "reference": row[spec["answer"]],
        }


def code_examples(task):
    if task == "humaneval":
        from evalplus.data import get_human_eval_plus

        examples = get_human_eval_plus()
    else:
        from evalplus.data import get_mbpp_plus

        examples = get_mbpp_plus()
    for task_id, row in sorted(examples.items()):
        yield {
            "task_id": task_id,
            "question": row["prompt"],
            "reference": None,
        }


def prefetch_code_datasets(tasks, rank, world):
    if world == 1:
        return
    if rank == 0:
        for task in tasks:
            if task not in DATASETS:
                list(code_examples(task))
    dist.barrier()


def prompt_for(task, question):
    if task in DATASETS:
        return (
            "Solve the following problem carefully. Put only the final answer "
            f"inside \\boxed{{}}.\n\n{question}"
        )
    return (
        "Complete the following Python task. Return only the Python code that "
        f"should follow the provided prefix.\n\n{question}"
    )


def trim_at_eos(token_ids, eos_token_id):
    if eos_token_id is None:
        return token_ids
    eos_token_ids = (
        {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
    )
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return token_ids[: index + 1]
    return token_ids


def generate(model, tokenizer, prompts, args, seed):
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        for prompt in prompts
    ]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)
    torch.manual_seed(seed)
    output = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        top_k=args.top_k if args.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_rows = output[:, encoded["input_ids"].shape[1]:].tolist()
    results = []
    for index, generated_ids in enumerate(generated_rows):
        generated_ids = trim_at_eos(
            generated_ids, model.generation_config.eos_token_id
        )
        prompt_ids = encoded["input_ids"][index][
            encoded["attention_mask"][index].bool()
        ].tolist()
        results.append(
            {
                "rendered_prompt": rendered[index],
                "prompt_ids": prompt_ids,
                "generated_ids": generated_ids,
                "generated_text": tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ),
            }
        )
    return results


def run_task(task, model, tokenizer, args, rank, world):
    output_dir = args.output_dir / task
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = list(math_examples(task) if task in DATASETS else code_examples(task))
    rank_examples = [
        (index, example)
        for index, example in enumerate(examples)
        if assigned(index, rank, world)
    ]
    existing = set()
    shard_glob = f"rank-{rank:02d}-of-{world:02d}*.jsonl"
    for prior_shard in sorted(output_dir.glob(shard_glob)):
        existing.update(record["task_id"] for record in read_jsonl(prior_shard))
    if all(example["task_id"] in existing for _, example in rank_examples):
        return len(examples)

    shard = output_dir / (
        f"rank-{rank:02d}-of-{world:02d}-{uuid.uuid4().hex}.jsonl"
    )
    with shard.open("a") as handle:
        for start in range(0, len(rank_examples), args.batch_size):
            batch = rank_examples[start : start + args.batch_size]
            if all(example["task_id"] in existing for _, example in batch):
                continue
            generated_batch = generate(
                model,
                tokenizer,
                [prompt_for(task, example["question"]) for _, example in batch],
                args,
                args.seed + batch[0][0],
            )
            for (_, example), generated in zip(batch, generated_batch):
                if example["task_id"] in existing:
                    continue
                record = {**example, **generated}
                if task in DATASETS:
                    prediction, target, correct = score_answer(
                        generated["generated_text"], example["reference"]
                    )
                    record.update(
                        prediction=prediction,
                        normalized_reference=target,
                        exact_or_numeric_match=correct,
                        score_status=(
                            "judge_pending" if task == "omnimath" else "scored"
                        ),
                    )
                else:
                    record["code"] = strip_code_fence(generated["generated_text"])
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                existing.add(example["task_id"])
    return len(examples)


def finalize_task(task, output_dir, expected_rows):
    task_dir = output_dir / task
    records = []
    for shard in sorted(task_dir.glob("rank-*.jsonl")):
        records.extend(read_jsonl(shard))
    records.sort(key=lambda record: record["task_id"])
    assert len(records) == expected_rows
    assert len({record["task_id"] for record in records}) == expected_rows

    if task in DATASETS:
        scored = [record for record in records if record["score_status"] == "scored"]
        aggregate = {
            "task": task,
            "rows": len(records),
            "scored_rows": len(scored),
            "accuracy": (
                sum(record["exact_or_numeric_match"] for record in scored) / len(scored)
                if scored
                else None
            ),
            "note": (
                "Omni-MATH requires the separately pinned judge protocol."
                if task == "omnimath"
                else "Conservative exact-or-numeric normalization."
            ),
        }
        write_once(
            task_dir / "aggregate.json",
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        )
    else:
        samples = []
        for record in records:
            key = "completion" if task == "humaneval" else "solution"
            samples.append({"task_id": record["task_id"], key: record["code"]})
        write_once(
            task_dir / "evalplus-samples.jsonl",
            "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="math500,gsm8k,omnimath,humaneval,mbpp",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = args.tasks.split(",")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    prefetch_code_datasets(tasks, rank, world)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).to(f"cuda:{local_rank}").eval()

    config = {
        "model": args.model,
        "model_revision": args.model_revision,
        "tasks": tasks,
        "thinking": args.thinking,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "source_sha": os.environ.get("SCALEQ_SHA"),
        "world_size": world,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = args.output_dir / "generation-config.json"
        if config_path.exists():
            assert json.loads(config_path.read_text()) == config, (
                "existing evaluation shards use different generation controls"
            )
        else:
            write_once(
                config_path,
                json.dumps(config, indent=2, sort_keys=True) + "\n",
            )
    if world > 1:
        dist.barrier()

    for task in config["tasks"]:
        expected_rows = run_task(task, model, tokenizer, args, rank, world)
        if world > 1:
            dist.barrier()
        if rank == 0:
            finalize_task(task, args.output_dir, expected_rows)
        if world > 1:
            dist.barrier()

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
