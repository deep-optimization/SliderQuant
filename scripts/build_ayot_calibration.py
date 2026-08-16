#!/usr/bin/env python
"""Build an immutable AYOT calibration artifact for ScaleQ Qwen3-1.7B.

  1. deterministically select 1024 MetaMathQA + 1024 OpenCodeInstruct questions
     from pinned dataset revisions and freeze them into ``selection.jsonl``;
  2. self-generate a response for every selected question with the *same* pinned
     Qwen3-1.7B snapshot that will later be quantized (chat template, thinking
     mode on, greedy decoding), sharded across torchrun ranks;
  3. tokenize question+response to exactly 2048 positions (right truncation,
     right padding) into safetensors shards;
  4. seal the versioned directory with SHA256SUMS and an atomically published
     manifest.json, plus balanced 32-row / 128-row subset manifests.

Use a distinct artifact version for each configuration. The builder is
restartable: completed per-row
generations are skipped, and manifest.json only appears after all 2048 rows
pass the acceptance checks. Selection never resamples: once selection.jsonl
exists it is consumed as-is.

    python scripts/build_ayot_calibration.py --out-root artifacts/calibration
    torchrun --standalone --nproc_per_node=4 scripts/build_ayot_calibration.py \
        --out-root artifacts/calibration

Dataset, model, and tokenizer defaults use immutable public revisions. Every
selected revision is recorded in the manifest.

Two settings are explicit recipe choices rather than published facts:
greedy decoding with thinking mode enabled (Qwen's own model card recommends
sampling for thinking mode), and right-side truncation of over-length
contexts. Both are recorded in generation-config.json / manifest.json so an
ablation can replace them under a new artifact version.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARTIFACT_PREFIX = "ayot-qwen3-1p7b"
FORMAT_VERSION = 1
PER_DOMAIN = 1024
SEQ_LEN = 2048
TOKENIZED_SHARDS = 8
SUBSET_SIZES = (32, 128)
UNHASHED_FILES = {"SHA256SUMS", "manifest.json"}

SELECTION_FILE = "selection.jsonl"
REJECTED_FILE = "selection-rejected.jsonl"
GENERATION_CONFIG_FILE = "generation-config.json"
MANIFEST_FILE = "manifest.json"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(text):
    """Canonical form used for cross-source duplicate detection."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def write_text_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl_atomic(path, records):
    write_text_atomic(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_shard(path):
    """Read an append-only shard, dropping and truncating a torn trailing line."""
    if not path.exists():
        return []
    data = path.read_bytes()
    records, complete = [], 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        records.append(json.loads(line))
        complete += len(line)
    if complete != len(data):
        with path.open("rb+") as handle:
            handle.truncate(complete)
    return records


def completed_sample_ids(records):
    return {
        record["sample_id"]
        for record in records
        if not record.get("failed", False)
    }


# --------------------------------------------------------------------------- #
# stage 1: selection
# --------------------------------------------------------------------------- #

def select_domain(
    dataset,
    domain,
    source,
    seed,
    target,
    seen_hashes,
    excluded_hashes=frozenset(),
):
    """Walk a seeded shuffle of row indexes until ``target`` rows are accepted.

    ``dataset`` only needs ``len()`` and integer indexing returning a mapping.
    ``seen_hashes`` is shared across domains so duplicates are removed within
    and across sources. Returns (accepted, rejected) records.
    """
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)

    accepted, rejected = [], []
    for row_index in order:
        if len(accepted) == target:
            break
        row = dataset[row_index]
        question = row[source["question_field"]]
        upstream_id = row[source["id_field"]] if source["id_field"] else None
        record = {
            "domain": domain,
            "dataset_repo": source["repo"],
            "dataset_revision": source["revision"],
            "dataset_config": source["config"],
            "split": source["split"],
            "row_index": row_index,
            "upstream_id": upstream_id,
            "question_field": source["question_field"],
            "selection_seed": seed,
        }
        normalized = normalize_question(question)
        if not normalized:
            rejected.append({**record, "rejection_reason": "empty_question"})
            continue
        norm_sha256 = sha256_text(normalized)
        if norm_sha256 in excluded_hashes:
            rejected.append(
                {
                    **record,
                    "rejection_reason": "evaluation_overlap",
                    "question_norm_sha256": norm_sha256,
                }
            )
            continue
        if norm_sha256 in seen_hashes:
            rejected.append({**record, "rejection_reason": "duplicate_question",
                             "question_norm_sha256": norm_sha256})
            continue
        seen_hashes.add(norm_sha256)
        accepted.append({
            **record,
            "sample_id": f"{domain}-{len(accepted):06d}",
            "question": question,
            "question_sha256": sha256_text(question),
            "question_norm_sha256": norm_sha256,
        })

    assert len(accepted) == target, (
        f"{domain}: only {len(accepted)} of {target} rows accepted from {len(dataset)} candidates"
    )
    return accepted, rejected


def build_selection(out_dir, args):
    from datasets import DownloadConfig, load_dataset

    seen_hashes = set()
    download_config = DownloadConfig(num_proc=1, max_retries=10)
    excluded_hashes = evaluation_question_hashes(download_config)
    accepted, rejected = [], []
    for domain, source in sources(args).items():
        print(f"[selection] loading {source['repo']}@{source['revision'][:12]}", flush=True)
        dataset = load_dataset(
            source["repo"],
            source["config"],
            split=source["split"],
            revision=source["revision"],
            download_config=download_config,
        )
        rows, drops = select_domain(
            dataset,
            domain,
            source,
            args.seed,
            PER_DOMAIN,
            seen_hashes,
            excluded_hashes,
        )
        print(f"[selection] {domain}: {len(rows)} accepted, {len(drops)} rejected", flush=True)
        accepted += rows
        rejected += drops

    write_jsonl_atomic(out_dir / REJECTED_FILE, rejected)
    write_jsonl_atomic(out_dir / SELECTION_FILE, accepted)


def evaluation_question_hashes(download_config):
    from datasets import load_dataset
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    sources = (
        (
            "HuggingFaceH4/MATH-500",
            "default",
            "test",
            "problem",
            "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        ),
        (
            "openai/gsm8k",
            "main",
            "test",
            "question",
            "740312add88f781978c0658806c59bc2815b9866",
        ),
        (
            "KbsdJames/Omni-MATH",
            "default",
            "test",
            "problem",
            "40ba231d8f16e29ecd40e6407e2c8640145a8f62",
        ),
    )
    questions = []
    for repo, config, split, field, revision in sources:
        dataset = load_dataset(
            repo,
            config,
            split=split,
            revision=revision,
            download_config=download_config,
        )
        questions.extend(dataset[field])
    questions.extend(row["prompt"] for row in get_human_eval_plus().values())
    questions.extend(row["prompt"] for row in get_mbpp_plus().values())
    return {sha256_text(normalize_question(question)) for question in questions}


def sources(args):
    return {
        "math": {
            "repo": args.math_repo, "revision": args.math_revision,
            "config": args.math_config, "split": args.math_split,
            "question_field": args.math_question_field, "id_field": args.math_id_field,
            "license": "mit",
        },
        "code": {
            "repo": args.code_repo, "revision": args.code_revision,
            "config": args.code_config, "split": args.code_split,
            "question_field": args.code_question_field, "id_field": args.code_id_field,
            "license": "cc-by-4.0",
        },
    }


# --------------------------------------------------------------------------- #
# stage 2: self-generation
# --------------------------------------------------------------------------- #

def generation_config(args, tokenizer):
    import torch
    import transformers

    assert isinstance(tokenizer.chat_template, str), "pinned tokenizer exposes no chat template"
    return {
        "model_repo": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.model_revision,
        "chat_template_sha256": sha256_text(tokenizer.chat_template),
        "thinking_mode": True,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "source_sha": os.environ.get("SCALEQ_SHA"),
        "torch_dtype": "bfloat16",
        "attn_implementation": "eager",
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
    }


def decoding_controls(config):
    """Everything in generation-config.json that changes the produced tokens."""
    return {k: v for k, v in config.items() if k != "software"}


def load_tokenizer(args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)


def pad_token_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def generate_rows(out_dir, rows, tokenizer, args, rank, world, local_rank):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    shard_path = out_dir / f"generations-{rank:05d}-of-{world:05d}.jsonl"
    done = completed_sample_ids(load_shard(shard_path))
    pending = [row for i, row in enumerate(rows) if i % world == rank and row["sample_id"] not in done]
    print(f"[generate] rank {rank}: {len(pending)} pending, {len(done)} already complete", flush=True)
    if not pending:
        return

    device = f"cuda:{local_rank}"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    gpu = torch.cuda.get_device_name(local_rank)
    pad_id = pad_token_id(tokenizer)

    # ponytail: batch size 1 keeps greedy decoding independent of padding;
    # add left-padded batching if wall-clock becomes the bottleneck.
    with shard_path.open("a", encoding="utf-8") as handle:
        for row in tqdm(pending, desc=f"rank{rank}", disable=rank != 0):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            prompt_ids = encoded["input_ids"][0].tolist()
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None, pad_token_id=pad_id,
                )
            generated_ids = output[0, len(prompt_ids):].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            hit_limit = len(generated_ids) >= args.max_new_tokens
            failed = not generated_text.strip()
            handle.write(json.dumps({
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "question": row["question"],
                "question_sha256": row["question_sha256"],
                "prompt_text": prompt,
                "prompt_sha256": sha256_text(prompt),
                "prompt_ids": prompt_ids,
                "generated_ids": generated_ids,
                "generated_text": generated_text,
                "generated_text_sha256": sha256_text(generated_text),
                "generated_tokens": len(generated_ids),
                "finish_reason": "length" if hit_limit else "stop",
                "truncated_generation": hit_limit,
                "failed": failed,
                "failure_reason": "empty_generation" if failed else None,
                "rank": rank,
                "gpu": gpu,
            }, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def combine_generations(out_dir, rows):
    """Rank-0 merge: selection order wins, so the result never depends on shard layout."""
    merged = {}
    for shard in sorted(out_dir.glob("generations-*.jsonl")):
        for record in load_shard(shard):
            merged[record["sample_id"]] = record
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in merged]
    assert not missing, f"{len(missing)} rows never generated, first: {missing[0]}"
    return [merged[row["sample_id"]] for row in rows]


# --------------------------------------------------------------------------- #
# stage 3: fixed-width tokenization
# --------------------------------------------------------------------------- #

def encode_fixed_length(tokenizer, text, width, pad_id):
    """Right-truncate then right-pad to exactly ``width`` positions."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    length = min(len(ids), width)
    truncated = len(ids) > width
    input_ids = list(ids[:width]) + [pad_id] * (width - length)
    attention_mask = [1] * length + [0] * (width - length)
    return input_ids, attention_mask, length, truncated


def write_tokenized_shards(out_dir, records, tokenizer):
    import numpy as np
    from safetensors.numpy import save_file

    pad_id = pad_token_id(tokenizer)
    per_shard = len(records) // TOKENIZED_SHARDS
    assert per_shard * TOKENIZED_SHARDS == len(records), "row count must divide the shard count"

    shards, placements, truncated_contexts = [], [], 0
    for shard_index in range(TOKENIZED_SHARDS):
        chunk = records[shard_index * per_shard:(shard_index + 1) * per_shard]
        name = f"tokenized-{shard_index:05d}-of-{TOKENIZED_SHARDS:05d}.safetensors"
        input_ids, masks, lengths, flags = [], [], [], []
        for shard_row, record in enumerate(chunk):
            ids, mask, length, truncated = encode_fixed_length(
                tokenizer, record["prompt_text"] + record["generated_text"], SEQ_LEN, pad_id
            )
            input_ids.append(ids)
            masks.append(mask)
            lengths.append(length)
            flags.append(truncated)
            truncated_contexts += truncated
            placements.append({
                "sample_id": record["sample_id"], "domain": record["domain"],
                "global_row": shard_index * per_shard + shard_row,
                "shard": name, "shard_row": shard_row,
            })
        save_file(
            {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                "attention_mask": np.asarray(masks, dtype=np.int8),
                "length": np.asarray(lengths, dtype=np.int32),
                "truncated": np.asarray(flags, dtype=np.int8),
            },
            out_dir / name,
            metadata={
                "sample_ids": json.dumps([r["sample_id"] for r in chunk]),
                "domains": json.dumps([r["domain"] for r in chunk]),
                "seq_len": str(SEQ_LEN),
                "pad_token_id": str(pad_id),
            },
        )
        shards.append({"file": name, "rows": len(chunk)})
    return shards, placements, truncated_contexts


# --------------------------------------------------------------------------- #
# stage 4: subsets, checksums, manifest
# --------------------------------------------------------------------------- #

def write_subsets(out_dir, artifact_id, placements):
    """Derive balanced subsets from the sealed order instead of resampling."""
    by_domain = {"math": [p for p in placements if p["domain"] == "math"],
                 "code": [p for p in placements if p["domain"] == "code"]}
    subsets = {}
    for size in SUBSET_SIZES:
        half = size // 2
        chosen = by_domain["math"][:half] + by_domain["code"][:half]
        assert len(chosen) == size, f"not enough rows for the {size}-row subset"
        name = f"subset-{size}.json"
        write_text_atomic(out_dir / name, json.dumps({
            "artifact_id": artifact_id,
            "rows": size,
            "math_rows": half,
            "code_rows": half,
            "seq_len": SEQ_LEN,
            "samples": sorted(chosen, key=lambda p: p["global_row"]),
        }, indent=2, sort_keys=True) + "\n")
        subsets[str(size)] = name
    return subsets


def write_sha256sums(out_dir):
    names = sorted(
        p.name for p in out_dir.iterdir()
        if p.is_file() and p.name not in UNHASHED_FILES and not p.name.endswith(".tmp")
    )
    checksums = {name: sha256_file(out_dir / name) for name in names}
    write_text_atomic(out_dir / "SHA256SUMS", "".join(f"{checksums[n]}  {n}\n" for n in names))
    return checksums


def publish_manifest(out_dir, manifest):
    tmp = out_dir / (MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_dir / MANIFEST_FILE)


def finalize(out_dir, artifact_id, rows, args, tokenizer):
    records = combine_generations(out_dir, rows)
    failures = [r["sample_id"] for r in records if r["failed"]]
    assert not failures, f"{len(failures)} failed generations, first: {failures[0]}"

    shards, placements, truncated_contexts = write_tokenized_shards(out_dir, records, tokenizer)
    subsets = write_subsets(out_dir, artifact_id, placements)

    domains = [row["domain"] for row in rows]
    acceptance = {
        "rows": len(rows),
        "math_rows": domains.count("math"),
        "code_rows": domains.count("code"),
        "seq_len": SEQ_LEN,
        "nominal_calibration_tokens": len(rows) * SEQ_LEN,
        "duplicate_question_hashes": len(rows) - len({r["question_norm_sha256"] for r in rows}),
        "generation_failures": 0,
        "evaluation_overlap_exact": 0,
        "evaluation_overlap_check": "normalized_exact_hash",
    }
    assert acceptance["rows"] == 2 * PER_DOMAIN
    assert acceptance["math_rows"] == PER_DOMAIN
    assert acceptance["code_rows"] == PER_DOMAIN
    assert acceptance["duplicate_question_hashes"] == 0
    assert acceptance["nominal_calibration_tokens"] == 4_194_304

    checksums = write_sha256sums(out_dir)
    publish_manifest(out_dir, {
        "format_version": FORMAT_VERSION,
        "artifact_id": artifact_id,
        "source_sha": os.environ.get("SCALEQ_SHA"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"repo": args.model, "revision": args.model_revision,
                  "pad_token_id": pad_token_id(tokenizer)},
        "sources": sources(args),
        "selection": {"file": SELECTION_FILE, "rejected_file": REJECTED_FILE,
                      "seed": args.seed, "per_domain": PER_DOMAIN, "resampling": "disabled"},
        "generation": {
            "config_file": GENERATION_CONFIG_FILE,
            "shards": sorted(p.name for p in out_dir.glob("generations-*.jsonl")),
            "truncated_generations": sum(r["truncated_generation"] for r in records),
        },
        "tokenized": {
            "shards": shards, "rows": len(records), "seq_len": SEQ_LEN,
            "pad_token_id": pad_token_id(tokenizer),
            "truncation_side": "right", "padding_side": "right",
            "truncated_contexts": truncated_contexts,
            "dtypes": {"input_ids": "int32", "attention_mask": "int8"},
        },
        "subsets": subsets,
        "acceptance": acceptance,
        "sha256sums_file": "SHA256SUMS",
        "sha256sums_sha256": sha256_file(out_dir / "SHA256SUMS"),
        "files": checksums,
    })


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    return rank, world, local_rank


def barrier(world):
    if world > 1:
        import torch.distributed as dist

        dist.barrier()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--version", default="v1")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--math-repo", default="meta-math/MetaMathQA")
    parser.add_argument("--math-revision", default="aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18")
    parser.add_argument("--math-config", default="default")
    parser.add_argument("--math-split", default="train")
    parser.add_argument("--math-question-field", default="query")
    parser.add_argument("--math-id-field", default="", help="MetaMathQA has no stable row ID")
    parser.add_argument("--code-repo", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--code-revision", default="8f3ba5bafe4d6e8db46082cf7ae6741bc370604d")
    parser.add_argument("--code-config", default="train")
    parser.add_argument("--code-split", default="train")
    parser.add_argument("--code-question-field", default="input")
    parser.add_argument("--code-id-field", default="id")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    artifact_id = f"{ARTIFACT_PREFIX}-{args.version}"
    out_dir = args.out_root / args.version
    rank, world, local_rank = init_distributed()

    manifest_path = out_dir / MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        source_sha = os.environ.get("SCALEQ_SHA")
        if source_sha is not None:
            assert manifest["source_sha"] == source_sha, (
                "published AYOT artifact belongs to a different source SHA; "
                "use a new --version"
            )
        print(f"[done] {manifest_path} already published; artifacts are immutable", flush=True)
        if world > 1:
            import torch.distributed as dist

            dist.destroy_process_group()
        return

    tokenizer = load_tokenizer(args)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / SELECTION_FILE).exists():
            build_selection(out_dir, args)
        config = generation_config(args, tokenizer)
        config_path = out_dir / GENERATION_CONFIG_FILE
        if config_path.exists():
            recorded = json.loads(config_path.read_text())
            assert decoding_controls(recorded) == decoding_controls(config), (
                "resumed run changes the recorded generation controls; start a new --version instead"
            )
        else:
            write_text_atomic(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    barrier(world)

    rows = read_jsonl(out_dir / SELECTION_FILE)
    generate_rows(out_dir, rows, tokenizer, args, rank, world, local_rank)
    barrier(world)

    if rank == 0:
        finalize(out_dir, artifact_id, rows, args, tokenizer)
        print(f"[done] published {out_dir / MANIFEST_FILE}", flush=True)
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
