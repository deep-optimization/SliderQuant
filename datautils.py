import pdb
import hashlib
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from safetensors.torch import load_file

c4_path = os.environ.get("SLIDERQUANT_C4_PATH", "datasets_local/c4")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    print(f"get_wikitext2 from_start:False")
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    print(f"wikitext train total tokens:{trainenc.input_ids.shape[1]}")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model,from_start=False):
    print(f"get_c4 from_start:{from_start}")
    # import ipdb;ipdb.set_trace()

    traindata = load_dataset(
            c4_path, data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )


    valdata = load_dataset(
       c4_path, data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        if from_start is True:
            i = 0
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc


def get_ayot(manifest_path, nsamples, seqlen, subset_path=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checksums_path = manifest_path.parent / manifest["sha256sums_file"]
    assert sha256_file(checksums_path) == manifest["sha256sums_sha256"]
    checksums = {
        name: digest
        for digest, name in (
            line.split("  ", 1) for line in checksums_path.read_text().splitlines()
        )
    }
    assert checksums == manifest["files"]

    tokenized = manifest["tokenized"]
    assert tokenized["seq_len"] == seqlen
    assert nsamples <= tokenized["rows"]

    input_ids = []
    attention_masks = []
    for shard_info in tokenized["shards"]:
        relative_path = shard_info["file"]
        shard_path = manifest_path.parent / relative_path
        assert sha256_file(shard_path) == checksums[relative_path]
        shard = load_file(shard_path)
        input_ids.append(shard["input_ids"])
        attention_masks.append(shard["attention_mask"])

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    if subset_path is not None:
        subset_path = Path(subset_path)
        assert sha256_file(subset_path) == checksums[subset_path.name]
        subset = json.loads(subset_path.read_text())
        assert subset["artifact_id"] == manifest["artifact_id"]
        assert subset["seq_len"] == seqlen
        assert subset["rows"] == nsamples == len(subset["samples"])
        indexes = torch.tensor(
            [sample["global_row"] for sample in subset["samples"]], dtype=torch.long
        )
        assert len(indexes.unique()) == nsamples
        assert indexes.min() >= 0 and indexes.max() < tokenized["rows"]
        input_ids = input_ids[indexes]
        attention_masks = attention_masks[indexes]
    else:
        input_ids = input_ids[:nsamples]
        attention_masks = attention_masks[:nsamples]
    assert input_ids.shape == attention_masks.shape == (nsamples, seqlen)

    trainloader = []
    for ids, mask in zip(input_ids, attention_masks):
        ids = ids.unsqueeze(0)
        mask = mask.unsqueeze(0)
        target = ids.clone()
        target[:, :-1] = -100
        trainloader.append((ids, target, mask))
    return trainloader, None



def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='',args=None
):
    if name == "ayot":
        assert args is not None and args.calib_manifest is not None
        return get_ayot(
            args.calib_manifest,
            nsamples,
            seqlen,
            subset_path=args.calib_subset,
        )

    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)

    if 'c4' in name:
        if "start" in name:
            from_start = True
        else:
            from_start = False
        return get_c4(nsamples, seed, seqlen, model,from_start=from_start)
