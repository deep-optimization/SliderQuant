import hashlib
import os
import random
import tempfile

import numpy as np
import torch


def atomic_torch_save(value, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(
        dir=directory, prefix="checkpoint-", suffix=".pt"
    )
    os.close(handle)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def window_checkpoint(layers, next_round, args):
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    with open(args.calib_manifest, "rb") as handle:
        calibration_hash = hashlib.sha256(handle.read()).hexdigest()
    return {
        "__format__": "scaleq_window_v1",
        "layers": layers,
        "next_round": next_round,
        "model": args.model,
        "model_revision": args.model_revision,
        "calib_manifest": args.calib_manifest,
        "calib_manifest_sha256": calibration_hash,
        "source_sha": os.environ.get("SCALEQ_SHA"),
        "code_sha": os.environ.get("SCALEQ_CODE_SHA"),
        "config": vars(args),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": cuda_rng,
        },
    }
