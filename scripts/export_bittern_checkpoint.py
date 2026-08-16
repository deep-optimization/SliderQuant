#!/usr/bin/env python3
import argparse
from collections import OrderedDict

import torch

from quantize.checkpoint import atomic_torch_save


FACTOR_NAMES = {
    "raw_scale": "generate_scale_factor.bound_factor",
    "raw_mu": "generate_mu_factor.bound_factor",
    "raw_round": "generate_round_factor.bound_factor",
}


def convert_layer(state, require_lora=False):
    converted = OrderedDict()
    factor_names = {}
    lora_names = {}
    for name, value in state.items():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in FACTOR_NAMES:
            prefix = name[: -(len(suffix) + 1)]
            module = prefix.removesuffix(".weight_quantizer")
            factor_names.setdefault(module, set()).add(suffix)
            converted[f"{prefix}.{FACTOR_NAMES[suffix]}"] = (
                value.detach().cpu().clone()
            )
            continue

        prefix, kind, index = name.rsplit(".", 2)
        if kind not in {"lora_A", "lora_B"}:
            raise ValueError(f"unsupported checkpoint key: {name}")
        lora_names.setdefault(prefix, {}).setdefault(index, set()).add(kind)
        converted[name] = value.detach().cpu().clone()

    expected_factors = set(FACTOR_NAMES)
    for module, names in factor_names.items():
        if names != expected_factors:
            raise ValueError(
                f"incomplete CAT-Q factors for {module}: {sorted(names)}"
            )
    for module, adapters in lora_names.items():
        for index, names in adapters.items():
            if names != {"lora_A", "lora_B"}:
                raise ValueError(
                    f"incomplete LoRA adapter for {module}.{index}: "
                    f"{sorted(names)}"
                )
    if lora_names and set(lora_names) != set(factor_names):
        raise ValueError(
            "LoRA and CAT-Q modules differ: "
            f"lora={sorted(lora_names)}, factors={sorted(factor_names)}"
        )
    if require_lora and set(lora_names) != set(factor_names):
        raise ValueError("rank-64 CAT-Q export requires LoRA for every module")
    return converted


def convert_checkpoint(slider_parameters, require_lora=False):
    return {
        int(layer_id): convert_layer(
            layer_state,
            require_lora=require_lora,
        )
        for layer_id, layer_state in slider_parameters.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-lora", action="store_true")
    args = parser.parse_args()

    slider_parameters = torch.load(
        args.input,
        map_location="cpu",
        weights_only=True,
    )
    converted = convert_checkpoint(
        slider_parameters,
        require_lora=args.require_lora,
    )
    assert converted
    assert all(layer for layer in converted.values())
    atomic_torch_save(converted, args.output)


if __name__ == "__main__":
    main()
