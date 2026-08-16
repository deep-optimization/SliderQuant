#!/usr/bin/env python3
"""Render a parameterized Volcano manifest for one ScaleQ stage."""

import os
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

import yaml


TEMPLATE = Path(__file__).with_name("job_template.yaml")
STAGES = ("ayot", "baseline", "smoke", "pilot", "full", "eval")


def required(name):
    value = os.environ.get(name)
    assert value, f"{name} is required"
    return value


def runtime_settings():
    settings = {
        "namespace": required("SCALEQ_NAMESPACE"),
        "queue": required("SCALEQ_QUEUE"),
        "pvc": required("SCALEQ_PVC"),
        "mount_path": required("SCALEQ_MOUNT_PATH"),
        "shared_repo": required("SCALEQ_SHARED_REPO"),
        "root": required("SCALEQ_ROOT"),
        "work_repo": required("SCALEQ_WORK_REPO"),
        "image": required("SCALEQ_CONTAINER_IMAGE"),
        "job_prefix": required("SCALEQ_JOB_PREFIX"),
        "run_id": required("SCALEQ_RUN_ID"),
        "priority_class": required("SCALEQ_PRIORITY_CLASS"),
        "cpu": required("SCALEQ_CPU"),
        "memory": required("SCALEQ_MEMORY"),
        "gpu_count": required("SCALEQ_GPU_COUNT"),
        "shm_size": required("SCALEQ_SHM_SIZE"),
        "deadline": int(required("SCALEQ_DEADLINE_SECONDS")),
        "eval_batch_size": required("SCALEQ_EVAL_BATCH_SIZE"),
    }
    root = PurePosixPath(settings["root"])
    settings["model_dir"] = os.environ.get(
        "SCALEQ_MODEL_DIR",
        str(root / "models" / "Qwen3-1.7B"),
    )
    settings["ayot_dir"] = os.environ.get(
        "SCALEQ_AYOT_DIR",
        str(root / "ayot" / settings["run_id"]),
    )
    return settings


def validate_runtime_settings(settings):
    paths = {
        key: PurePosixPath(settings[key])
        for key in (
            "mount_path",
            "shared_repo",
            "root",
            "work_repo",
            "model_dir",
            "ayot_dir",
        )
    }
    for key, path in paths.items():
        assert path.is_absolute(), f"{key} must be an absolute container path"
        assert ".." not in path.parts, f"{key} must not contain parent traversal"

    mount_path = paths["mount_path"]
    root = paths["root"]
    shared_repo = paths["shared_repo"]
    work_repo = paths["work_repo"]
    model_dir = paths["model_dir"]
    ayot_dir = paths["ayot_dir"]
    assert root == mount_path or mount_path in root.parents, (
        "root must be contained by mount_path"
    )
    assert root != work_repo, "root and work_repo must differ"
    assert root not in work_repo.parents and work_repo not in root.parents, (
        "root and work_repo must not overlap"
    )
    assert shared_repo != work_repo, "shared_repo and work_repo must differ"
    assert (
        shared_repo not in work_repo.parents
        and work_repo not in shared_repo.parents
    ), "shared_repo and work_repo must not overlap"
    assert root in model_dir.parents, "model_dir must be contained by root"
    assert root in ayot_dir.parents, "ayot_dir must be contained by root"
    assert int(settings["gpu_count"]) > 0
    assert int(settings["eval_batch_size"]) > 0
    assert int(settings["deadline"]) > 0


def command(stage, settings):
    shared_repo = shlex.quote(settings["shared_repo"])
    work_repo = shlex.quote(settings["work_repo"])
    commands = [
        "set -euo pipefail",
        'mkdir -p "$SCALEQ_ROOT/provenance"',
        f'exec > >(tee -a "$SCALEQ_ROOT/provenance/{stage}-$SCALEQ_CODE_SHA.log") 2>&1',
    ]
    commands.extend(
        [
            f'test "$(git -C {shared_repo} rev-parse HEAD)" = "$SCALEQ_CODE_SHA"',
            f"git clone --local --no-hardlinks {shared_repo} {work_repo}",
            f'git -C {work_repo} checkout --detach "$SCALEQ_CODE_SHA"',
            'export PIP_CACHE_DIR="$SCALEQ_ROOT/cache/pip"',
            f"python -m pip install --no-deps -e {work_repo}",
            f"python -m pip install -r {work_repo}/requirements-scaleq.txt",
            f"cd {work_repo}",
            "python -m compileall -q main.py datautils.py train_utils.py models quantize scripts cluster tests",
            "python -m unittest discover -s tests -v",
            f"bash {work_repo}/scripts/cluster/run_{stage}.sh",
        ]
    )
    return "\n".join(commands) + "\n"


def render(stage, source_sha, code_sha, settings):
    assert stage in STAGES, f"stage must be one of {STAGES}"
    validate_runtime_settings(settings)
    manifest = yaml.safe_load(TEMPLATE.read_text())
    name = f'{settings["job_prefix"]}-{stage}-{settings["run_id"]}'
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name)

    manifest["metadata"].update(name=name, namespace=settings["namespace"])
    manifest["spec"].update(
        queue=settings["queue"],
        priorityClassName=settings["priority_class"],
    )
    pod = manifest["spec"]["tasks"][0]["template"]
    pod["metadata"]["labels"]["app"] = name
    pod["spec"]["priorityClassName"] = settings["priority_class"]
    pod["spec"]["activeDeadlineSeconds"] = settings["deadline"]
    container = pod["spec"]["containers"][0]
    container["image"] = settings["image"]
    container["command"] = ["/bin/bash", "-lc", command(stage, settings)]
    resources = {
        "cpu": settings["cpu"],
        "memory": settings["memory"],
        "nvidia.com/gpu": settings["gpu_count"],
    }
    container["resources"] = {"limits": resources, "requests": resources.copy()}
    env = {item["name"]: item for item in container["env"]}
    env["SCALEQ_SHA"]["value"] = source_sha
    env["SCALEQ_CODE_SHA"]["value"] = code_sha
    env["SCALEQ_ROOT"]["value"] = settings["root"]
    container["env"].extend(
        [
            {"name": "SCALEQ_STAGE", "value": stage},
            {"name": "SCALEQ_RUN_ID", "value": settings["run_id"]},
            {"name": "SCALEQ_GPU_COUNT", "value": settings["gpu_count"]},
            {
                "name": "SCALEQ_EVAL_BATCH_SIZE",
                "value": settings["eval_batch_size"],
            },
            {"name": "SCALEQ_MODEL_DIR", "value": settings["model_dir"]},
            {"name": "SCALEQ_AYOT_DIR", "value": settings["ayot_dir"]},
        ]
    )
    container["volumeMounts"][1]["mountPath"] = settings["mount_path"]
    pod["spec"]["volumes"][0]["emptyDir"]["sizeLimit"] = settings["shm_size"]
    pod["spec"]["volumes"][1]["persistentVolumeClaim"]["claimName"] = settings["pvc"]
    return yaml.safe_dump(manifest, sort_keys=False)


def main():
    (stage,) = sys.argv[1:]
    source_sha = required("SCALEQ_SHA")
    code_sha = required("SCALEQ_CODE_SHA")
    assert re.fullmatch(r"[0-9a-f]{40}", source_sha)
    assert re.fullmatch(r"[0-9a-f]{40}", code_sha)
    sys.stdout.write(render(stage, source_sha, code_sha, runtime_settings()))


if __name__ == "__main__":
    main()
