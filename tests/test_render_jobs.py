import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from cluster.render_jobs import STAGES, command, render, runtime_settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RenderJobsTest(unittest.TestCase):
    def environment(self):
        return {
            "SCALEQ_NAMESPACE": "example-namespace",
            "SCALEQ_QUEUE": "example-queue",
            "SCALEQ_PVC": "example-volume",
            "SCALEQ_MOUNT_PATH": "/example/artifacts",
            "SCALEQ_SHARED_REPO": "/example/shared/source",
            "SCALEQ_ROOT": "/example/artifacts/scaleq",
            "SCALEQ_WORK_REPO": "/example/worktree",
            "SCALEQ_CONTAINER_IMAGE": "registry.example.invalid/scaleq:test",
            "SCALEQ_JOB_PREFIX": "scaleq",
            "SCALEQ_RUN_ID": "fixture",
            "SCALEQ_PRIORITY_CLASS": "example-priority",
            "SCALEQ_CPU": "3",
            "SCALEQ_MEMORY": "12Gi",
            "SCALEQ_GPU_COUNT": "2",
            "SCALEQ_SHM_SIZE": "4Gi",
            "SCALEQ_DEADLINE_SECONDS": "900",
            "SCALEQ_EVAL_BATCH_SIZE": "2",
        }

    def settings(self, **overrides):
        values = {
            "namespace": "example-namespace",
            "queue": "example-queue",
            "pvc": "example-volume",
            "mount_path": "/example/artifacts",
            "shared_repo": "/example/shared/source",
            "root": "/example/artifacts/scaleq",
            "work_repo": "/example/worktree",
            "model_dir": "/example/artifacts/scaleq/models/Qwen3-1.7B",
            "ayot_dir": "/example/artifacts/scaleq/ayot/fixture",
            "image": "registry.example.invalid/scaleq:test",
            "job_prefix": "scaleq",
            "run_id": "fixture",
            "priority_class": "example-priority",
            "cpu": "3",
            "memory": "12Gi",
            "gpu_count": "2",
            "shm_size": "4Gi",
            "deadline": 900,
            "eval_batch_size": "2",
        }
        values.update(overrides)
        return values

    def test_render_uses_portable_runtime_settings(self):
        manifest = yaml.safe_load(render("full", "a" * 40, "b" * 40, self.settings()))
        self.assertEqual(manifest["metadata"]["name"], "scaleq-full-fixture")
        self.assertEqual(manifest["metadata"]["namespace"], "example-namespace")
        self.assertEqual(manifest["spec"]["queue"], "example-queue")

        pod = manifest["spec"]["tasks"][0]["template"]["spec"]
        container = pod["containers"][0]
        self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], "2")
        self.assertEqual(
            pod["volumes"][1]["persistentVolumeClaim"]["claimName"],
            "example-volume",
        )
        self.assertEqual(
            container["volumeMounts"][1]["mountPath"],
            "/example/artifacts",
        )
        self.assertEqual(pod["volumes"][0]["emptyDir"]["sizeLimit"], "4Gi")

    def test_every_stage_forwards_revisions_and_artifact_paths(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                manifest = yaml.safe_load(
                    render(stage, "a" * 40, "b" * 40, self.settings())
                )
                container = manifest["spec"]["tasks"][0]["template"]["spec"][
                    "containers"
                ][0]
                env = {item["name"]: item["value"] for item in container["env"]}
                self.assertEqual(env["SCALEQ_SHA"], "a" * 40)
                self.assertEqual(env["SCALEQ_CODE_SHA"], "b" * 40)
                self.assertEqual(env["SCALEQ_STAGE"], stage)
                self.assertEqual(env["SCALEQ_RUN_ID"], "fixture")
                self.assertEqual(env["SCALEQ_EVAL_BATCH_SIZE"], "2")
                self.assertEqual(
                    env["SCALEQ_MODEL_DIR"],
                    "/example/artifacts/scaleq/models/Qwen3-1.7B",
                )
                self.assertEqual(
                    env["SCALEQ_AYOT_DIR"],
                    "/example/artifacts/scaleq/ayot/fixture",
                )

    def test_job_name_must_be_dns_compatible(self):
        with self.assertRaises(AssertionError):
            render("full", "a" * 40, "b" * 40, self.settings(job_prefix="ScaleQ"))

    def test_artifact_root_must_be_inside_the_mounted_volume(self):
        render(
            "full",
            "a" * 40,
            "b" * 40,
            self.settings(
                mount_path="/example/artifacts/scaleq",
                root="/example/artifacts/scaleq",
            ),
        )
        with self.assertRaisesRegex(AssertionError, "contained by mount_path"):
            render(
                "full",
                "a" * 40,
                "b" * 40,
                self.settings(root="/example/other"),
            )
        with self.assertRaisesRegex(AssertionError, "contained by mount_path"):
            render(
                "full",
                "a" * 40,
                "b" * 40,
                self.settings(mount_path="/example/artifacts/scaleq/mounted"),
            )

    def test_runtime_paths_must_not_contain_parent_traversal(self):
        for name in (
            "mount_path",
            "shared_repo",
            "root",
            "work_repo",
            "model_dir",
            "ayot_dir",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(AssertionError, "parent traversal"):
                    render(
                        "full",
                        "a" * 40,
                        "b" * 40,
                        self.settings(**{name: "/example/../outside"}),
                    )

    def test_runtime_paths_must_be_absolute(self):
        for name in (
            "mount_path",
            "shared_repo",
            "root",
            "work_repo",
            "model_dir",
            "ayot_dir",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(AssertionError, "absolute container path"):
                    render(
                        "full",
                        "a" * 40,
                        "b" * 40,
                        self.settings(**{name: "relative/path"}),
                    )

    def test_artifact_root_must_not_overlap_the_worktree(self):
        with self.assertRaisesRegex(AssertionError, "must not overlap"):
            render(
                "full",
                "a" * 40,
                "b" * 40,
                self.settings(work_repo="/example/artifacts/scaleq/worktree"),
            )
        with self.assertRaisesRegex(AssertionError, "must differ"):
            render(
                "full",
                "a" * 40,
                "b" * 40,
                self.settings(work_repo="/example/artifacts/scaleq"),
            )
        with self.assertRaisesRegex(AssertionError, "must not overlap"):
            render(
                "full",
                "a" * 40,
                "b" * 40,
                self.settings(work_repo="/example/artifacts"),
            )

    def test_shared_checkout_must_not_overlap_the_worktree(self):
        for shared_repo, work_repo in (
            ("/example/source", "/example/source"),
            ("/example/source", "/example/source/worktree"),
            ("/example/source/shared", "/example/source"),
        ):
            with self.subTest(shared_repo=shared_repo, work_repo=work_repo):
                with self.assertRaisesRegex(AssertionError, "must (differ|not overlap)"):
                    render(
                        "full",
                        "a" * 40,
                        "b" * 40,
                        self.settings(shared_repo=shared_repo, work_repo=work_repo),
                    )

    def test_model_and_ayot_overrides_must_remain_under_the_artifact_root(self):
        for name in ("model_dir", "ayot_dir"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(AssertionError, "contained by root"):
                    render(
                        "full",
                        "a" * 40,
                        "b" * 40,
                        self.settings(**{name: "/example/outside"}),
                    )
            with self.subTest(name=name, case="equal-to-root"):
                with self.assertRaisesRegex(AssertionError, "contained by root"):
                    render(
                        "full",
                        "a" * 40,
                        "b" * 40,
                        self.settings(**{name: "/example/artifacts/scaleq"}),
                    )
            with self.subTest(name=name, case="parent-of-root"):
                with self.assertRaisesRegex(AssertionError, "contained by root"):
                    render(
                        "full",
                        "a" * 40,
                        "b" * 40,
                        self.settings(**{name: "/example/artifacts"}),
                    )
            with self.subTest(name=name, case="parent-traversal"):
                with self.assertRaisesRegex(AssertionError, "parent traversal"):
                    render(
                        "full",
                        "a" * 40,
                        "b" * 40,
                        self.settings(
                            **{
                                name: "/example/artifacts/scaleq/inside/../../outside"
                            }
                        ),
                    )

    def test_runtime_settings_resolve_and_forward_artifact_overrides(self):
        with patch.dict(os.environ, self.environment(), clear=True):
            defaults = runtime_settings()
        self.assertEqual(
            defaults["model_dir"],
            "/example/artifacts/scaleq/models/Qwen3-1.7B",
        )
        self.assertEqual(
            defaults["ayot_dir"],
            "/example/artifacts/scaleq/ayot/fixture",
        )

        environment = self.environment()
        environment.update(
            SCALEQ_MODEL_DIR="/example/artifacts/scaleq/models/custom",
            SCALEQ_AYOT_DIR="/example/artifacts/scaleq/ayot/custom",
        )
        with patch.dict(os.environ, environment, clear=True):
            settings = runtime_settings()
        manifest = yaml.safe_load(render("ayot", "a" * 40, "b" * 40, settings))
        container = manifest["spec"]["tasks"][0]["template"]["spec"]["containers"][0]
        env = {item["name"]: item["value"] for item in container["env"]}
        self.assertEqual(
            env["SCALEQ_MODEL_DIR"],
            "/example/artifacts/scaleq/models/custom",
        )
        self.assertEqual(
            env["SCALEQ_AYOT_DIR"],
            "/example/artifacts/scaleq/ayot/custom",
        )

    def test_runtime_settings_require_every_deployment_value(self):
        environment = self.environment()
        for name in environment:
            incomplete = environment.copy()
            del incomplete[name]
            with self.subTest(name=name), patch.dict(os.environ, incomplete, clear=True):
                with self.assertRaisesRegex(AssertionError, f"{name} is required"):
                    runtime_settings()

    def test_all_runtime_outputs_are_contained_by_the_artifact_root(self):
        preflight = (REPOSITORY_ROOT / "scripts/cluster/preflight.sh").read_text()
        for expected in (
            'CACHE_DIR="$SCALEQ_ROOT/cache/scaleq"',
            'MODEL_DIR="$SCALEQ_MODEL_DIR"',
            'AYOT_DIR="$SCALEQ_AYOT_DIR"',
            'export HF_HOME="$SCALEQ_ROOT/cache/huggingface"',
            '"$SCALEQ_ROOT/models"',
            '"$SCALEQ_ROOT/environment/$SCALEQ_CODE_SHA"',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, preflight)
        self.assertNotIn('CACHE_DIR="$(dirname "$REPO_DIR")', preflight)

        rendered_command = command("full", self.settings())
        self.assertIn('mkdir -p "$SCALEQ_ROOT/provenance"', rendered_command)
        self.assertIn(
            '"$SCALEQ_ROOT/provenance/full-$SCALEQ_CODE_SHA.log"',
            rendered_command,
        )

        stage_expectations = {
            "run_ayot.sh": ('--out-root "$SCALEQ_ROOT/ayot"',),
            "run_smoke.sh": ('RUN_DIR="$SCALEQ_ROOT/runs/smoke-$SCALEQ_RUN_ID"',),
            "run_pilot.sh": ('RUN_DIR="$SCALEQ_ROOT/runs/pilot-$SCALEQ_RUN_ID"',),
            "run_full.sh": ('RUN_DIR="$SCALEQ_ROOT/runs/full-$SCALEQ_RUN_ID"',),
            "run_baseline.sh": (
                'RUN_DIR="$SCALEQ_ROOT/evaluation/baseline-$SCALEQ_RUN_ID"',
            ),
            "run_eval.sh": (
                'RUN_DIR="$SCALEQ_ROOT/evaluation/scaleq-$SCALEQ_RUN_ID"',
                'MERGED_ROOT="$SCALEQ_ROOT/merged-model/$SCALEQ_RUN_ID"',
            ),
        }
        for script, expectations in stage_expectations.items():
            content = (REPOSITORY_ROOT / "scripts/cluster" / script).read_text()
            for expected in expectations:
                with self.subTest(script=script, expected=expected):
                    self.assertIn(expected, content)

    def test_template_contains_placeholders_without_operational_fallbacks(self):
        template = (REPOSITORY_ROOT / "cluster/job_template.yaml").read_text()
        for placeholder in (
            "__SCALEQ_JOB_NAME__",
            "__SCALEQ_NAMESPACE__",
            "__SCALEQ_QUEUE__",
            "__SCALEQ_PRIORITY_CLASS__",
            "__SCALEQ_DEADLINE_SECONDS__",
            "__SCALEQ_CONTAINER_IMAGE__",
            "__SCALEQ_ROOT__",
            "__SCALEQ_MOUNT_PATH__",
            "__SCALEQ_SHM_SIZE__",
            "__SCALEQ_PVC__",
        ):
            with self.subTest(placeholder=placeholder):
                self.assertIn(placeholder, template)

        submitted_runtime = template + (
            REPOSITORY_ROOT / "cluster/render_jobs.py"
        ).read_text()
        for forbidden in (
            "SCALEQ_BOOTSTRAP_COMMAND",
            "pytorch/pytorch:",
            "namespace: default",
            "queue: default",
            "priorityClassName: high",
            "activeDeadlineSeconds: 21600",
            "sizeLimit: 32Gi",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, submitted_runtime)


if __name__ == "__main__":
    unittest.main()
