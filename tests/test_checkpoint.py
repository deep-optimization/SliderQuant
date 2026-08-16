import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from quantize.checkpoint import atomic_torch_save, window_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_atomic_save_replaces_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_state.pt"
            atomic_torch_save({"step": 1}, path)
            atomic_torch_save({"step": 2}, path)
            self.assertEqual(
                torch.load(path, weights_only=True),
                {"step": 2},
            )
            self.assertEqual(list(Path(directory).glob("checkpoint-*")), [])

    def test_window_checkpoint_records_runtime_code(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}")
            args = SimpleNamespace(
                calib_manifest=str(manifest),
                model="model",
                model_revision="revision",
            )
            with mock.patch.dict(
                "os.environ",
                {
                    "SCALEQ_SHA": "a" * 40,
                    "SCALEQ_CODE_SHA": "b" * 40,
                },
            ):
                checkpoint = window_checkpoint({}, 0, args)
            self.assertEqual(checkpoint["source_sha"], "a" * 40)
            self.assertEqual(checkpoint["code_sha"], "b" * 40)


if __name__ == "__main__":
    unittest.main()
