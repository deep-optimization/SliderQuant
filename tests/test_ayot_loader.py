import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from datautils import get_ayot


class AYOTLoaderTest(unittest.TestCase):
    def seal(self, root, rows):
        names = sorted(
            path.name
            for path in root.iterdir()
            if path.is_file() and path.name != "manifest.json"
        )
        checksums = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names
        }
        sums = "".join(f"{checksums[name]}  {name}\n" for name in names)
        (root / "SHA256SUMS").write_text(sums)
        manifest = {
            "artifact_id": "ayot-test-v1",
            "sha256sums_file": "SHA256SUMS",
            "sha256sums_sha256": hashlib.sha256(sums.encode()).hexdigest(),
            "files": checksums,
            "tokenized": {
                "rows": rows,
                "seq_len": 8,
                "shards": [{"file": "tokens.safetensors", "rows": rows}],
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_loads_sealed_fixed_width_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {
                    "input_ids": torch.arange(16).reshape(2, 8),
                    "attention_mask": torch.tensor(
                        [[1, 1, 1, 1, 1, 0, 0, 0], [1] * 8],
                        dtype=torch.int64,
                    ),
                },
                root / "tokens.safetensors",
            )
            self.seal(root, rows=2)

            rows, validation = get_ayot(root / "manifest.json", 2, 8)

            self.assertIsNone(validation)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0].shape, (1, 8))
            self.assertEqual(rows[0][2].dtype, torch.int64)
            self.assertEqual(rows[0][2].sum().item(), 5)

    def test_loads_rows_named_by_subset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {
                    "input_ids": torch.arange(24).reshape(3, 8),
                    "attention_mask": torch.ones(3, 8, dtype=torch.int64),
                },
                root / "tokens.safetensors",
            )
            (root / "subset.json").write_text(
                json.dumps(
                    {
                        "artifact_id": "ayot-test-v1",
                        "rows": 2,
                        "seq_len": 8,
                        "samples": [
                            {"global_row": 2},
                            {"global_row": 0},
                        ]
                    }
                )
            )
            self.seal(root, rows=3)

            rows, _ = get_ayot(
                root / "manifest.json",
                2,
                8,
                subset_path=root / "subset.json",
            )

            torch.testing.assert_close(rows[0][0], torch.arange(16, 24).reshape(1, 8))
            torch.testing.assert_close(rows[1][0], torch.arange(8).reshape(1, 8))

    def test_rejects_a_corrupted_token_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {
                    "input_ids": torch.arange(8).reshape(1, 8),
                    "attention_mask": torch.ones(1, 8, dtype=torch.int64),
                },
                root / "tokens.safetensors",
            )
            self.seal(root, rows=1)
            with (root / "tokens.safetensors").open("ab") as handle:
                handle.write(b"corrupt")

            with self.assertRaises(AssertionError):
                get_ayot(root / "manifest.json", 1, 8)


if __name__ == "__main__":
    unittest.main()
