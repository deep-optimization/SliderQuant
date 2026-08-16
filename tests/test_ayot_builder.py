"""Unit tests for scripts/build_ayot_calibration.py.

Covers the pieces that must be right before any expensive generation runs:
question normalization, deterministic balanced selection with cross-source
deduplication, fixed-width tokenization, and checksum/manifest sealing.

    python -m unittest discover -s tests -v
"""

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_ayot_calibration as ayot  # noqa: E402


class FakeTokenizer:
    """Whitespace tokenizer with a stable vocabulary, enough for encode_fixed_length."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "chat-rendered text must not get extra specials"
        return {"input_ids": [self.vocab.setdefault(t, len(self.vocab) + 10) for t in text.split()]}


def fake_source(question_field="query", id_field=""):
    return {
        "repo": "fake/dataset", "revision": "deadbeef", "config": "default", "split": "train",
        "question_field": question_field, "id_field": id_field,
    }


class TestNormalizeQuestion(unittest.TestCase):
    def test_collapses_whitespace_and_case(self):
        self.assertEqual(
            ayot.normalize_question("  What\tis\n\n 2 + 2 ?  "), "what is 2 + 2 ?"
        )

    def test_unicode_and_zero_width_are_canonicalized(self):
        self.assertEqual(
            ayot.normalize_question("\ufb01le\u200b ①"), ayot.normalize_question("File 1")
        )

    def test_blank_questions_normalize_to_empty(self):
        for blank in ("", "   ", "\n\t", "\u200b"):
            self.assertEqual(ayot.normalize_question(blank), "")

    def test_distinct_questions_stay_distinct(self):
        self.assertNotEqual(ayot.normalize_question("2 + 2"), ayot.normalize_question("2 + 3"))


class TestSelectDomain(unittest.TestCase):
    def setUp(self):
        self.rows = [{"query": f"Question {i}", "id": f"m{i}"} for i in range(50)]
        # 20 unique questions, 20 blanks, 20 normalized duplicates of the unique ones
        self.messy = []
        for i in range(60):
            if i % 3 == 0:
                self.messy.append({"query": f"Question {i}", "id": f"m{i}"})
            elif i % 3 == 1:
                self.messy.append({"query": "  \n\t ", "id": f"m{i}"})
            else:
                self.messy.append({"query": f"  question {i - 2}  ", "id": f"m{i}"})

    def test_selects_exactly_the_target_and_is_deterministic(self):
        first, first_rejected = ayot.select_domain(
            self.rows, "math", fake_source(id_field="id"), seed=2, target=10, seen_hashes=set()
        )
        second, second_rejected = ayot.select_domain(
            self.rows, "math", fake_source(id_field="id"), seed=2, target=10, seen_hashes=set()
        )
        self.assertEqual(len(first), 10)
        self.assertEqual(first, second)
        self.assertEqual(first_rejected, second_rejected)
        self.assertEqual([r["sample_id"] for r in first],
                         [f"math-{i:06d}" for i in range(10)])

    def test_a_different_seed_selects_different_rows(self):
        a, _ = ayot.select_domain(self.rows, "math", fake_source(), 2, 10, set())
        b, _ = ayot.select_domain(self.rows, "math", fake_source(), 3, 10, set())
        self.assertNotEqual([r["row_index"] for r in a], [r["row_index"] for r in b])

    def test_records_provenance_and_hashes(self):
        accepted, _ = ayot.select_domain(
            self.rows, "math", fake_source(id_field="id"), 2, 5, set()
        )
        record = accepted[0]
        raw = self.rows[record["row_index"]]["query"]
        self.assertEqual(record["question"], raw)
        self.assertEqual(record["question_sha256"],
                         hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(record["question_norm_sha256"],
                         hashlib.sha256(ayot.normalize_question(raw).encode()).hexdigest())
        self.assertEqual(record["upstream_id"], f"m{record['row_index']}")
        self.assertEqual(record["question_field"], "query")
        self.assertEqual(record["dataset_repo"], "fake/dataset")
        self.assertEqual(record["dataset_revision"], "deadbeef")
        self.assertEqual(record["selection_seed"], 2)

    def test_upstream_id_is_none_when_the_source_has_no_id_field(self):
        accepted, _ = ayot.select_domain(self.rows, "math", fake_source(), 2, 5, set())
        self.assertTrue(all(r["upstream_id"] is None for r in accepted))

    def test_rejects_blank_and_duplicate_questions_with_reasons(self):
        accepted, rejected = ayot.select_domain(
            self.messy, "math", fake_source(id_field="id"), 2, 20, set()
        )
        self.assertEqual(len(accepted), 20)
        self.assertEqual(len({r["question_norm_sha256"] for r in accepted}), 20)
        self.assertTrue(all(r["row_index"] % 3 != 1 for r in accepted))

        reasons = {r["row_index"]: r["rejection_reason"] for r in rejected}
        self.assertTrue(reasons, "the scan must have encountered rejectable rows")
        for row_index, reason in reasons.items():
            expected = "empty_question" if row_index % 3 == 1 else "duplicate_question"
            self.assertEqual(reason, expected, row_index)
        self.assertIn("empty_question", reasons.values())
        self.assertIn("duplicate_question", reasons.values())
        self.assertFalse({r["row_index"] for r in accepted} & set(reasons))

    def test_deduplicates_across_sources_via_shared_hash_set(self):
        seen = set()
        math, _ = ayot.select_domain(self.rows, "math", fake_source(), 2, 10, seen)
        code_rows = [{"input": r["question"]} for r in math] + \
                    [{"input": f"Code {i}"} for i in range(10)]
        code, rejected = ayot.select_domain(
            code_rows, "code", fake_source(question_field="input"), 2, 10, seen
        )
        self.assertEqual(len(code), 10)
        self.assertTrue(all(r["question"].startswith("Code ") for r in code))
        self.assertTrue(any(r["rejection_reason"] == "duplicate_question" for r in rejected))
        self.assertEqual([r["sample_id"] for r in code], [f"code-{i:06d}" for i in range(10)])

    def test_raises_when_the_source_cannot_fill_the_target(self):
        with self.assertRaises(AssertionError):
            ayot.select_domain(self.rows, "math", fake_source(), 2, len(self.rows) + 1, set())

    def test_rejects_evaluation_overlap_before_acceptance(self):
        order = list(range(len(self.rows)))
        random.Random(2).shuffle(order)
        excluded_question = self.rows[order[0]]["query"]
        excluded = {ayot.sha256_text(ayot.normalize_question(excluded_question))}
        accepted, rejected = ayot.select_domain(
            self.rows,
            "math",
            fake_source(),
            2,
            10,
            set(),
            excluded,
        )
        self.assertNotIn(excluded_question, {row["question"] for row in accepted})
        self.assertTrue(
            any(row["rejection_reason"] == "evaluation_overlap" for row in rejected)
        )


class TestEncodeFixedLength(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_right_pads_short_sequences(self):
        ids, mask, length, truncated = ayot.encode_fixed_length(
            self.tokenizer, "a b c", width=8, pad_id=0
        )
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(mask), 8)
        self.assertEqual(length, 3)
        self.assertFalse(truncated)
        self.assertEqual(ids[3:], [0, 0, 0, 0, 0])
        self.assertEqual(mask, [1, 1, 1, 0, 0, 0, 0, 0])

    def test_right_truncates_long_sequences(self):
        text = " ".join(f"t{i}" for i in range(20))
        full = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids, mask, length, truncated = ayot.encode_fixed_length(
            self.tokenizer, text, width=8, pad_id=0
        )
        self.assertEqual(ids, full[:8])
        self.assertEqual(mask, [1] * 8)
        self.assertEqual(length, 8)
        self.assertTrue(truncated)

    def test_exact_width_is_not_flagged_as_truncated(self):
        text = " ".join(f"t{i}" for i in range(8))
        ids, mask, length, truncated = ayot.encode_fixed_length(
            self.tokenizer, text, width=8, pad_id=0
        )
        self.assertEqual(length, 8)
        self.assertEqual(mask, [1] * 8)
        self.assertFalse(truncated)
        self.assertEqual(len(ids), 8)

    def test_empty_text_is_all_padding(self):
        ids, mask, length, truncated = ayot.encode_fixed_length(
            self.tokenizer, "", width=4, pad_id=7
        )
        self.assertEqual(ids, [7, 7, 7, 7])
        self.assertEqual(mask, [0, 0, 0, 0])
        self.assertEqual(length, 0)
        self.assertFalse(truncated)

    def test_production_width_is_2048(self):
        self.assertEqual(ayot.SEQ_LEN, 2048)
        self.assertEqual(2 * ayot.PER_DOMAIN * ayot.SEQ_LEN, 4_194_304)


class TestChecksumsAndManifest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent))
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name)
        (self.out_dir / "selection.jsonl").write_text('{"sample_id": "math-000000"}\n')
        (self.out_dir / "generation-config.json").write_text("{}\n")
        (self.out_dir / "tokenized-00000-of-00008.safetensors").write_bytes(b"\x00\x01\x02")

    def test_sha256sums_covers_every_artifact_file_in_sorted_order(self):
        checksums = ayot.write_sha256sums(self.out_dir)
        lines = (self.out_dir / "SHA256SUMS").read_text().splitlines()
        names = [line.split("  ", 1)[1] for line in lines]
        self.assertEqual(names, sorted(names))
        self.assertEqual(set(names), {"generation-config.json", "selection.jsonl",
                                      "tokenized-00000-of-00008.safetensors"})
        self.assertEqual(set(checksums), set(names))
        for line in lines:
            digest, name = line.split("  ", 1)
            self.assertEqual(digest, hashlib.sha256((self.out_dir / name).read_bytes()).hexdigest())
            self.assertEqual(digest, checksums[name])

    def test_sha256sums_excludes_itself_and_the_manifest(self):
        (self.out_dir / "manifest.json").write_text("{}\n")
        ayot.write_sha256sums(self.out_dir)
        names = {line.split("  ", 1)[1] for line in (self.out_dir / "SHA256SUMS").read_text().splitlines()}
        self.assertNotIn("SHA256SUMS", names)
        self.assertNotIn("manifest.json", names)

    def test_sha256sums_changes_when_a_file_changes(self):
        ayot.write_sha256sums(self.out_dir)
        before = (self.out_dir / "SHA256SUMS").read_text()
        (self.out_dir / "selection.jsonl").write_text('{"sample_id": "math-000001"}\n')
        ayot.write_sha256sums(self.out_dir)
        self.assertNotEqual(before, (self.out_dir / "SHA256SUMS").read_text())

    def test_manifest_is_published_atomically_and_chains_to_sha256sums(self):
        ayot.write_sha256sums(self.out_dir)
        digest = ayot.sha256_file(self.out_dir / "SHA256SUMS")
        ayot.publish_manifest(self.out_dir, {"artifact_id": "ayot-qwen3-1p7b-v1",
                                             "sha256sums_sha256": digest})
        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["sha256sums_sha256"], digest)
        self.assertEqual(list(self.out_dir.glob("*.tmp")), [])

    def test_sha256_file_matches_hashlib(self):
        path = self.out_dir / "tokenized-00000-of-00008.safetensors"
        self.assertEqual(ayot.sha256_file(path),
                         hashlib.sha256(path.read_bytes()).hexdigest())


class TestSubsetManifests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent))
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name)
        self.placements = []
        for i in range(256):
            domain = "math" if i < 128 else "code"
            self.placements.append({
                "sample_id": f"{domain}-{i % 128:06d}", "domain": domain,
                "global_row": i, "shard": f"tokenized-{i // 128:05d}-of-00002.safetensors",
                "shard_row": i % 128,
            })

    def test_subsets_are_balanced_deterministic_and_derived(self):
        names = ayot.write_subsets(self.out_dir, "ayot-qwen3-1p7b-v1", self.placements)
        self.assertEqual(names, {"32": "subset-32.json", "128": "subset-128.json"})
        for size in ayot.SUBSET_SIZES:
            subset = json.loads((self.out_dir / f"subset-{size}.json").read_text())
            domains = [s["domain"] for s in subset["samples"]]
            self.assertEqual(len(subset["samples"]), size)
            self.assertEqual(domains.count("math"), size // 2)
            self.assertEqual(domains.count("code"), size // 2)
            self.assertEqual(subset["seq_len"], ayot.SEQ_LEN)
            self.assertEqual([s["global_row"] for s in subset["samples"]],
                             sorted(s["global_row"] for s in subset["samples"]))
            self.assertTrue({s["sample_id"] for s in subset["samples"]}.issubset(
                {p["sample_id"] for p in self.placements}))

    def test_the_32_row_subset_is_contained_in_the_128_row_subset(self):
        ayot.write_subsets(self.out_dir, "ayot-qwen3-1p7b-v1", self.placements)
        small = {s["sample_id"] for s in
                 json.loads((self.out_dir / "subset-32.json").read_text())["samples"]}
        large = {s["sample_id"] for s in
                 json.loads((self.out_dir / "subset-128.json").read_text())["samples"]}
        self.assertTrue(small.issubset(large))

    def test_raises_when_a_domain_cannot_fill_a_subset(self):
        with self.assertRaises(AssertionError):
            ayot.write_subsets(self.out_dir, "v", self.placements[:20])


class TestShardRecovery(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent))
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "generations-00000-of-00001.jsonl"

    def test_torn_trailing_record_is_dropped_and_truncated(self):
        self.path.write_text('{"sample_id": "math-000000"}\n{"sample_id": "math-00')
        self.assertEqual(ayot.load_shard(self.path), [{"sample_id": "math-000000"}])
        self.assertEqual(self.path.read_text(), '{"sample_id": "math-000000"}\n')
        self.assertEqual(ayot.load_shard(self.path), [{"sample_id": "math-000000"}])

    def test_missing_shard_reads_as_empty(self):
        self.assertEqual(ayot.load_shard(self.path), [])

    def test_complete_but_corrupt_record_raises(self):
        self.path.write_text('{"sample_id": "math-000000"}\nnot json\n')
        with self.assertRaises(json.JSONDecodeError):
            ayot.load_shard(self.path)

    def test_failed_generation_is_not_considered_complete(self):
        records = [
            {"sample_id": "math-000000", "failed": True},
            {"sample_id": "math-000001", "failed": False},
        ]
        self.assertEqual(
            ayot.completed_sample_ids(records),
            {"math-000001"},
        )


class TestCombineGenerations(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent))
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name)
        self.rows = [{"sample_id": f"math-{i:06d}"} for i in range(4)]

    def _write(self, name, sample_ids):
        (self.out_dir / name).write_text(
            "".join(json.dumps({"sample_id": s, "shard": name}) + "\n" for s in sample_ids)
        )

    def test_merges_shards_in_selection_order(self):
        self._write("generations-00000-of-00002.jsonl", ["math-000000", "math-000002"])
        self._write("generations-00001-of-00002.jsonl", ["math-000003", "math-000001"])
        merged = ayot.combine_generations(self.out_dir, self.rows)
        self.assertEqual([r["sample_id"] for r in merged],
                         [r["sample_id"] for r in self.rows])

    def test_missing_generation_raises(self):
        self._write("generations-00000-of-00001.jsonl", ["math-000000", "math-000001"])
        with self.assertRaises(AssertionError):
            ayot.combine_generations(self.out_dir, self.rows)


if __name__ == "__main__":
    unittest.main()
