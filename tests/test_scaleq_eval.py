import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
import scaleq_eval  # noqa: E402


class ScaleQEvalTest(unittest.TestCase):
    def test_generate_batches_prompts_and_trims_each_result(self):
        class Encoded(dict):
            def to(self, device):
                return self

        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = (
            lambda messages, **_: messages[0]["content"]
        )
        tokenizer.return_value = Encoded(
            input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        )
        tokenizer.decode.side_effect = lambda token_ids, **_: str(token_ids)
        model = SimpleNamespace(
            device="cpu",
            generation_config=SimpleNamespace(eos_token_id=9),
            generate=Mock(
                return_value=torch.tensor(
                    [[0, 1, 2, 6, 9, 9, 9], [3, 4, 5, 7, 8, 9, 9]]
                )
            ),
        )
        args = SimpleNamespace(
            thinking=True,
            max_new_tokens=4,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )

        results = scaleq_eval.generate(
            model, tokenizer, ["first", "second"], args, seed=2
        )

        self.assertEqual(results[0]["prompt_ids"], [1, 2])
        self.assertEqual(results[0]["generated_ids"], [6, 9])
        self.assertEqual(results[1]["prompt_ids"], [3, 4, 5])
        self.assertEqual(results[1]["generated_ids"], [7, 8, 9])

    def test_extracts_last_balanced_box(self):
        text = r"First \boxed{1}. Final answer: \boxed{\frac{3}{4}}"
        self.assertEqual(scaleq_eval.extract_answer(text), r"\frac{3}{4}")

    def test_scores_gsm8k_numeric_answer(self):
        prediction, target, correct = scaleq_eval.score_answer(
            "The final answer is 1,024.",
            "work\n#### 1024",
        )
        self.assertEqual(prediction, "1024")
        self.assertEqual(target, "1024")
        self.assertTrue(correct)

    def test_uses_last_nonempty_line_as_fallback(self):
        self.assertEqual(scaleq_eval.extract_answer("reasoning\n\n42\n"), "42")

    def test_strips_python_fence(self):
        self.assertEqual(
            scaleq_eval.strip_code_fence("text\n```python\nreturn 1\n```"),
            "return 1",
        )

    def test_deterministic_rank_assignment(self):
        self.assertEqual(
            [index for index in range(8) if scaleq_eval.assigned(index, 1, 3)],
            [1, 4, 7],
        )

    def test_rank_zero_prefetches_code_datasets_before_distributed_evaluation(self):
        code_examples = Mock(return_value=[])
        with (
            patch.object(scaleq_eval, "code_examples", code_examples),
            patch.object(scaleq_eval.dist, "barrier") as barrier,
        ):
            scaleq_eval.prefetch_code_datasets(
                ["math500", "humaneval", "mbpp"],
                rank=0,
                world=8,
            )

        self.assertEqual(
            [call.args[0] for call in code_examples.call_args_list],
            ["humaneval", "mbpp"],
        )
        barrier.assert_called_once_with()

    def test_nonzero_rank_waits_for_code_dataset_prefetch(self):
        with (
            patch.object(scaleq_eval, "code_examples") as code_examples,
            patch.object(scaleq_eval.dist, "barrier") as barrier,
        ):
            scaleq_eval.prefetch_code_datasets(
                ["humaneval", "mbpp"],
                rank=1,
                world=8,
            )

        code_examples.assert_not_called()
        barrier.assert_called_once_with()

    def test_torn_trailing_record_is_ignored_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank-00-of-01.jsonl"
            path.write_text('{"task_id": "math500-000000"}\n{"task_id": "math')
            self.assertEqual(
                scaleq_eval.read_jsonl(path),
                [{"task_id": "math500-000000"}],
            )
            self.assertEqual(
                path.read_text(),
                '{"task_id": "math500-000000"}\n{"task_id": "math',
            )

    def test_trims_generated_padding_after_first_eos(self):
        self.assertEqual(scaleq_eval.trim_at_eos([1, 2, 9, 9, 9], 9), [1, 2, 9])
        self.assertEqual(scaleq_eval.trim_at_eos([1, 2, 3], [8, 9]), [1, 2, 3])

    def test_finalize_rejects_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            task_dir = output_dir / "math500"
            task_dir.mkdir()
            (task_dir / "rank-00-of-01.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "math500-000000",
                        "score_status": "scored",
                        "exact_or_numeric_match": True,
                    }
                )
                + "\n"
            )
            with self.assertRaises(AssertionError):
                scaleq_eval.finalize_task("math500", output_dir, expected_rows=2)


if __name__ == "__main__":
    unittest.main()
