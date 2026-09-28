"""
test_metrics.py — Unit tests for official evaluation metric and blocking stats
"""

import unittest
from src.metrics import compute_entity_f05, compute_macro_f05, compute_blocking_metrics


class TestMetrics(unittest.TestCase):

    def test_singleton_cases(self):
        # Empty true set, predicted empty -> 1.0
        self.assertEqual(compute_entity_f05(set(), set()), 1.0)

        # Empty true set, predicted something -> 0.0 (fatal false positive!)
        self.assertEqual(compute_entity_f05({"S2-1"}, set()), 0.0)

    def test_non_singleton_cases(self):
        # Non-empty true set, predicted empty -> 0.0
        self.assertEqual(compute_entity_f05(set(), {"S2-1"}), 0.0)

        # Exact match -> 1.0
        self.assertEqual(compute_entity_f05({"S2-1", "S3-2"}, {"S2-1", "S3-2"}), 1.0)

        # Official README example:
        # Ground truth: {"S2-00047", "S3-00812"}
        # Predicted: {"S2-00047", "S2-00193", "S3-00812"}
        # Expected F0.5 ~ 0.714
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        score = compute_entity_f05(pred, true)
        self.assertAlmostEqual(score, 0.714, places=3)

    def test_macro_f05(self):
        gt = {
            "S1-1": set(),  # singleton
            "S1-2": {"S2-1", "S3-1"},  # 2 matches
            "S1-3": {"S2-2"}  # 1 match
        }
        pred = {
            "S1-1": set(),  # correct singleton -> 1.0
            "S1-2": {"S2-1", "S3-1"},  # exact match -> 1.0
            "S1-3": set()  # missed -> 0.0
        }
        macro, details = compute_macro_f05(gt, pred, return_details=True)
        # mean(1.0, 1.0, 0.0) = 2/3 ~ 0.6667
        self.assertAlmostEqual(macro, 2.0 / 3.0, places=4)
        self.assertEqual(details["singleton_count"], 1)
        self.assertEqual(details["singleton_f05"], 1.0)
        self.assertEqual(details["non_singleton_count"], 2)

    def test_blocking_metrics(self):
        gt = {
            "S1-1": {"S2-1", "S2-2"},
            "S1-2": {"S3-1"},
            "S1-3": set()
        }
        cands = {
            "S1-1": {"S2-1", "S2-99"},  # recalled 1 of 2
            "S1-2": {"S3-1"},  # recalled 1 of 1
            "S1-3": set()
        }
        stats = compute_blocking_metrics(gt, cands, total_s23_count=100)
        # Total true pairs = 3, covered = 2 -> PC = 2/3
        self.assertAlmostEqual(stats["pair_completeness"], 2.0 / 3.0, places=4)
        self.assertGreater(stats["reduction_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
