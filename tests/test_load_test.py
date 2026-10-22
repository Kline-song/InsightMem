import unittest

from scripts.load_test import percentile


class LoadTestMetricsTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 100.0], 0.95), 100.0)
        self.assertEqual(percentile([1.0], 0.95), 1.0)
        self.assertIsNone(percentile([], 0.95))


if __name__ == "__main__":
    unittest.main()
