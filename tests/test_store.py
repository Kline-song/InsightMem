import unittest

import numpy as np

from insightmem.store import bm25_scores
from insightmem.text import lexical_tokens


class LexicalTests(unittest.TestCase):
    def test_tokens_cover_english_cjk_and_punctuation(self):
        tokens = lexical_tokens("Parser 解析器 使用 unescape()!")
        self.assertIn("parser", tokens)
        self.assertIn("解", tokens)
        self.assertIn("unescape", tokens)

    def test_bm25_ranks_matching_documents(self):
        documents = ["alpha beta", "gamma delta", "alpha alpha"]
        scores = bm25_scores(documents, "alpha")
        self.assertGreater(scores[0], 0)
        self.assertGreater(scores[2], scores[0])
        self.assertEqual(scores[1], 0)

    def test_bm25_handles_empty_input(self):
        self.assertEqual(len(bm25_scores([], "alpha")), 0)
        np.testing.assert_array_equal(bm25_scores(["alpha"], ""), [0.0])


if __name__ == "__main__":
    unittest.main()
