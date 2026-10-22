import unittest

from insightmem.chunking import chunk_document
from insightmem.evaluation import AggregateMetrics, token_f1
from insightmem.schema import Document, Query


class CoreTests(unittest.TestCase):
    def test_chunking_uses_overlap(self):
        document = Document(id="d1", text="one two three four five six")
        chunks = chunk_document(document, chunk_size=4, overlap=2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "one two three four")
        self.assertEqual(chunks[1].text, "three four five six")

    def test_metrics(self):
        metrics = AggregateMetrics()
        query = Query(id="q", question="q", answer="Ada Lovelace", relevant_doc_ids=("ada",))
        row = metrics.add(query, ["ada", "babbage"], "Ada Lovelace wrote notes.", "Ada Lovelace")
        self.assertEqual(row["retrieval_recall"], 1.0)
        self.assertEqual(metrics.summary()["mrr"], 1.0)
        self.assertEqual(token_f1("Ada Lovelace", "Ada Lovelace"), 1.0)

    def test_recall_counts_all_relevant_documents(self):
        metrics = AggregateMetrics()
        query = Query(id="q", question="q", relevant_doc_ids=("d1", "d2"))
        row = metrics.add(query, ["d1"], "", None)
        self.assertEqual(row["retrieval_recall"], 0.5)
        self.assertIsNone(metrics.summary()["exact_match"])


if __name__ == "__main__":
    unittest.main()
