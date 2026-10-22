import unittest
from datetime import datetime, timezone

from scripts.local_benchmark import chunk_messages
from scripts.prepare_public_data import convert_locomo, convert_longmemeval, timestamp_ms


class PublicDataTests(unittest.TestCase):
    def test_locomo_conversion_resolves_evidence(self):
        source = [
            {
                "sample_id": "sample-1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "I moved to Shanghai."},
                        {"speaker": "Bob", "dia_id": "D1:2", "text": "Thanks for telling me."},
                    ],
                },
                "qa": [{"question": "Where did Alice move?", "answer": "Shanghai", "evidence": ["D1:1"]}],
            }
        ]
        result = convert_locomo(source)[0]
        self.assertEqual(result["messages"][0]["role"], "user")
        expected_timestamp = int(datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual([message["timestamp"] for message in result["messages"]], [expected_timestamp] * 2)
        self.assertEqual(result["questions"][0]["evidence_texts"], ["I moved to Shanghai."])

    def test_timestamp_ms_parses_locomo_session_date(self):
        expected = int(datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(timestamp_ms("1:56 pm on 8 May, 2023"), expected)

    def test_longmemeval_conversion(self):
        source = [
            {
                "question_id": "q1",
                "question": "What color?",
                "answer": "blue",
                "haystack_session_ids": ["s1"],
                "answer_session_ids": ["s1"],
                "haystack_sessions": [[{"role": "user", "content": "My favorite color is blue."}]],
            }
        ]
        result = convert_longmemeval(source)[0]
        self.assertEqual(result["id"], "q1")
        self.assertEqual(result["questions"][0]["evidence_texts"], ["My favorite color is blue."])

    def test_chunk_messages_splits_large_single_message(self):
        messages = [{"role": "user", "content": " ".join(f"w{i}" for i in range(12))}]
        chunks = chunk_messages(messages, max_messages=20, max_words=5)
        self.assertEqual([len(item[0]["content"].split()) for item in chunks], [5, 5, 2])


if __name__ == "__main__":
    unittest.main()
