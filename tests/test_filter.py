import unittest

from insightmem.coding.filter import low_information


class AssistantFilterTests(unittest.TestCase):
    def test_pure_process_messages_are_filtered(self):
        examples = [
            "",
            "Let me inspect alpha.py and check the helper function.",
            "Good. Let me inspect the implementation. Let me run the relevant tests.",
            "Now I will read the source files and run the tests.",
            "我来查看一下相关代码。",
            "The fix is complete and verified. Let me provide a summary of the changes.",
            "Let me run the existing tests to verify the fix.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(low_information(text))

    def test_messages_with_findings_are_retained(self):
        examples = [
            "Let me inspect the code. It raises ValueError for negative inputs.",
            "Now update Parser.rank() to preserve attrs.",
            "Now fix errorbar validation — add NaN check.",
            "All tests pass. Let me verify the final state of the change.",
            "See foo.py for the implementation.",
            "Let me read the retry limit 3.",
            "解析器在遇到转义字段时会丢失反斜杠。",
            "[tool_use Read] {\"path\": \"src/parser.py\"}",
            "[tool_result] 12 passed in 0.4s",
            "The bug is in unescape(): it drops backslashes because the replacement runs twice.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertFalse(low_information(text))

    def test_legacy_transition_rule_is_retained(self):
        self.assertTrue(low_information("Let me inspect the code."))
        self.assertFalse(low_information("Let me inspect the code, which uses zero-based indexing."))


if __name__ == "__main__":
    unittest.main()
