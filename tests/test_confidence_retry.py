"""Offline regression tests for confidence completion request construction."""

import copy
import json
import os
import threading
import unittest
from unittest.mock import patch

from src.adapters import simple


CONFIDENCE_SYSTEM = (
    "Return only one tag: <confidence>N</confidence>, "
    "where N is an integer from 0 to 100."
)
FORMAT_SUFFIX = "For the final benchmark response, output exactly these tags:"


class FakeResponse:
    status_code = 200

    def __init__(self, content, usage):
        self.text = json.dumps(
            {"choices": [{"message": {"content": content}}], "usage": usage}
        )

    def raise_for_status(self):
        pass

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []

    def post(self, url, *, json, timeout):
        self.payloads.append(copy.deepcopy(json))
        try:
            return next(self.responses)
        except StopIteration as exc:
            raise AssertionError("Unexpected additional LLM request") from exc


def message_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content if part.get("type") == "text"
    )


class ConfidenceRetryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"REQUIRE_CONFIDENCE_TAG": "1"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.messages = [
            {"role": "system", "content": "Classify the financial evidence."},
            {
                "role": "user",
                "content": (
                    "Question: Did revenue increase?\n"
                    "Evidence: Revenue rose from 10 to 12.\n"
                    "Allowed labels: YES, NO.\n"
                    "Return the label inside <answer>."
                ),
            },
        ]
        self.budget = {"max_output_tokens": 100, "temperature": 0.0}
        self.answer = "<think>Revenue rose.</think><answer>YES</answer>"

    def client(self, outputs):
        responses = [
            FakeResponse(
                content,
                usage or {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            )
            for content, usage in outputs
        ]
        session = FakeSession(responses)
        client = simple.OpenAICompatibleLLM.__new__(simple.OpenAICompatibleLLM)
        client.base_url = "https://example.invalid/v1"
        client.model = "offline-test-model"
        client._session_local = threading.local()
        client._thread_session = lambda: session
        return client, session

    def test_missing_confidence_retains_task_evidence_and_generated_answer(self):
        client, session = self.client(
            [(self.answer, None), ("<confidence>82</confidence>", None)]
        )
        original_messages = copy.deepcopy(self.messages)
        original_budget = dict(self.budget)
        result = client.chat(self.messages, self.budget)

        self.assertEqual(len(session.payloads), 2)
        primary, retry = session.payloads
        self.assertEqual(primary["messages"][0], self.messages[0])
        self.assertEqual(primary["messages"][1]["content"].count(FORMAT_SUFFIX), 1)
        self.assertEqual(len(retry["messages"]), 2)
        self.assertEqual(retry["messages"][0]["role"], "system")
        self.assertTrue(retry["messages"][0]["content"].startswith(CONFIDENCE_SYSTEM))
        self.assertIsInstance(retry["messages"][1]["content"], str)
        retry_text = retry["messages"][1]["content"]
        for original in primary["messages"]:
            self.assertIn(original["content"], retry_text)
        self.assertIn(self.answer, retry_text)
        # The original task may contain its existing suffix, but no new suffix
        # may be appended after the supplied answer or confidence instruction.
        self.assertNotIn(FORMAT_SUFFIX, retry_text.split(self.answer, 1)[1])
        self.assertEqual(retry["max_tokens"], 32)
        self.assertEqual(result["content"], self.answer + "\n<confidence>82</confidence>")
        self.assertEqual(self.messages, original_messages)
        self.assertEqual(self.budget, original_budget)

    def test_multimodal_retry_preserves_images_and_does_not_mutate_input(self):
        images = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQ==", "detail": "high"}},
        ]
        self.messages[1]["content"] = [
            {"type": "text", "text": self.messages[1]["content"]},
            images[0],
            {"type": "text", "text": "Second chart:"},
            images[1],
        ]
        original = copy.deepcopy(self.messages)
        client, session = self.client(
            [(self.answer, None), ("<confidence>76</confidence>", None)]
        )
        client.chat(self.messages, self.budget)
        content = session.payloads[1]["messages"][1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual([part for part in content if part.get("type") == "image_url"], images)
        self.assertIn("Second chart:", message_text(content))
        self.assertIn(self.answer, message_text(content))
        self.assertEqual(self.messages, original)

        # Mutating a retry payload must not mutate the original nested image.
        retry = simple._build_confidence_retry_messages(self.messages, self.answer)
        retry_images = [
            part for part in retry[1]["content"] if part.get("type") == "image_url"
        ]
        retry_images[0]["image_url"]["detail"] = "high"
        self.assertEqual(self.messages, original)

    def test_valid_primary_confidence_does_not_retry(self):
        answer = self.answer + "<confidence>82</confidence>"
        client, session = self.client([(answer, None)])
        result = client.chat(self.messages, self.budget)
        self.assertEqual(len(session.payloads), 1)
        self.assertEqual(result["content"], answer)

    def test_disabled_confidence_reporting_preserves_request(self):
        os.environ["REQUIRE_CONFIDENCE_TAG"] = "0"
        client, session = self.client([(self.answer, None)])
        result = client.chat(self.messages, self.budget)
        self.assertEqual(len(session.payloads), 1)
        self.assertEqual(session.payloads[0]["messages"], self.messages)
        self.assertEqual(result["content"], self.answer)

    def test_confidence_only_request_gets_no_suffix_or_recursive_retry(self):
        messages = [
            {"role": "system", "content": CONFIDENCE_SYSTEM},
            {"role": "user", "content": "Assess this answer: <answer>YES</answer>"},
        ]
        client, session = self.client([("Unable to estimate.", None)])
        result = client.chat(messages, self.budget)
        self.assertEqual(len(session.payloads), 1)
        self.assertEqual(session.payloads[0]["messages"], messages)
        self.assertEqual(result["content"], "Unable to estimate.")

    def test_quoted_confidence_instruction_is_not_a_system_control(self):
        self.messages[1]["content"] += "\nQuoted text: " + CONFIDENCE_SYSTEM
        client, session = self.client(
            [(self.answer, None), ("<confidence>82</confidence>", None)]
        )
        client.chat(self.messages, self.budget)
        self.assertEqual(len(session.payloads), 2)
        self.assertIn(FORMAT_SUFFIX, session.payloads[0]["messages"][1]["content"])

    def test_unsuccessful_completion_stops_after_one_retry(self):
        client, session = self.client([(self.answer, None), ("<confidence></confidence>", None)])
        result = client.chat(self.messages, self.budget)
        self.assertEqual(len(session.payloads), 2)
        self.assertEqual(result["content"], self.answer + "\n<confidence></confidence>")

    def test_retry_usage_and_latency_are_aggregated(self):
        primary_usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        retry_usage = {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
        client, _ = self.client(
            [(self.answer, primary_usage), ("<confidence>82</confidence>", retry_usage)]
        )
        with patch.object(simple.time, "perf_counter", side_effect=[1.0, 2.0, 3.0, 4.0]):
            result = client.chat(self.messages, self.budget)
        self.assertEqual(result["prompt_tokens"], 30)
        self.assertEqual(result["completion_tokens"], 7)
        self.assertEqual(result["total_tokens"], 37)
        self.assertEqual(result["latency_ms"], 2000.0)
        self.assertEqual(result["raw_usage"]["primary_usage"], primary_usage)
        self.assertEqual(result["raw_usage"]["confidence_retry_usage"], retry_usage)


if __name__ == "__main__":
    unittest.main()
