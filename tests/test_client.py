from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from whitebox.client import Whitebox
from whitebox.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    WhiteboxError,
)
from whitebox.models import Batch, Decision, Review


def mock_response(status_code=200, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.ok = 200 <= status_code < 300
    resp.text = ""
    return resp


DECISION_DICT = {
    "id": "dec_abc123",
    "status": "completed",
    "value": "approve",
    "confidence": 0.92,
    "verdict": "consensus",
    "escalated": False,
    "runs": [
        {"model": "gpt-4o", "answer": "approve", "logprob": -0.05, "latency_ms": 320},
        {"model": "claude-3", "answer": "approve", "logprob": -0.08, "latency_ms": 280},
    ],
    "latency_ms": 640,
    "cost_usd": 0.003,
    "created_at": "2026-04-27T10:00:00Z",
    "mode": "standard",
}

BATCH_DICT = {
    "id": "bat_xyz789",
    "status": "processing",
    "total": 10,
    "completed": 4,
    "failed": 0,
    "progress": 0.4,
    "webhook_url": None,
    "completed_at": None,
    "created_at": "2026-04-27T10:05:00Z",
}

REVIEW_DICT = {
    "id": 42,
    "decision_id": "dec_esc456",
    "status": "pending",
    "input": "some text",
    "options": ["approve", "reject"],
    "model_votes": {"approve": 3, "reject": 4},
    "confidence": 0.51,
    "sla_deadline": "2026-04-28T10:00:00Z",
    "created_at": "2026-04-27T11:00:00Z",
}


class TestConstructor(unittest.TestCase):
    def test_stores_api_key(self):
        client = Whitebox(api_key="sk-test-key")
        self.assertEqual(client.api_key, "sk-test-key")

    def test_default_base_url(self):
        client = Whitebox(api_key="sk-test-key")
        self.assertEqual(client.base_url, "https://whiteboxhq.ai/api/v1")

    def test_custom_base_url(self):
        client = Whitebox(api_key="sk-test-key", base_url="https://custom.api.com/v2/")
        self.assertEqual(client.base_url, "https://custom.api.com/v2")

    def test_custom_base_url_strips_trailing_slash(self):
        client = Whitebox(api_key="sk-test-key", base_url="https://custom.api.com/v2///")
        self.assertEqual(client.base_url, "https://custom.api.com/v2")

    def test_session_headers(self):
        client = Whitebox(api_key="sk-test-key")
        self.assertEqual(client.session.headers["Authorization"], "Bearer sk-test-key")
        self.assertEqual(client.session.headers["Content-Type"], "application/json")


@patch("requests.Session.request")
class TestDecide(unittest.TestCase):
    def test_sends_correct_post_body(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        client.decide(input="classify this", options=["a", "b"])

        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/decide", args[1])
        body = kwargs["json"]
        self.assertEqual(body["input"], "classify this")
        self.assertEqual(body["options"], ["a", "b"])
        self.assertEqual(body["runs"], 7)
        self.assertEqual(body["threshold"], 0.75)
        self.assertTrue(body["sync"])
        self.assertEqual(body["mode"], "standard")

    def test_returns_decision_object(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        result = client.decide(input="text", options=["a", "b"])
        self.assertIsInstance(result, Decision)
        self.assertEqual(result.id, "dec_abc123")
        self.assertEqual(result.value, "approve")

    def test_passes_models_param(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        client.decide(input="text", options=["a", "b"], models=["gpt-4o", "claude-3"])
        body = mock_req.call_args[1]["json"]
        self.assertEqual(body["models"], ["gpt-4o", "claude-3"])

    def test_omits_models_when_none(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        client.decide(input="text", options=["a", "b"])
        body = mock_req.call_args[1]["json"]
        self.assertNotIn("models", body)

    def test_includes_prompt_when_provided(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        client.decide(input="text", options=["a", "b"], prompt="custom prompt")
        body = mock_req.call_args[1]["json"]
        self.assertEqual(body["prompt"], "custom prompt")


@patch("requests.Session.request")
class TestDecideFast(unittest.TestCase):
    def test_sends_fast_mode_and_sync(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        client.decide_fast(input="text", options=["a", "b"])

        body = mock_req.call_args[1]["json"]
        self.assertEqual(body["mode"], "fast")
        self.assertTrue(body["sync"])


@patch("requests.Session.request")
class TestDecideBulk(unittest.TestCase):
    def test_sends_items_array(self, mock_req):
        mock_req.return_value = mock_response(json_data=BATCH_DICT)
        client = Whitebox(api_key="sk-test")
        items = [{"input": "text1", "options": ["a", "b"]}, {"input": "text2", "options": ["a", "b"]}]
        client.decide_bulk(items=items, options=["a", "b"])

        body = mock_req.call_args[1]["json"]
        self.assertEqual(body["items"], items)

    def test_returns_batch_object(self, mock_req):
        mock_req.return_value = mock_response(json_data=BATCH_DICT)
        client = Whitebox(api_key="sk-test")
        result = client.decide_bulk(items=[{"input": "text"}])
        self.assertIsInstance(result, Batch)
        self.assertEqual(result.id, "bat_xyz789")


@patch("requests.Session.request")
class TestGetDecision(unittest.TestCase):
    def test_sends_get_with_correct_path(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        client.get_decision("dec_abc123")

        args, kwargs = mock_req.call_args
        self.assertEqual(args[0], "GET")
        self.assertTrue(args[1].endswith("/decisions/dec_abc123"))

    def test_returns_decision(self, mock_req):
        mock_req.return_value = mock_response(json_data=DECISION_DICT)
        client = Whitebox(api_key="sk-test")
        result = client.get_decision("dec_abc123")
        self.assertIsInstance(result, Decision)
        self.assertEqual(result.id, "dec_abc123")


@patch("requests.Session.request")
class TestListDecisions(unittest.TestCase):
    def test_sends_page_params(self, mock_req):
        mock_req.return_value = mock_response(json_data={"decisions": [DECISION_DICT]})
        client = Whitebox(api_key="sk-test")
        client.list_decisions(page=2, per_page=50)

        kwargs = mock_req.call_args[1]
        self.assertEqual(kwargs["params"], {"page": 2, "per_page": 50})

    def test_returns_list_of_decisions(self, mock_req):
        mock_req.return_value = mock_response(json_data={"decisions": [DECISION_DICT, DECISION_DICT]})
        client = Whitebox(api_key="sk-test")
        result = client.list_decisions()
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Decision)

    def test_handles_list_response(self, mock_req):
        mock_req.return_value = mock_response(json_data=[DECISION_DICT])
        client = Whitebox(api_key="sk-test")
        result = client.list_decisions()
        self.assertEqual(len(result), 1)


@patch("requests.Session.request")
class TestGetBatch(unittest.TestCase):
    def test_returns_batch(self, mock_req):
        mock_req.return_value = mock_response(json_data=BATCH_DICT)
        client = Whitebox(api_key="sk-test")
        result = client.get_batch("bat_xyz789")
        self.assertIsInstance(result, Batch)
        self.assertEqual(result.id, "bat_xyz789")
        self.assertEqual(result.status, "processing")


@patch("requests.Session.request")
class TestGetBatchResults(unittest.TestCase):
    def test_returns_dict(self, mock_req):
        results_data = {"decisions": [DECISION_DICT], "batch_id": "bat_xyz789"}
        mock_req.return_value = mock_response(json_data=results_data)
        client = Whitebox(api_key="sk-test")
        result = client.get_batch_results("bat_xyz789")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["batch_id"], "bat_xyz789")


@patch("requests.Session.request")
class TestListReviews(unittest.TestCase):
    def test_returns_list_of_reviews(self, mock_req):
        mock_req.return_value = mock_response(json_data={"reviews": [REVIEW_DICT, REVIEW_DICT]})
        client = Whitebox(api_key="sk-test")
        result = client.list_reviews()
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Review)
        self.assertEqual(result[0].id, 42)


@patch("requests.Session.request")
class TestResolveReview(unittest.TestCase):
    def test_sends_patch_with_answer(self, mock_req):
        resolved = {**REVIEW_DICT, "status": "resolved"}
        mock_req.return_value = mock_response(json_data=resolved)
        client = Whitebox(api_key="sk-test")
        client.resolve_review(review_id=42, answer="approve")

        args, kwargs = mock_req.call_args
        self.assertEqual(args[0], "PATCH")
        self.assertTrue(args[1].endswith("/reviews/42"))
        self.assertEqual(kwargs["json"], {"answer": "approve"})

    def test_returns_review(self, mock_req):
        resolved = {**REVIEW_DICT, "status": "resolved"}
        mock_req.return_value = mock_response(json_data=resolved)
        client = Whitebox(api_key="sk-test")
        result = client.resolve_review(review_id=42, answer="approve")
        self.assertIsInstance(result, Review)


@patch("requests.Session.request")
class TestListModels(unittest.TestCase):
    def test_returns_dict_with_models_key(self, mock_req):
        models_data = {
            "models": ["gpt-4o", "claude-3", "gemini-pro"],
            "defaults": {"standard": "gpt-4o", "fast": "claude-3"},
        }
        mock_req.return_value = mock_response(json_data=models_data)
        client = Whitebox(api_key="sk-test")
        result = client.list_models()
        self.assertIsInstance(result, dict)
        self.assertIn("models", result)
        self.assertEqual(len(result["models"]), 3)


@patch("requests.Session.request")
class TestErrorHandling(unittest.TestCase):
    def test_401_raises_authentication_error(self, mock_req):
        mock_req.return_value = mock_response(status_code=401)
        client = Whitebox(api_key="bad-key")
        with self.assertRaises(AuthenticationError) as ctx:
            client.list_models()
        self.assertEqual(ctx.exception.status_code, 401)

    def test_402_raises_insufficient_credits_error(self, mock_req):
        mock_req.return_value = mock_response(status_code=402)
        client = Whitebox(api_key="sk-test")
        with self.assertRaises(InsufficientCreditsError) as ctx:
            client.decide(input="text", options=["a", "b"])
        self.assertEqual(ctx.exception.status_code, 402)

    def test_429_raises_rate_limit_error_with_retry_after(self, mock_req):
        mock_req.return_value = mock_response(
            status_code=429, headers={"Retry-After": "30"}
        )
        client = Whitebox(api_key="sk-test")
        with self.assertRaises(RateLimitError) as ctx:
            client.decide(input="text", options=["a", "b"])
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.retry_after, 30)

    def test_429_without_retry_after_header(self, mock_req):
        mock_req.return_value = mock_response(status_code=429, headers={})
        client = Whitebox(api_key="sk-test")
        with self.assertRaises(RateLimitError) as ctx:
            client.decide(input="text", options=["a", "b"])
        self.assertIsNone(ctx.exception.retry_after)

    def test_500_raises_whitebox_error(self, mock_req):
        resp = mock_response(status_code=500, json_data={"error": "Internal server error"})
        mock_req.return_value = resp
        client = Whitebox(api_key="sk-test")
        with self.assertRaises(WhiteboxError) as ctx:
            client.list_models()
        self.assertEqual(ctx.exception.status_code, 500)


class TestDecisionModel(unittest.TestCase):
    def test_from_dict(self):
        d = Decision.from_dict(DECISION_DICT)
        self.assertEqual(d.id, "dec_abc123")
        self.assertEqual(d.status, "completed")
        self.assertEqual(d.value, "approve")
        self.assertEqual(d.confidence, 0.92)
        self.assertEqual(len(d.runs), 2)
        self.assertEqual(d.runs[0].model, "gpt-4o")

    def test_shipped(self):
        """A decision is 'shipped' when status is completed and not escalated."""
        d = Decision.from_dict({**DECISION_DICT, "status": "completed", "escalated": False})
        self.assertEqual(d.status, "completed")
        self.assertFalse(d.escalated)

    def test_escalated(self):
        d = Decision.from_dict({**DECISION_DICT, "escalated": True})
        self.assertTrue(d.escalated)

    def test_from_dict_minimal(self):
        d = Decision.from_dict({"id": "dec_min", "status": "pending"})
        self.assertEqual(d.id, "dec_min")
        self.assertIsNone(d.value)
        self.assertEqual(d.runs, [])


class TestBatchModel(unittest.TestCase):
    def test_from_dict(self):
        b = Batch.from_dict(BATCH_DICT)
        self.assertEqual(b.id, "bat_xyz789")
        self.assertEqual(b.status, "processing")
        self.assertEqual(b.total, 10)
        self.assertEqual(b.completed, 4)
        self.assertEqual(b.progress, 0.4)

    def test_is_complete_false(self):
        b = Batch.from_dict(BATCH_DICT)
        self.assertFalse(b.is_complete)

    def test_is_complete_true(self):
        b = Batch.from_dict({**BATCH_DICT, "status": "complete"})
        self.assertTrue(b.is_complete)


class TestReviewModel(unittest.TestCase):
    def test_from_dict(self):
        r = Review.from_dict(REVIEW_DICT)
        self.assertEqual(r.id, 42)
        self.assertEqual(r.decision_id, "dec_esc456")
        self.assertEqual(r.status, "pending")
        self.assertEqual(r.input, "some text")
        self.assertEqual(r.options, ["approve", "reject"])
        self.assertEqual(r.confidence, 0.51)

    def test_from_dict_minimal(self):
        r = Review.from_dict({"id": 1, "decision_id": "d1", "status": "pending"})
        self.assertIsNone(r.input)
        self.assertIsNone(r.options)


if __name__ == "__main__":
    unittest.main()
