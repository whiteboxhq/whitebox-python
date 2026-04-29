from __future__ import annotations

from typing import Any, Optional

import requests

from .exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    WhiteboxError,
)
from .models import Batch, Decision, Review


class Whitebox:
    """Client for the WhiteBox AI decision observability API.

    Args:
        api_key: Your WhiteBox API key.
        base_url: Override the default API base URL.
        timeout: Request timeout in seconds (default 30).
    """

    BASE_URL = "https://whiteboxhq.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # ── Single decision ───────────────────────────────────────────────

    def decide(
        self,
        input: str,
        options: list[str],
        prompt: Optional[str] = None,
        runs: int = 7,
        threshold: float = 0.75,
        sync: bool = True,
        mode: str = "standard",
        models: Optional[list[str]] = None,
    ) -> Decision:
        """Submit a single classification query.

        Args:
            input: The text to classify.
            options: List of possible classification labels.
            prompt: Optional custom prompt for the models.
            runs: Number of model runs (default 7).
            threshold: Confidence threshold for auto-decision (default 0.75).
            sync: If True, block until the decision completes (default True).
            mode: "standard" or "fast" (default "standard").

        Returns:
            A Decision object with the result.
        """
        payload: dict[str, Any] = {
            "input": input,
            "options": options,
            "runs": runs,
            "threshold": threshold,
            "sync": sync,
            "mode": mode,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if models is not None:
            payload["models"] = models

        data = self._request("POST", "/decide", json=payload)
        return Decision.from_dict(data)

    def decide_fast(
        self,
        input: str,
        options: list[str],
        prompt: Optional[str] = None,
        threshold: float = 0.75,
    ) -> Decision:
        """Fast mode shortcut: 3 runs, 2 models, synchronous.

        Args:
            input: The text to classify.
            options: List of possible classification labels.
            prompt: Optional custom prompt for the models.
            threshold: Confidence threshold (default 0.75).

        Returns:
            A Decision object with the result.
        """
        return self.decide(
            input=input,
            options=options,
            prompt=prompt,
            mode="fast",
            sync=True,
            threshold=threshold,
        )

    # ── Bulk decisions ────────────────────────────────────────────────

    def decide_bulk(
        self,
        items: list[dict],
        prompt: Optional[str] = None,
        options: Optional[list[str]] = None,
        runs: int = 7,
        threshold: float = 0.75,
        webhook_url: Optional[str] = None,
    ) -> Batch:
        """Submit up to 100 decisions at once.

        Args:
            items: List of dicts, each with at least an "input" key.
            prompt: Shared prompt applied to all items.
            options: Shared options list applied to all items.
            runs: Number of model runs per decision (default 7).
            threshold: Confidence threshold (default 0.75).
            webhook_url: URL to receive a POST when the batch completes.

        Returns:
            A Batch object for tracking progress.
        """
        payload: dict[str, Any] = {
            "items": items,
            "runs": runs,
            "threshold": threshold,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if options is not None:
            payload["options"] = options
        if webhook_url is not None:
            payload["webhook_url"] = webhook_url

        data = self._request("POST", "/decide/bulk", json=payload)
        return Batch.from_dict(data)

    # ── Decisions ─────────────────────────────────────────────────────

    def get_decision(self, decision_id: str) -> Decision:
        """Retrieve a single decision by ID.

        Args:
            decision_id: The decision's unique identifier.

        Returns:
            A Decision object.
        """
        data = self._request("GET", f"/decisions/{decision_id}")
        return Decision.from_dict(data)

    def list_decisions(self, page: int = 1, per_page: int = 20) -> list[Decision]:
        """List decisions with pagination.

        Args:
            page: Page number (default 1).
            per_page: Results per page (default 20).

        Returns:
            A list of Decision objects.
        """
        data = self._request(
            "GET", "/decisions", params={"page": page, "per_page": per_page}
        )
        items = data if isinstance(data, list) else data.get("decisions", data.get("data", []))
        return [Decision.from_dict(d) for d in items]

    # ── Batches ───────────────────────────────────────────────────────

    def get_batch(self, batch_id: str) -> Batch:
        """Get the status of a batch.

        Args:
            batch_id: The batch's unique identifier.

        Returns:
            A Batch object.
        """
        data = self._request("GET", f"/batches/{batch_id}")
        return Batch.from_dict(data)

    def get_batch_results(self, batch_id: str) -> dict:
        """Get the full results of a completed batch.

        Args:
            batch_id: The batch's unique identifier.

        Returns:
            A dict containing the batch results payload.
        """
        return self._request("GET", f"/batches/{batch_id}/results")

    # ── Reviews ───────────────────────────────────────────────────────

    def list_reviews(self) -> list[Review]:
        """List all pending human reviews.

        Returns:
            A list of Review objects.
        """
        data = self._request("GET", "/reviews")
        items = data if isinstance(data, list) else data.get("reviews", data.get("data", []))
        return [Review.from_dict(r) for r in items]

    def resolve_review(self, review_id: int, answer: str) -> Review:
        """Resolve a pending review by providing a human answer.

        Args:
            review_id: The review's numeric identifier.
            answer: The chosen classification label.

        Returns:
            The updated Review object.
        """
        data = self._request(
            "PATCH", f"/reviews/{review_id}", json={"answer": answer}
        )
        return Review.from_dict(data)

    # ── Models ────────────────────────────────────────────────────────

    def models(self) -> list[dict]:
        """List supported models.

        Returns:
            A list of model info dicts.
        """
        return self._request("GET", "/models")

    # ── Internal ──────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make an API request with error handling.

        Args:
            method: HTTP method (GET, POST, PATCH, etc.).
            path: API path relative to base_url (e.g. "/decide").
            **kwargs: Extra keyword arguments forwarded to requests.

        Returns:
            Parsed JSON response body.

        Raises:
            AuthenticationError: On 401 responses.
            InsufficientCreditsError: On 402 responses.
            RateLimitError: On 429 responses.
            WhiteboxError: On any other non-2xx response.
        """
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)

        try:
            response = self.session.request(method, url, **kwargs)
        except requests.ConnectionError as exc:
            raise WhiteboxError(f"Connection error: {exc}") from exc
        except requests.Timeout as exc:
            raise WhiteboxError(f"Request timed out after {self.timeout}s") from exc
        except requests.RequestException as exc:
            raise WhiteboxError(f"Request failed: {exc}") from exc

        if response.status_code == 401:
            raise AuthenticationError(
                "Invalid or missing API key",
                status_code=401,
                response=response,
            )

        if response.status_code == 402:
            raise InsufficientCreditsError(
                "Insufficient credits — add credits at https://whiteboxhq.ai/billing",
                status_code=402,
                response=response,
            )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    retry_after = int(retry_after)
                except (ValueError, TypeError):
                    pass
            raise RateLimitError(
                "Rate limit exceeded",
                retry_after=retry_after,
                status_code=429,
                response=response,
            )

        if not response.ok:
            try:
                body = response.json()
                message = body.get("error", body.get("message", response.text))
            except (ValueError, KeyError):
                message = response.text
            raise WhiteboxError(
                f"API error {response.status_code}: {message}",
                status_code=response.status_code,
                response=response,
            )

        if response.status_code == 204:
            return {}

        return response.json()
