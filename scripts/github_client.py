"""
Thin GitHub REST API client.

Handles:
  - authentication via GITHUB_TOKEN environment variable
  - automatic rate-limit back-off
  - pagination via Link headers
  - transient-error retries (5xx, connection resets)
"""

import os
import time
import logging
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_BASE = "https://api.github.com"


def _make_session() -> requests.Session:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        log.warning(
            "GITHUB_TOKEN not set — unauthenticated requests are capped at "
            "60 req/hr.  Set the token in .env or as an environment variable."
        )

    sess = requests.Session()
    sess.headers.update(headers)

    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


_sess = _make_session()


def _handle_rate_limit(resp: requests.Response) -> None:
    """Sleep until the rate limit resets if we are running low."""
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 100))
    if remaining < 5:
        reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset_ts - time.time() + 2, 1)
        log.info(
            "Rate limit nearly exhausted (%d requests left). "
            "Sleeping %.0f s until reset.",
            remaining, wait,
        )
        time.sleep(wait)


def get(path: str, params: dict | None = None) -> Any:
    """Single authenticated GET.  Returns parsed JSON; raises on HTTP errors."""
    url = path if path.startswith("http") else f"{_BASE}{path}"
    resp = _sess.get(url, params=params, timeout=30)
    _handle_rate_limit(resp)
    resp.raise_for_status()
    return resp.json()


def paginate(path: str, params: dict | None = None) -> Iterator[list[dict]]:
    """
    Yield each page (a list of items) from a paginated endpoint.

    Follows 'Link: rel="next"' headers automatically.  The first request
    always includes per_page=100; subsequent requests use the full next-page
    URL (which already encodes all parameters).
    """
    url = path if path.startswith("http") else f"{_BASE}{path}"
    current_params: dict | None = {"per_page": 100, **(params or {})}

    while url:
        resp = _sess.get(url, params=current_params, timeout=30)
        _handle_rate_limit(resp)
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, list):
            yield data
        else:
            # Some endpoints wrap items in a dict (e.g. search results).
            yield data.get("items", [])

        # Parse the Link header for the next page URL.
        url = None
        current_params = None   # next-page URL already contains all params
        for part in resp.headers.get("Link", "").split(","):
            part = part.strip()
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
                break
