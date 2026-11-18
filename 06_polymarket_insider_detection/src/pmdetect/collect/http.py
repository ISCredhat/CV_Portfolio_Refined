"""Throttled, retrying, caching JSON GET helper."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "raw_cache"
MIN_INTERVAL = 0.2  # ~5 req/s
TIMEOUT = 30.0
MAX_429_SLEEP = 120.0

_lock = threading.Lock()
_last_request = 0.0
_blocked_until = 0.0  # shared 429 back-off: every thread waits, not just the one that was limited
_client: httpx.Client | None = None


class RetryableHTTPError(Exception):
    pass


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=TIMEOUT,
            headers={"User-Agent": "pmdetect/0.1 (research)", "Accept": "application/json"},
            follow_redirects=True,
        )
    return _client


def _set_blocked(delay: float) -> None:
    global _blocked_until
    with _lock:
        _blocked_until = max(_blocked_until, time.monotonic() + delay)


def _throttle() -> None:
    global _last_request
    with _lock:
        now = time.monotonic()
        wait = max(MIN_INTERVAL - (now - _last_request), _blocked_until - now)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def cache_key(url: str, params: dict | None) -> str:
    items = sorted((str(k), str(v)) for k, v in (params or {}).items() if v is not None)
    return hashlib.sha256(json.dumps([url, items]).encode()).hexdigest()


def _cache_path(url: str, params: dict | None) -> Path:
    return CACHE_DIR / f"{cache_key(url, params)}.json"


@retry(
    retry=retry_if_exception_type(
        (RetryableHTTPError, httpx.TimeoutException, httpx.TransportError)
    ),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def _fetch(url: str, params: dict | None) -> Any:
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    while True:
        _throttle()
        resp = _get_client().get(url, params=clean)
        if resp.status_code == 429:
            try:
                delay = float(resp.headers.get("Retry-After", "1"))
            except ValueError:
                delay = 1.0
            _set_blocked(min(max(delay, 0.0), MAX_429_SLEEP))
            continue
        if resp.status_code >= 500:
            raise RetryableHTTPError(f"{resp.status_code} for {url}")
        resp.raise_for_status()
        return resp.json()


def get_json(url: str, params: dict | None = None, *, use_cache: bool = True) -> Any:
    """GET url (+params) as JSON; throttled, retried, cached under data/raw_cache/."""
    path = _cache_path(url, params)
    if use_cache and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    data = _fetch(url, params)
    if not use_cache:  # bulk pages: the caller stores them (Parquet); never write a cache file
        return data
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
    return data


def paginate_cursor(
    url: str,
    params: dict,
    *,
    data_key: str = "data",
    cursor_param: str = "cursor",
    use_cache: bool = True,
) -> Iterator[dict]:
    """Yield rows across cursor pages until pagination.next_cursor is null."""
    cursor: str | None = None
    while True:
        p = dict(params)
        if cursor:
            p[cursor_param] = cursor
        page = get_json(url, p, use_cache=use_cache)
        rows = page.get(data_key) if isinstance(page, dict) else page
        yield from rows or []
        pag = (page.get("pagination") if isinstance(page, dict) else None) or {}
        cursor = pag.get("next_cursor")
        if not cursor:
            return

