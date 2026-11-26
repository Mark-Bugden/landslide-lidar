"""The only way this repo talks to the network (BRIEF.md: Requests; PLAN.md §3 rule 4).

- Only the listed public services are contacted (host allowlist).
- `outSR` is never sent: query in native EPSG:5514 or EPSG:4326 and transform locally with pyproj.
- An explicit User-Agent, at least 0.3 s between requests, and retries with exponential backoff.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "landslide-lidar-replication (+https://github.com/Mark-Bugden/landslide-lidar)"
ALLOWED_HOSTS = frozenset({"ags.cuzk.gov.cz", "mapy.geology.cz"})
MIN_INTERVAL_S = 0.3


class ForbiddenRequest(ValueError):
    """A request that the brief's rules forbid (outSR, or a host outside the allowlist)."""


def _check_params(params: Mapping[str, Any] | None) -> None:
    if not params:
        return
    for key in params:
        if str(key).strip().lower() == "outsr":
            raise ForbiddenRequest("outSR must never be passed; transform locally with pyproj")


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise ForbiddenRequest(f"host not in the allowlist {sorted(ALLOWED_HOSTS)}: {url}")
    _check_params(dict(parse_qsl(parts.query)))


class Client:
    """A rate-limited, retrying session for the ČÚZK and ČGS ArcGIS services."""

    def __init__(self, min_interval_s: float = MIN_INTERVAL_S, retries: int = 5, backoff_s: float = 2.0,
                 timeout_s: float = 120.0) -> None:
        if min_interval_s < MIN_INTERVAL_S:
            raise ValueError(f"min_interval_s must be >= {MIN_INTERVAL_S}")
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last_end = 0.0
        self.n_requests = 0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        retry = Retry(total=retries, connect=retries, read=retries, status=retries, backoff_factor=backoff_s,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "POST"),
                      raise_on_status=False, respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    def _wait(self) -> None:
        dt = time.monotonic() - self._last_end
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)

    def request(self, method: str, url: str, params: Mapping[str, Any] | None = None,
                data: Mapping[str, Any] | None = None, **kw: Any) -> requests.Response:
        _check_url(url)
        _check_params(params)
        _check_params(data)
        self._wait()
        try:
            resp = self.session.request(method, url, params=params, data=data,
                                        timeout=kw.pop("timeout", self.timeout_s), **kw)
        finally:
            self._last_end = time.monotonic()
            self.n_requests += 1
        resp.raise_for_status()
        return resp

    def get(self, url: str, params: Mapping[str, Any] | None = None, **kw: Any) -> requests.Response:
        return self.request("GET", url, params=params, **kw)

    def post(self, url: str, data: Mapping[str, Any] | None = None, **kw: Any) -> requests.Response:
        return self.request("POST", url, data=data, **kw)

    def get_json(self, url: str, params: Mapping[str, Any] | None = None, **kw: Any) -> Any:
        """GET and decode JSON. ArcGIS reports errors as HTTP 200 with an `error` member; raise on those."""
        body = self.get(url, params=params, **kw).json()
        if isinstance(body, dict) and "error" in body:
            raise RuntimeError(f"ArcGIS error from {url}: {body['error']}")
        return body

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
