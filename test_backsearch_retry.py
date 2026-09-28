"""Offline tests for PatientTransport and the toolset's give-up path.

    python -m pytest test_backsearch_retry.py -q

No network: the inner transport is a scripted fake, and sleep/clock are injected so a
600 s budget runs instantly.
"""

import asyncio
import dataclasses
import json

import pytest
from openreward.toolsets._web_common import SearchBackendUnavailable, WebSearchParams
from openreward.web_service import RawResponse, WebServiceConfig, WebTransportError

from backsearch import BrowseCompBackSearch, PatientTransport

SEARCH_URL = "https://search.example/search"
FETCH_URL = "https://search.example/fetch"
OK = RawResponse(200, {}, json.dumps({"hits": [{"title": "T", "url": "https://a.example/x",
                                                "snippet": "s"}], "mode": "hybrid"}))
DEGRADED = RawResponse(200, {}, json.dumps({"hits": [], "degraded": True,
                                            "corpora_missing": ["cc_web"]}))
BUSY = RawResponse(429, {"Retry-After": "7"}, '{"detail": "server overloaded"}')


class Script:
    """Inner transport that replays a list of responses (or exceptions), then repeats
    the last one."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    async def request(self, method, url, **kwargs):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return step

    async def aclose(self):
        pass


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    async def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def patient(inner, budget_s=600.0):
    t = FakeTime()
    return PatientTransport(inner, search_url=SEARCH_URL, budget_s=budget_s,
                            sleep=t.sleep, clock=t.clock), t


def run(coro):
    return asyncio.run(coro)


def test_waits_out_429_and_honours_retry_after():
    p, t = patient(Script(BUSY, BUSY, OK))
    resp = run(p.request("POST", SEARCH_URL))
    assert resp is OK and p.gave_up is None and p.retries == 2
    assert all(s >= 7 for s in t.sleeps)


def test_retries_degraded_search_until_clean():
    p, _ = patient(Script(DEGRADED, OK))
    assert run(p.request("POST", SEARCH_URL)) is OK and p.retries == 1


def test_degraded_body_on_another_url_is_not_retried():
    p, _ = patient(Script(DEGRADED, OK))
    assert run(p.request("POST", FETCH_URL)) is DEGRADED
    assert p.retries == 0


def test_client_error_is_returned_immediately():
    bad = RawResponse(400, {}, '{"detail": "bad query"}')
    p, _ = patient(Script(bad))
    assert run(p.request("POST", SEARCH_URL)) is bad and p.retries == 0


def test_transport_error_is_retried():
    p, _ = patient(Script(WebTransportError("reset"), OK))
    assert run(p.request("POST", SEARCH_URL)) is OK and p.retries == 1


def test_gives_up_within_budget_then_short_circuits():
    inner = Script(DEGRADED)
    p, t = patient(inner, budget_s=120)
    assert run(p.request("POST", SEARCH_URL)) is DEGRADED
    assert p.gave_up and "degraded" in p.gave_up
    assert t.now <= 120
    calls = inner.calls
    # Later requests (the SDK's own retry loop) must not spend the budget again.
    assert run(p.request("POST", SEARCH_URL)) is DEGRADED and inner.calls == calls



def test_fetch_waits_out_overload_only():
    p, _ = patient(Script(BUSY, OK))
    assert run(p.request("POST", FETCH_URL)) is OK and p.retries == 1
    page_error = RawResponse(502, {}, "")
    p, _ = patient(Script(page_error, OK))
    assert run(p.request("POST", FETCH_URL)) is page_error and p.retries == 0


def test_fetch_transport_error_is_not_retried():
    p, _ = patient(Script(WebTransportError("slow page"), OK))
    with pytest.raises(WebTransportError):
        run(p.request("POST", FETCH_URL))
    assert p.retries == 0 and p.gave_up is None


def test_backoff_is_capped_and_never_near_zero():
    p, t = patient(Script(RawResponse(503, {}, "")), budget_s=3600)
    run(p.request("POST", SEARCH_URL))
    assert t.sleeps[0] >= 1.0  # base 2 s, equal jitter -> at least half
    assert max(t.sleeps) <= 60.0


# ---- toolset end to end -----------------------------------------------------------


def toolset(inner, budget_s=600.0):
    cfg = WebServiceConfig.from_env({"api_key": "test-key"})
    cfg = dataclasses.replace(cfg, search_url=SEARCH_URL)
    ts = BrowseCompBackSearch(None, config=cfg)
    ts.retry_budget_s = budget_s
    ts._inner_transport = lambda: inner
    return ts


@pytest.fixture(autouse=True)
def instant_sleep(monkeypatch):
    async def no_sleep(_s):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)


def test_web_search_recovers_after_overload():
    out = run(toolset(Script(BUSY, OK)).web_search(WebSearchParams(query="some query")))
    assert "a.example" in out.blocks[0].text and out.reward is None


def test_web_search_raises_when_backend_stays_degraded():
    with pytest.raises(SearchBackendUnavailable) as ei:
        run(toolset(Script(DEGRADED), budget_s=5).web_search(WebSearchParams(query="some query")))
    assert "degraded" in str(ei.value)


def test_web_search_raises_when_backend_stays_overloaded():
    with pytest.raises(SearchBackendUnavailable):
        run(toolset(Script(BUSY), budget_s=5).web_search(WebSearchParams(query="some query")))
