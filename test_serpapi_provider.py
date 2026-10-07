"""Offline tests for the opt-in SerpAPI provider, live fetch and the two-level cache.

    python -m pytest test_serpapi_provider.py -q

No network and no real key: SerpAPI, the pages and the Wayback Machine are a scripted
fake HTTP client, and sleep/clock are injected.
"""

import asyncio
import dataclasses
import json

import pytest
from openreward.toolsets._web_common import SearchBackendUnavailable, WebFetchParams, WebSearchParams
from openreward.tools.web import format_search_output
from openreward.web_service import RawResponse, WebServiceConfig

import live_web
import web_cache
from backsearch import BrowseCompBackSearch
from live_web import HttpResponse, HttpTransportError, LeakFilter, LiveWeb, map_serpapi
from web_cache import LocalDirStore, TwoLevelCache

KEY = "sk-serp-TEST-0123456789abcdef"
QUESTION = ("Please identify the fictional character who occasionally breaks the fourth wall "
            "with the audience, has a backstory involving help from selfless ascetics.")

SERP_BODY = {
    "search_metadata": {"id": "abc", "status": "Success"},
    "search_parameters": {"engine": "google", "q": "who wrote hamlet"},
    "answer_box": {"title": "Hamlet - Wikipedia", "link": "https://en.wikipedia.org/wiki/Hamlet",
                   "answer": "William Shakespeare"},
    "organic_results": [
        {"position": 1, "title": "Hamlet - Wikipedia", "link": "https://en.wikipedia.org/wiki/Hamlet",
         "snippet": "Hamlet is a tragedy by Shakespeare."},
        {"position": 2, "title": "Hamlet | Folger", "link": "https://www.folger.edu/hamlet",
         "snippet": "Read Hamlet.", "date": "Mar 3, 2024"},
        {"position": 3, "title": "BrowseComp dump", "link": "https://huggingface.co/datasets/x/BrowseComp-Plus",
         "snippet": "all the answers"},
    ],
    "knowledge_graph": {"title": "Hamlet", "description": "Play by Shakespeare",
                        "source": {"name": "Britannica", "link": "https://www.britannica.com/topic/Hamlet"}},
}


def resp(status=200, body=b"", headers=None, url="https://x.example/"):
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    elif isinstance(body, str):
        body = body.encode()
    return HttpResponse(status=status, headers=headers or {}, body=body, url=url)


class FakeHttp:
    """Routes by URL prefix; each route is a list of responses/exceptions replayed in
    order (the last repeats). Records every request."""

    def __init__(self, routes):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls = []
        self.counts = {}

    async def get(self, url, *, params=None, headers=None, timeout_s, max_bytes, allow_redirects):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        for prefix, steps in self.routes.items():
            if url.startswith(prefix):
                n = self.counts.get(prefix, 0)
                self.counts[prefix] = n + 1
                step = steps[min(n, len(steps) - 1)]
                if isinstance(step, Exception):
                    raise step
                return step
        return resp(404, "not found", url=url)

    def n(self, prefix):
        return sum(1 for c in self.calls if c["url"].startswith(prefix))


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    async def sleep(self, s):
        self.now += s


def run(coro):
    return asyncio.run(coro)


def make_live(http, *, tmp_path=None, ttl_s=3600.0, question=QUESTION, secrets=None, clock=None, budget_s=600.0):
    store = LocalDirStore(str(tmp_path)) if tmp_path is not None else None
    wall = clock or FakeTime()
    sc = TwoLevelCache("serpapi-search", store=store, ttl_s=ttl_s, mem_bytes=1 << 20, clock=wall.clock)
    fc = TwoLevelCache("live-fetch", store=store, ttl_s=ttl_s, mem_bytes=1 << 20, clock=wall.clock)
    t = FakeTime()
    return LiveWeb(secrets=secrets if secrets is not None else {"serpapi_api_key": KEY},
                   question=question, task_id="browsecomp_0", http=http, search_cache=sc,
                   fetch_cache=fc, retry_budget_s=budget_s, sleep=t.sleep, clock=t.clock)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("BROWSECOMP_SEARCH_PROVIDER", "SERPAPI_API_KEY", "BROWSECOMP_WEB_CACHE_URL",
              "BROWSECOMP_LEAK_URL_PATTERNS", "BROWSECOMP_LEAK_URL_PATTERNS_EXTRA"):
        monkeypatch.delenv(k, raising=False)
    web_cache.reset_caches()
    yield
    web_cache.reset_caches()


# ---- switch -------------------------------------------------------------------------


def _toolset(env=None):
    cfg = WebServiceConfig.from_env({"api_key": "test-key"})
    cfg = dataclasses.replace(cfg, search_url="https://search.example/search")
    return BrowseCompBackSearch(env, config=cfg)


class _Script:
    def __init__(self, r):
        self.r = r
        self.calls = 0

    async def request(self, method, url, **kw):
        self.calls += 1
        return self.r

    async def aclose(self):
        pass


def test_default_provider_is_backsearch_and_output_unchanged():
    ts = _toolset()
    assert ts._live is None
    ok = RawResponse(200, {}, json.dumps({"hits": [{"title": "T", "url": "https://a.example/x",
                                                    "snippet": "s"}], "mode": "hybrid"}))
    inner = _Script(ok)
    ts._inner_transport = lambda: inner
    out = run(ts.web_search(WebSearchParams(query="some query")))
    assert out.blocks[0].text == format_search_output(
        "some query", [{"title": "T", "url": "https://a.example/x", "snippet": "s"}], include_snippets=True)
    assert out.metadata == {"query": "some query", "hits": [{"title": "T", "url": "https://a.example/x",
                                                              "snippet": "s"}], "mode": "hybrid"}
    assert inner.calls == 1


def test_switch_selects_serpapi(monkeypatch):
    monkeypatch.setenv("BROWSECOMP_SEARCH_PROVIDER", "serpapi")
    env = type("E", (), {"search_secrets": {"serpapi_api_key": KEY},
                         "config": type("C", (), {"problem": QUESTION, "id": "browsecomp_7"})()})()
    ts = _toolset(env)
    assert isinstance(ts._live, LiveWeb) and ts._live.task_id == "browsecomp_7"
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    ts._live.http = http
    ts._live.search_cache = TwoLevelCache("s", store=None, ttl_s=60, mem_bytes=1 << 20)
    out = run(ts.web_search(WebSearchParams(query="who wrote hamlet")))
    assert out.metadata["provider"] == "serpapi" and http.n(live_web.SERPAPI_URL) == 1
    # Backsearch transport never touched.
    ts._inner_transport = lambda: (_ for _ in ()).throw(AssertionError("backsearch used"))


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("BROWSECOMP_SEARCH_PROVIDER", "tavily")
    with pytest.raises(ValueError):
        _toolset()


def test_key_resolution_rejects_platform_token(monkeypatch):
    token = "__SECRET:sid123:serpapi_api_key__"
    key, why = live_web.resolve_serpapi_key({"serpapi_api_key": token})
    assert key is None and "env_overrides" in why
    monkeypatch.setenv("SERPAPI_API_KEY", KEY)
    assert live_web.resolve_serpapi_key({"serpapi_api_key": token}) == (KEY, "env")
    assert live_web.resolve_serpapi_key({"serpapi_api_key": "real"}) == ("real", "secret")


# ---- search -------------------------------------------------------------------------


def test_one_request_per_call_with_expected_params():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    lw = make_live(http)
    out = run(lw.search("who wrote hamlet"))
    assert http.n(live_web.SERPAPI_URL) == 1
    p = http.calls[0]["params"]
    assert p == {"engine": "google", "q": "who wrote hamlet", "num": "10", "api_key": KEY}
    assert out.metadata["serpapi_requests"] == 1 and out.metadata["cache_hit"] is False
    assert out.reward is None and not out.finished


def test_output_format_matches_backsearch_envelope():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    out = run(make_live(http).search("who wrote hamlet"))
    expected_hits = [
        {"title": "Hamlet - Wikipedia", "url": "https://en.wikipedia.org/wiki/Hamlet", "snippet": "William Shakespeare"},
        {"title": "Hamlet | Folger", "url": "https://www.folger.edu/hamlet", "snippet": "Mar 3, 2024 — Read Hamlet."},
        {"title": "Hamlet", "url": "https://www.britannica.com/topic/Hamlet", "snippet": "Play by Shakespeare"},
    ]
    # Byte-identical to what backsearch renders for the same hits.
    assert out.blocks[0].text == format_search_output("who wrote hamlet", expected_hits, include_snippets=True)
    assert out.metadata["query"] == "who wrote hamlet"


def test_map_caps_and_dedupes():
    body = {"organic_results": [{"title": f"t{i}", "link": f"https://e{i}.example/", "snippet": "s"} for i in range(15)]
            + [{"title": "dup", "link": "https://e0.example/"}]}
    hits = map_serpapi(body, 10)
    assert len(hits) == 10 and len({h["url"] for h in hits}) == 10


def test_no_results_is_an_empty_success():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, {"error": "Google hasn't returned any results for this query."})]})
    out = run(make_live(http).search("zzqx nothing"))
    assert "No links found." in out.blocks[0].text and "error" not in out.metadata


def test_429_and_5xx_are_retried_and_counted():
    t = FakeTime()
    http = FakeHttp({live_web.SERPAPI_URL: [resp(429, {"error": "rate"}, {"Retry-After": "3"}), resp(503, ""),
                                            HttpTransportError("timed out after 30s"), resp(200, SERP_BODY)]})
    lw = make_live(http)
    lw._sleep, lw._clock = t.sleep, t.clock
    out = run(lw.search("who wrote hamlet"))
    assert http.n(live_web.SERPAPI_URL) == 4 and out.metadata["serpapi_requests"] == 4
    assert t.now >= 3


def test_persistent_failure_raises_provider_unavailable():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(503, "")]})
    with pytest.raises(SearchBackendUnavailable) as ei:
        run(make_live(http, budget_s=30).search("who wrote hamlet"))
    assert ei.value.error_code == "provider-unavailable"


def test_out_of_searches_is_fatal_without_retry():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(429, {"error": "Your account has run out of searches."})]})
    with pytest.raises(SearchBackendUnavailable) as ei:
        run(make_live(http).search("who wrote hamlet"))
    assert ei.value.error_code == "quota-exceeded" and http.n(live_web.SERPAPI_URL) == 1


def test_bad_key_is_fatal_and_missing_key_is_not_configured():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(401, {"error": "Invalid API key. Your API key should be here"})]})
    with pytest.raises(SearchBackendUnavailable) as ei:
        run(make_live(http).search("who wrote hamlet"))
    assert ei.value.error_code == "not-configured"
    with pytest.raises(SearchBackendUnavailable) as ei:
        run(make_live(FakeHttp({}), secrets={}).search("who wrote hamlet"))
    assert ei.value.error_code == "not-configured"


def test_bad_request_is_soft():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(400, {"error": "Missing query `q` parameter."})]})
    out = run(make_live(http).search("who wrote hamlet"))
    assert out.metadata["error"] == "search-failed" and "Search error" in out.blocks[0].text


def test_domain_filters_become_site_operators():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    out = run(make_live(http).search("hamlet", allowed_domains=["wikipedia.org"]))
    assert http.calls[0]["params"]["q"] == "hamlet (site:wikipedia.org)"
    assert [h["url"] for h in out.metadata["hits"]] == ["https://en.wikipedia.org/wiki/Hamlet"]


def test_key_never_in_outputs_logs_or_cache(tmp_path, capfd, caplog):
    caplog.set_level("INFO", logger="browsecomp")
    http = FakeHttp({live_web.SERPAPI_URL: [resp(500, f"upstream echoed api_key={KEY}"),
                                            resp(200, {**SERP_BODY, "search_parameters": {"api_key": KEY}})]})
    lw = make_live(http, tmp_path=tmp_path)
    out = run(lw.search("who wrote hamlet"))
    with pytest.raises(SearchBackendUnavailable) as ei:
        run(make_live(FakeHttp({live_web.SERPAPI_URL: [resp(429, {"error": f"bad {KEY}"})]}), budget_s=1)
            .search("other query"))
    assert KEY not in str(ei.value)
    assert KEY not in json.dumps(out.metadata) and KEY not in out.blocks[0].text
    captured = capfd.readouterr()
    assert KEY not in captured.out + captured.err + caplog.text
    assert "serpapi_search" in caplog.text
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert KEY not in f.read_text()


# ---- leak blocklist -------------------------------------------------------------


def test_leak_filter_drops_hits_and_logs(caplog):
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    lw = make_live(http)
    out = run(lw.search("who wrote hamlet"))
    assert "huggingface.co" not in out.blocks[0].text
    assert out.metadata["web_usage"]["blocked"] == 1
    assert "browsecomp_leak_block" in caplog.text


def test_leak_rules():
    lf = LeakFilter(QUESTION, patterns=list(live_web.DEFAULT_LEAK_URL_PATTERNS))
    assert lf.url_rule("https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py")
    assert lf.url_rule("https://raw.githubusercontent.com/x/y/main/Browse_Comp/data.csv")
    assert lf.url_rule("https://huggingface.co/datasets/Tevatron/browsecomp-plus")
    assert lf.url_rule("https://en.wikipedia.org/wiki/Hamlet") is None
    assert lf.text_rule("dump: " + QUESTION.upper() + " answer: X") == "question-text"
    assert lf.snippet_rule("who occasionally breaks the fourth wall with the audience, has a backstory involving help...")
    assert lf.snippet_rule("breaks the fourth wall") is None  # too short to be a copy


def test_leak_patterns_configurable(monkeypatch):
    monkeypatch.setenv("BROWSECOMP_LEAK_URL_PATTERNS_EXTRA", json.dumps([r"answers\.example"]))
    lf = LeakFilter(QUESTION)
    assert lf.url_rule("https://answers.example/q") and lf.url_rule("https://huggingface.co/BrowseComp")


# ---- fetch --------------------------------------------------------------------------

HTML = ("<html><head><title>Hamlet</title></head><body><article><h1>Hamlet</h1>"
        + "<p>Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601. "
        "It is Shakespeare's longest play.</p>" * 20 + "</article></body></html>")


def test_live_fetch_renders_like_backsearch_and_caches(tmp_path):
    url = "https://www.folger.edu/hamlet"
    http = FakeHttp({url: [resp(200, HTML, {"Content-Type": "text/html; charset=utf-8"}, url=url)]})
    lw = make_live(http, tmp_path=tmp_path)
    out = run(lw.fetch(url, "who wrote it"))
    text = out.blocks[0].text
    assert text.startswith(f"Fetched content from {url}:\n\n") and "William Shakespeare" in text
    assert out.metadata["url"] == url and out.metadata["prompt"] == "who wrote it"
    assert "content" not in out.metadata and out.metadata["cache_hit"] is False
    assert out.metadata["live_requests"] == 1
    out2 = run(lw.fetch(url + "#section", "again"))
    assert out2.metadata["cache_hit"] is True and http.n(url) == 1
    assert out2.blocks[0].text.split("\n\n", 1)[1] == text.split("\n\n", 1)[1]
    # A fresh process (new in-memory level) reads the persistent level.
    out3 = run(make_live(FakeHttp({}), tmp_path=tmp_path).fetch(url, "x"))
    assert out3.metadata["cache_level"] == "persistent"


def test_fetch_falls_back_to_wayback_on_block():
    url = "https://blocked.example/page"
    challenge = "<html><title>Just a moment...</title><body>Enable JavaScript and cookies to continue</body></html>"
    http = FakeHttp({
        url: [resp(403, challenge, {"Content-Type": "text/html", "cf-mitigated": "challenge"}, url=url)],
        live_web.WAYBACK_PREFIX: [resp(200, HTML, {"Content-Type": "text/html"},
                                       url=f"https://web.archive.org/web/20240102030405id_/{url}")],
    })
    out = run(make_live(http).fetch(url, "p"))
    text = out.blocks[0].text
    assert "[Archived copy:" in text and "2024-01-02" in text and "HTTP 403" in text
    assert out.metadata["fetch_source"] == "wayback" and out.metadata["wayback_requests"] == 1
    assert out.metadata["web_usage"]["wayback_served"] == 1
    assert http.calls[-1]["url"].startswith("https://web.archive.org/web/") and "id_/" + url in http.calls[-1]["url"]


def test_fetch_challenge_page_with_200_falls_back():
    url = "https://cf.example/a"
    page = "<html><head><title>Just a moment...</title></head><body><script src='/cdn-cgi/challenge-platform/x.js'></script></body></html>"
    http = FakeHttp({url: [resp(200, page, {"Content-Type": "text/html"}, url=url)],
                     live_web.WAYBACK_PREFIX: [resp(404, "", url="https://web.archive.org/x")]})
    out = run(make_live(http).fetch(url, "p"))
    assert out.metadata["error"] == "fetch-error" and "no Wayback Machine capture" in out.blocks[0].text


def test_fetch_never_uses_backsearch(monkeypatch):
    monkeypatch.setenv("BROWSECOMP_SEARCH_PROVIDER", "serpapi")
    env = type("E", (), {"search_secrets": {"serpapi_api_key": KEY},
                         "config": type("C", (), {"problem": QUESTION, "id": "b"})()})()
    ts = _toolset(env)
    ts._inner_transport = lambda: (_ for _ in ()).throw(AssertionError("backsearch used"))
    url = "https://gone.example/x"
    ts._live.http = FakeHttp({url: [HttpTransportError("ClientConnectorError")],
                              live_web.WAYBACK_PREFIX: [HttpTransportError("timed out")]})
    ts._live.fetch_cache = TwoLevelCache("f", store=None, ttl_s=60, mem_bytes=1 << 20)
    out = run(ts.web_fetch(WebFetchParams(url=url, prompt="p")))
    assert out.metadata["error"] == "fetch-error"


def test_fetch_cross_host_redirect_uses_sdk_template():
    url = "https://a.example/x"
    http = FakeHttp({url: [resp(301, "", {"Location": "https://b.example/y"}, url=url)]})
    out = run(make_live(http).fetch(url, "p"))
    assert out.blocks[0].text.startswith("REDIRECT DETECTED") and "https://b.example/y" in out.blocks[0].text


def test_fetch_pdf():
    pypdf = pytest.importorskip("pypdf")
    import io

    w = pypdf.PdfWriter()
    w.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    w.write(buf)
    url = "https://papers.example/p.pdf"
    http = FakeHttp({url: [resp(200, buf.getvalue(), {"Content-Type": "application/pdf"}, url=url)],
                     live_web.WAYBACK_PREFIX: [resp(404, "")]})
    # A blank PDF has no text: treated as an empty page, so Wayback is tried.
    out = run(make_live(http).fetch(url, "p"))
    assert http.n(live_web.WAYBACK_PREFIX) == 1 and out.metadata["error"] == "fetch-error"
    monkey = live_web._extract_pdf
    try:
        live_web._extract_pdf = lambda body, limit: "PDF TEXT"
        out = run(make_live(FakeHttp({url: [resp(200, buf.getvalue(), {"Content-Type": "application/pdf"}, url=url)]}))
                  .fetch(url, "p"))
    finally:
        live_web._extract_pdf = monkey
    assert out.blocks[0].text == f"Fetched content from {url}:\n\nPDF TEXT"


def test_fetch_blocks_leaky_url_and_question_text():
    http = FakeHttp({})
    lw = make_live(http)
    out = run(lw.fetch("https://huggingface.co/datasets/openai/BrowseComp", "p"))
    assert out.metadata["error"] == "domain-blocked" and http.calls == []
    url = "https://leak.example/dump"
    page = f"<html><body><article><p>Question: {QUESTION}</p><p>Answer: Foo</p>" + "<p>filler text here.</p>" * 30 + "</article></body></html>"
    http = FakeHttp({url: [resp(200, page, {"Content-Type": "text/html"}, url=url)]})
    out = run(make_live(http).fetch(url, "p"))
    assert out.metadata["error"] == "domain-blocked"


def test_fetch_truncates_like_sdk():
    url = "https://long.example/t"
    http = FakeHttp({url: [resp(200, "a" * 150_000, {"Content-Type": "text/plain"}, url=url)]})
    out = run(make_live(http).fetch(url, "p"))
    assert out.blocks[0].text.endswith("\n... (truncated)")
    assert len(out.blocks[0].text) == len(f"Fetched content from {url}:\n\n") + 100_000 + len("\n... (truncated)")


# ---- cache --------------------------------------------------------------------------


def test_identical_queries_cost_one_request(tmp_path):
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    lw = make_live(http, tmp_path=tmp_path)
    a = run(lw.search("Who  wrote Hamlet"))
    b = run(lw.search("who wrote hamlet"))
    assert http.n(live_web.SERPAPI_URL) == 1
    assert a.metadata["cache_hit"] is False and b.metadata["cache_hit"] is True
    assert b.metadata["serpapi_requests"] == 0 and b.metadata["cache_level"] == "memory"
    assert b.metadata["web_usage"]["serpapi_requests"] == 1 and b.metadata["web_usage"]["search_cache_hits"] == 1
    # Another session / process reads the persistent level and still pays nothing.
    http2 = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    c = run(make_live(http2, tmp_path=tmp_path).search("who wrote hamlet"))
    assert http2.calls == [] and c.metadata["cache_level"] == "persistent"


def test_concurrent_identical_queries_share_one_request():
    class Slow(FakeHttp):
        async def get(self, *a, **kw):
            await asyncio.sleep(0.01)
            return await super().get(*a, **kw)

    http = Slow({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    lw = make_live(http)

    async def both():
        return await asyncio.gather(lw.search("q one"), lw.search("q one"))

    a, b = run(both())
    assert http.n(live_web.SERPAPI_URL) == 1
    assert sorted([a.metadata["serpapi_requests"], b.metadata["serpapi_requests"]]) == [0, 1]


def test_ttl_expiry(tmp_path):
    wall = FakeTime()
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    lw = make_live(http, tmp_path=tmp_path, ttl_s=100, clock=wall)
    run(lw.search("who wrote hamlet"))
    wall.now += 50
    assert run(lw.search("who wrote hamlet")).metadata["cache_hit"] is True
    wall.now += 100
    out = run(lw.search("who wrote hamlet"))
    assert out.metadata["cache_hit"] is False and http.n(live_web.SERPAPI_URL) == 2


def test_failures_are_not_cached():
    http = FakeHttp({live_web.SERPAPI_URL: [resp(400, {"error": "bad"}), resp(200, SERP_BODY)]})
    lw = make_live(http)
    run(lw.search("who wrote hamlet"))
    out = run(lw.search("who wrote hamlet"))
    assert out.metadata["cache_hit"] is False and http.n(live_web.SERPAPI_URL) == 2


def test_cache_key_excludes_key_and_store_errors_are_misses(tmp_path):
    class Broken:
        async def get(self, path):
            raise OSError("down")

        async def put(self, path, data):
            raise OSError("down")

    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    lw = make_live(http)
    lw.search_cache = TwoLevelCache("s", store=Broken(), ttl_s=60, mem_bytes=1 << 20)
    out = run(lw.search("who wrote hamlet"))
    assert out.metadata["serpapi_requests"] == 1 and KEY not in out.metadata["cache_key"]
    assert out.metadata["cache_key"] == web_cache.cache_key({"engine": "google", "q": "who wrote hamlet", "num": 10})


def test_store_from_url():
    assert web_cache.store_from_url(None) is None
    assert isinstance(web_cache.store_from_url("file:///tmp/x"), LocalDirStore)
    gcs = web_cache.store_from_url("gs://bkt/some/prefix")
    assert isinstance(gcs, web_cache.GCSStore) and gcs.bucket == "bkt" and gcs.prefix == "some/prefix"
    with pytest.raises(ValueError):
        web_cache.store_from_url("s3://nope")


def test_serpapi_variant_class_pins_the_provider_without_the_env_var(monkeypatch):
    # The eval selects BrowseCompSerpApi as a variant; it must not depend on the env var.
    monkeypatch.delenv("BROWSECOMP_SEARCH_PROVIDER", raising=False)
    env = type("E", (), {"SEARCH_PROVIDER": "serpapi", "search_secrets": {"serpapi_api_key": KEY},
                         "config": type("C", (), {"problem": QUESTION, "id": "browsecomp_9"})()})()
    ts = _toolset(env)
    assert isinstance(ts._live, LiveWeb) and ts._live.task_id == "browsecomp_9"
    http = FakeHttp({live_web.SERPAPI_URL: [resp(200, SERP_BODY)]})
    ts._live.http = http
    ts._live.search_cache = TwoLevelCache("s", store=None, ttl_s=60, mem_bytes=1 << 20)
    out = run(ts.web_search(WebSearchParams(query="who wrote hamlet")))
    assert out.metadata["provider"] == "serpapi" and http.n(live_web.SERPAPI_URL) == 1


def test_default_class_keeps_backsearch_without_the_env_var(monkeypatch):
    monkeypatch.delenv("BROWSECOMP_SEARCH_PROVIDER", raising=False)
    env = type("E", (), {"SEARCH_PROVIDER": None, "search_secrets": {},
                         "config": type("C", (), {"problem": QUESTION, "id": "browsecomp_9"})()})()
    assert _toolset(env)._live is None


def test_server_registers_the_serpapi_variant():
    # Parsed, not imported: importing server.py loads the question CSV.
    import ast, pathlib
    tree = ast.parse(pathlib.Path(__file__).with_name("server.py").read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    variant = classes["BrowseCompSerpApi"]
    assert [b.id for b in variant.bases] == ["BrowseComp"]
    pinned = [a for a in variant.body if isinstance(a, ast.Assign) and a.targets[0].id == "SEARCH_PROVIDER"]
    assert pinned and pinned[0].value.value == "serpapi"
    served = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Server"]
    assert [e.id for e in served[0].args[0].elts] == ["BrowseComp", "BrowseCompSerpApi"]
