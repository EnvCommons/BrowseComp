"""Optional live-web provider for BrowseComp: SerpAPI (Google) search + live fetch.

Selected per process with ``BROWSECOMP_SEARCH_PROVIDER=serpapi``; the default
(``backsearch``) never imports anything from here that changes behaviour. The switch
moves BOTH tools, because Google returns URLs the backdated archive mostly does not
hold:

* ``web_search`` makes exactly one SerpAPI request per call (``engine=google``,
  ``num=10``) unless the two-level cache (``web_cache``) already holds the identical
  query. Retries happen only after a failed request (429/5xx/timeout) and every
  attempt is counted.
* ``web_fetch`` GETs the page live from the env pod, extracts HTML with trafilatura
  (the same extractor and settings the backsearch fetch service uses) and PDFs with
  pypdf, and falls back to the Wayback Machine when the site blocks us or the page is
  gone. It never falls back to the backsearch archive.

Both tools render through the SDK's own formatters, so the agent sees the same
envelope as with backsearch. A leak blocklist drops search hits and refuses fetches
that would publish BrowseComp questions or answers.

The API key: the platform hands session secrets to the env as opaque
``__SECRET:sid:key__`` tokens and its egress proxy substitutes the real value only in
request HEADERS. SerpAPI only accepts the key as the ``api_key`` query parameter, so a
token cannot work there. The key is therefore read from the ``serpapi_api_key`` session
secret when that holds a real value (local runs), else from the ``SERPAPI_API_KEY``
process environment variable (set per session pool with ``env_overrides``). It is
never logged, cached or returned; any text that could carry it is scrubbed.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from openreward.environments import TextBlock, ToolOutput
from openreward.toolsets._web_common import SearchBackendUnavailable, to_tool_output
from openreward.tools.web import (
    MAX_FETCH_TEXT_CHARS,
    WebToolResult,
    _is_same_host_redirect,
    _redirect_template,
    format_search_output,
    render_fetch_text,
    validate_fetch_url,
    validate_search_input,
)
from openreward.web_service import lookup_secret

from web_cache import TwoLevelCache, cache_key, get_cache

log = logging.getLogger("browsecomp.live_web")


def _ensure_log_output() -> None:
    """Usage and block lines must reach the pod log so a run's spend can be audited;
    the root logger's default WARNING level would drop the INFO ones."""
    root = logging.getLogger("browsecomp")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False


_ensure_log_output()

PROVIDER_ENV = "BROWSECOMP_SEARCH_PROVIDER"
PROVIDERS = ("backsearch", "serpapi")

SERPAPI_URL = "https://serpapi.com/search.json"
SERPAPI_ENGINE = "google"
SERPAPI_NUM = 10
SERPAPI_TIMEOUT_S = 30.0
SERPAPI_MAX_BYTES = 5 * 1024 * 1024

FETCH_TIMEOUT_S = 30.0
FETCH_MAX_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 10
PDF_MAX_PAGES = 300
WAYBACK_PREFIX = "https://web.archive.org/web/"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)
FETCH_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # No "br": aiohttp only decodes brotli when the optional package is installed.
    "Accept-Encoding": "gzip, deflate",
}

_SECRET_TOKEN_RE = re.compile(r"__SECRET:[^:]+:.+?__")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


def search_provider() -> str:
    """The configured provider. An unknown value raises rather than silently running
    backsearch, because a mislabelled run would invalidate the comparison."""
    value = (os.environ.get(PROVIDER_ENV) or "backsearch").strip().lower()
    if value not in PROVIDERS:
        raise ValueError(f"{PROVIDER_ENV}={value!r} is not one of {PROVIDERS}")
    return value


def resolve_serpapi_key(secrets: Optional[Mapping[str, Any]]) -> tuple[Optional[str], str]:
    """``(key, source)``; key is None when no usable key exists, and source then says
    why. An opaque platform token is not usable (see module docstring)."""
    secret = lookup_secret(secrets, "serpapi_api_key")
    if secret and not _SECRET_TOKEN_RE.search(secret):
        return secret, "secret"
    env_key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
    if env_key:
        return env_key, "env"
    if secret:
        return None, (
            "the serpapi_api_key session secret reached the env as a platform token, which "
            "the egress proxy only substitutes in request headers; SerpAPI takes the key as "
            "a query parameter, so set SERPAPI_API_KEY with env_overrides instead"
        )
    return None, "no serpapi_api_key secret and no SERPAPI_API_KEY environment variable"


# --------------------------------------------------------------------------- #
# HTTP layer (injectable for tests)                                           #
# --------------------------------------------------------------------------- #


class HttpTransportError(Exception):
    """No HTTP status: DNS failure, connection reset, timeout."""


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str
    truncated: bool = False

    def header(self, name: str) -> str:
        for k, v in self.headers.items():
            if k.lower() == name.lower():
                return v
        return ""


class HttpClient(Protocol):
    async def get(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout_s: float,
        max_bytes: int,
        allow_redirects: bool,
    ) -> HttpResponse: ...


class AiohttpClient:
    """Default client. ``trust_env`` so the pod's HTTPS_PROXY (the platform egress
    proxy) and SSL_CERT_FILE are honoured, like the SDK's own transport."""

    async def get(self, url, *, params=None, headers=None, timeout_s, max_bytes, allow_redirects):
        import aiohttp

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout_s), trust_env=True
            ) as session:
                async with session.get(
                    url, params=params, headers=headers, allow_redirects=allow_redirects
                ) as resp:
                    chunks: list[bytes] = []
                    size = 0
                    truncated = False
                    async for chunk in resp.content.iter_chunked(65536):
                        chunks.append(chunk)
                        size += len(chunk)
                        if size >= max_bytes:
                            truncated = True
                            break
                    return HttpResponse(
                        status=resp.status,
                        headers={k: v for k, v in resp.headers.items()},
                        body=b"".join(chunks)[:max_bytes],
                        url=str(resp.url),
                        truncated=truncated,
                    )
        except asyncio.TimeoutError as e:
            raise HttpTransportError(f"timed out after {timeout_s:.0f}s") from e
        except aiohttp.ClientError as e:
            # Only the class name: aiohttp messages can embed the request URL, and the
            # SerpAPI URL carries the key.
            raise HttpTransportError(type(e).__name__) from e


# --------------------------------------------------------------------------- #
# Usage accounting                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class WebUsage:
    """Per-session counters, reported in every tool output's metadata."""

    search_calls: int = 0
    serpapi_requests: int = 0
    search_cache_hits: int = 0
    fetch_calls: int = 0
    live_fetch_requests: int = 0
    wayback_requests: int = 0
    wayback_served: int = 0
    fetch_cache_hits: int = 0
    blocked: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Leak blocklist                                                              #
# --------------------------------------------------------------------------- #

DEFAULT_LEAK_URL_PATTERNS = (
    r"huggingface\.co/.*browse[\s_\-%20]*comp",
    r"(?:^|//|\.)(?:github\.com|githubusercontent\.com|github\.io)/.*browse[\s_\-]*comp",
    # The upstream (encrypted) test-set CSV.
    r"browse_comp_test_set",
    # Pages generated from other people's search queries. Their titles splice the answer onto
    # the question's own clues ("malakwa bc to new orleans museum of art distance walking"),
    # very likely built from agents running BrowseComp against Google. 7 of 150 sessions in the
    # first SerpAPI run (nemotron b0, 2026-10-07) saw the answer on one; 6 were graded right.
    r"/amphtml/news/articles/",                    # spam articles on hijacked domains
    r"(?:^|//|\.)instagram\.com/popular/",         # Instagram's auto topic pages
    r"(?:^|//|\.)tiktok\.com/discover/",           # TikTok's auto topic pages (same mechanism)
)
# A snippet this long that appears verbatim inside the question is a copy of it.
DEFAULT_SNIPPET_OVERLAP_CHARS = 80


def _norm_text(s: str) -> str:
    s = (s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.casefold().split())


def _load_patterns() -> list[str]:
    raw = os.environ.get("BROWSECOMP_LEAK_URL_PATTERNS")
    base = list(DEFAULT_LEAK_URL_PATTERNS) if raw is None else list(json.loads(raw))
    extra = os.environ.get("BROWSECOMP_LEAK_URL_PATTERNS_EXTRA")
    if extra:
        base += list(json.loads(extra))
    return base


class LeakFilter:
    """Blocks URLs and texts that would hand the agent a BrowseComp question or answer.

    Rules: URL regexes (default: huggingface.co or GitHub URLs mentioning BrowseComp,
    and the upstream test-set file; ``BROWSECOMP_LEAK_URL_PATTERNS`` replaces the list,
    ``..._EXTRA`` appends, both JSON lists of case-insensitive regexes), any title,
    snippet or page text containing the session's full question, and any snippet of at
    least ``BROWSECOMP_LEAK_SNIPPET_OVERLAP_CHARS`` characters found verbatim inside the
    question. Every block is logged.
    """

    def __init__(self, question: Optional[str], task_id: Optional[str] = None,
                 patterns: Optional[list[str]] = None) -> None:
        self.question = _norm_text(question or "")
        self.task_id = task_id
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns if patterns is not None else _load_patterns())]
        self.min_overlap = int(_float_env("BROWSECOMP_LEAK_SNIPPET_OVERLAP_CHARS", DEFAULT_SNIPPET_OVERLAP_CHARS))

    def url_rule(self, url: str) -> Optional[str]:
        candidates = {url or "", unquote(url or "")}
        for pat in self.patterns:
            if any(pat.search(c) for c in candidates):
                return f"url-pattern:{pat.pattern}"
        return None

    def text_rule(self, text: str) -> Optional[str]:
        if self.question and len(self.question) >= 20 and self.question in _norm_text(text):
            return "question-text"
        return None

    def snippet_rule(self, snippet: str) -> Optional[str]:
        s = _norm_text(re.sub(r"\.\.\.|…", " ", snippet or ""))
        if self.question and len(s) >= self.min_overlap and s in self.question:
            return "snippet-in-question"
        return None

    def hit_rule(self, hit: Mapping[str, Any]) -> Optional[str]:
        return (
            self.url_rule(str(hit.get("url") or ""))
            or self.text_rule(f"{hit.get('title', '')} {hit.get('snippet', '')}")
            or self.snippet_rule(str(hit.get("snippet") or ""))
        )

    def record(self, kind: str, url: str, rule: str) -> None:
        log.warning(json.dumps({
            "event": "browsecomp_leak_block", "kind": kind, "url": url, "rule": rule,
            "task_id": self.task_id,
        }))


# --------------------------------------------------------------------------- #
# SerpAPI search                                                              #
# --------------------------------------------------------------------------- #


def normalise_query(q: str) -> str:
    return " ".join((q or "").split()).casefold()


def _host_matches(host: str, domain: str) -> bool:
    host, domain = host.lower(), domain.lower().lstrip(".")
    return host == domain or host.endswith("." + domain)


def map_serpapi(body: Mapping[str, Any], max_results: int) -> list[dict[str, Any]]:
    """SerpAPI JSON -> backsearch-shaped hits ``{title, url, snippet, ...}``.

    The answer box (when it links a source) goes first, then organic results in rank
    order, then the knowledge graph's source; deduplicated by URL, capped at
    ``max_results``. A result's date, when Google shows one, is prefixed to the snippet
    the way Google displays it ("Mar 3, 2024 — ..."), so no new field reaches the agent.
    """
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(title: Any, url: Any, snippet: Any, date: Any = None, source: str = "organic", position: Any = None):
        if not url or not isinstance(url, str) or url in seen:
            return
        seen.add(url)
        snip = str(snippet or "").strip()
        if date:
            snip = f"{date} — {snip}" if snip else str(date)
        hits.append({
            "title": str(title or ""), "url": url, "snippet": snip, "date": date,
            "source": source, "position": position,
        })

    ab = body.get("answer_box")
    if isinstance(ab, Mapping) and ab.get("link"):
        snippet = ab.get("answer") or ab.get("snippet") or ab.get("result")
        if not snippet and isinstance(ab.get("snippet_highlighted_words"), list):
            snippet = ", ".join(map(str, ab["snippet_highlighted_words"]))
        add(ab.get("title"), ab.get("link"), snippet, ab.get("date"), "answer_box")
    for r in body.get("organic_results") or []:
        if isinstance(r, Mapping):
            add(r.get("title"), r.get("link"), r.get("snippet"), r.get("date"), "organic", r.get("position"))
    kg = body.get("knowledge_graph")
    if isinstance(kg, Mapping):
        src = kg.get("source") if isinstance(kg.get("source"), Mapping) else {}
        add(kg.get("title"), src.get("link") or kg.get("website"), kg.get("description"), None, "knowledge_graph")
    return hits[:max_results]


def _retry_after(resp: HttpResponse) -> Optional[float]:
    v = resp.header("Retry-After")
    try:
        return max(0.0, float(v)) if v else None
    except ValueError:
        return None


def _body_json(resp: HttpResponse) -> dict[str, Any]:
    try:
        data = json.loads(resp.body.decode("utf-8", errors="replace"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _strip_keys(obj: Any) -> Any:
    """Defensive: drop any ``api_key`` field before an entry is stored."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v) for k, v in obj.items() if k != "api_key"}
    if isinstance(obj, list):
        return [_strip_keys(v) for v in obj]
    return obj


_NO_RESULTS_RE = re.compile(r"hasn't returned any results|returned no results|no results found", re.I)
_OUT_OF_SEARCHES_RE = re.compile(r"run out of searches|out of searches|plan.*limit|searches per", re.I)
_BAD_KEY_RE = re.compile(r"invalid api key|api key.*(missing|invalid)", re.I)


@dataclass
class _Attempt:
    body: Optional[dict[str, Any]] = None  # successful SerpAPI JSON (may be an empty-result body)
    soft: Optional[WebToolResult] = None   # agent-recoverable error
    requests: int = 0


# --------------------------------------------------------------------------- #
# Live fetch                                                                  #
# --------------------------------------------------------------------------- #


def normalise_url(url: str) -> str:
    """Cache key form: lowercase scheme and host, no default port, no fragment."""
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower()
    port = p.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    return urlunsplit((p.scheme.lower(), netloc, p.path or "/", p.query, ""))


_CHALLENGE_MARKERS = (
    "just a moment...", "attention required! | cloudflare", "enable javascript and cookies to continue",
    "checking your browser before accessing", "/cdn-cgi/challenge-platform", "captcha-delivery.com",
    "px-captcha", "verify you are human", "please complete the security check",
)
_FALLBACK_STATUSES = frozenset({401, 403, 404, 408, 410, 429, 451})

_extract_sem: Optional[asyncio.Semaphore] = None


def _extract_semaphore() -> asyncio.Semaphore:
    # Extraction is CPU-bound and the pod has one CPU and a few GB shared by every
    # session, so bound how many pages are parsed at once.
    global _extract_sem
    if _extract_sem is None:
        _extract_sem = asyncio.Semaphore(int(_float_env("BROWSECOMP_FETCH_EXTRACT_CONCURRENCY", 2)))
    return _extract_sem


def _is_pdf(resp: HttpResponse) -> bool:
    return "application/pdf" in resp.header("Content-Type").lower() or resp.body[:5] == b"%PDF-"


def _extract_html(body: bytes) -> str:
    import trafilatura

    tree = trafilatura.load_html(body) if body else None
    if tree is None:
        return ""
    # Same call and settings as the backsearch fetch service (web-tools fetch_service).
    return trafilatura.extract(tree, include_comments=False, include_tables=True) or ""


def _extract_pdf(body: bytes, limit: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(body))
    parts: list[str] = []
    total = 0
    for i, page in enumerate(reader.pages):
        if i >= PDF_MAX_PAGES or total > limit:
            break
        t = page.extract_text() or ""
        parts.append(t)
        total += len(t)
    return "\n\n".join(p.strip() for p in parts if p.strip())


async def _extract(resp: HttpResponse, limit: int) -> tuple[str, Optional[str]]:
    """``(text, problem)``; problem is None on success."""
    ctype = resp.header("Content-Type").lower()
    async with _extract_semaphore():
        if _is_pdf(resp):
            if resp.truncated:
                return "", f"PDF larger than {FETCH_MAX_BYTES // (1024 * 1024)} MB"
            try:
                return await asyncio.to_thread(_extract_pdf, resp.body, limit), None
            except Exception as e:  # noqa: BLE001 - a broken PDF is one bad page
                return "", f"unreadable PDF ({type(e).__name__})"
        if ctype.startswith("text/plain") or "json" in ctype:
            return resp.body.decode("utf-8", errors="replace"), None
        head = resp.body[:2048].lstrip().lower()
        if ("html" in ctype or "xml" in ctype or not ctype or head.startswith(b"<")):
            try:
                return await asyncio.to_thread(_extract_html, resp.body), None
            except Exception as e:  # noqa: BLE001
                return "", f"unparseable HTML ({type(e).__name__})"
        return "", f"unsupported content type {ctype.split(';')[0] or 'unknown'}"


def _looks_like_challenge(resp: HttpResponse, text: str) -> bool:
    if resp.header("cf-mitigated").lower() == "challenge":
        return True
    if len(text.strip()) >= 1000:
        return False
    head = resp.body[:65536].decode("utf-8", errors="replace").lower()
    return any(m in head for m in _CHALLENGE_MARKERS)


# --------------------------------------------------------------------------- #
# The provider                                                                #
# --------------------------------------------------------------------------- #


class LiveWeb:
    """SerpAPI search + live fetch for one session. Holds the session's counters."""

    def __init__(
        self,
        *,
        secrets: Optional[Mapping[str, Any]],
        question: Optional[str],
        task_id: Optional[str],
        preapproved_hosts=(),
        http: Optional[HttpClient] = None,
        search_cache: Optional[TwoLevelCache] = None,
        fetch_cache: Optional[TwoLevelCache] = None,
        retry_budget_s: Optional[float] = None,
        sleep=None,
        clock=None,
    ) -> None:
        self._key, self._key_source = resolve_serpapi_key(secrets)
        self.leak = LeakFilter(question, task_id)
        self.task_id = task_id
        self.preapproved_hosts = frozenset(preapproved_hosts or ())
        self.http: HttpClient = http or AiohttpClient()
        self.search_cache = search_cache or get_cache("serpapi-search")
        self.fetch_cache = fetch_cache or get_cache("live-fetch")
        self.retry_budget_s = (
            retry_budget_s if retry_budget_s is not None
            else _float_env("BROWSECOMP_WEB_RETRY_BUDGET_S", 600.0)
        )
        self.max_results = int(_float_env("BROWSECOMP_SERPAPI_MAX_RESULTS", SERPAPI_NUM))
        self.fetch_max_bytes = int(_float_env("BROWSECOMP_FETCH_MAX_BYTES", FETCH_MAX_BYTES))
        self.fetch_timeout_s = _float_env("BROWSECOMP_FETCH_TIMEOUT_S", FETCH_TIMEOUT_S)
        self._sleep = sleep or (lambda s: asyncio.sleep(s))
        self._clock = clock or time.monotonic
        self.usage = WebUsage()

    # ---- helpers ----------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        if self._key and text:
            text = text.replace(self._key, "***")
        return _SECRET_TOKEN_RE.sub("***", text or "")

    def _emit(self, event: dict[str, Any]) -> None:
        event = {"task_id": self.task_id, **event}
        log.info(self._scrub(json.dumps(event, ensure_ascii=False)))

    def _output(self, result: WebToolResult, extra: dict[str, Any]) -> ToolOutput:
        out = to_tool_output(result, raise_on_fatal=True)
        meta = dict(out.metadata or {})
        meta.update(extra)
        meta["provider"] = "serpapi"
        meta["web_usage"] = self.usage.to_dict()
        # reward=None: searching and fetching are not graded (see backsearch._no_reward).
        return ToolOutput(blocks=out.blocks, metadata=meta, reward=None, finished=out.finished)

    # ---- search -------------------------------------------------------------------

    async def _serpapi_request(self, params: dict[str, str]) -> _Attempt:
        """One logical search: one request, plus retries only after a failed one."""
        if not self._key:
            raise SearchBackendUnavailable(
                "not-configured", f"SerpAPI search is not configured: {self._key_source}."
            )
        attempt_no = 0
        deadline = self._clock() + self.retry_budget_s
        out = _Attempt()
        while True:
            reason: str
            resp: Optional[HttpResponse] = None
            try:
                out.requests += 1
                self.usage.serpapi_requests += 1
                resp = await self.http.get(
                    SERPAPI_URL, params={**params, "api_key": self._key},
                    timeout_s=SERPAPI_TIMEOUT_S, max_bytes=SERPAPI_MAX_BYTES, allow_redirects=True,
                )
            except HttpTransportError as e:
                reason = f"transport error: {self._scrub(str(e))}"
            if resp is not None:
                body = _body_json(resp)
                err = self._scrub(str(body.get("error") or ""))
                if resp.status == 200 and not err:
                    out.body = body
                    return out
                if resp.status == 200 and _NO_RESULTS_RE.search(err):
                    out.body = {k: v for k, v in body.items() if k != "error"}
                    out.body.setdefault("organic_results", [])
                    return out
                if _OUT_OF_SEARCHES_RE.search(err):
                    raise SearchBackendUnavailable("quota-exceeded", f"SerpAPI quota exhausted: {err}")
                if resp.status in (401, 403) or _BAD_KEY_RE.search(err):
                    raise SearchBackendUnavailable(
                        "not-configured", f"SerpAPI rejected the API key (HTTP {resp.status}): {err}"
                    )
                if resp.status == 429 or resp.status >= 500:
                    reason = f"HTTP {resp.status}" + (f": {err}" if err else "")
                else:
                    # A rejected request (400) or an unexplained 200 error is about this
                    # query; the agent can rephrase.
                    out.soft = WebToolResult.error(
                        "search-failed", f"Search error (HTTP {resp.status}): {err or 'no detail'}"
                    )
                    return out
            step = min(60.0, 2.0 * (2 ** attempt_no))
            delay = step / 2 + random.uniform(0, step / 2)
            hint = _retry_after(resp) if resp is not None else None
            if hint is not None:
                delay = max(delay, hint)
            if self._clock() + delay > deadline:
                raise SearchBackendUnavailable(
                    "provider-unavailable",
                    f"SerpAPI unavailable: {reason}; still failing after {out.requests} requests. "
                    f"The rollout is an infrastructure failure, not an answer.",
                )
            await self._sleep(delay)
            attempt_no += 1

    async def search(self, query: str, allowed_domains=None, blocked_domains=None) -> ToolOutput:
        self.usage.search_calls += 1
        query, allowed, blocked, input_error = validate_search_input(query, allowed_domains, blocked_domains)
        if input_error is not None:
            return self._output(input_error, {"serpapi_requests": 0, "cache_hit": False})

        q = query
        if allowed:
            q += " (" + " OR ".join(f"site:{d}" for d in allowed) + ")"
        if blocked:
            q += " " + " ".join(f"-site:{d}" for d in blocked)
        params = {"engine": SERPAPI_ENGINE, "q": q, "num": str(SERPAPI_NUM)}
        material = {"engine": SERPAPI_ENGINE, "q": normalise_query(q), "num": SERPAPI_NUM}
        key = cache_key(material)

        body, level = await self.search_cache.get(key)
        requests = 0
        shared = False
        if body is None:
            async def make() -> _Attempt:
                attempt = await self._serpapi_request(params)
                if attempt.body is not None:
                    await self.search_cache.put(key, _strip_keys(attempt.body), material)
                return attempt

            attempt, shared = await self.search_cache.single_flight(key, make)
            if not shared:
                requests = attempt.requests
            if attempt.soft is not None:
                self._emit({"event": "serpapi_search", "query": query[:300], "cache": "miss",
                            "serpapi_requests": requests, "result": attempt.soft.error_code})
                return self._output(attempt.soft, {"serpapi_requests": requests, "cache_hit": False,
                                                   "cache_key": key})
            body = attempt.body
        cache_hit = body is not None and (level is not None or shared)
        if cache_hit:
            self.usage.search_cache_hits += 1

        hits = map_serpapi(body or {}, self.max_results)
        if allowed:
            hits = [h for h in hits if any(_host_matches(urlsplit(h["url"]).hostname or "", d) for d in allowed)]
        if blocked:
            hits = [h for h in hits if not any(_host_matches(urlsplit(h["url"]).hostname or "", d) for d in blocked)]
        kept = []
        for h in hits:
            rule = self.leak.hit_rule(h)
            if rule:
                self.usage.blocked += 1
                self.leak.record("search_hit", h["url"], rule)
                continue
            kept.append(h)

        result = WebToolResult.success(
            format_search_output(query, kept, include_snippets=True),
            {"query": query, "hits": kept, "mode": "serpapi-google"},
        )
        cache_label = level or ("inflight" if shared else "miss")
        self._emit({"event": "serpapi_search", "query": query[:300], "cache": cache_label,
                    "serpapi_requests": requests, "hits": len(kept), "blocked": len(hits) - len(kept)})
        return self._output(result, {"serpapi_requests": requests, "cache_hit": cache_hit,
                                     "cache_level": cache_label, "cache_key": key})

    # ---- fetch --------------------------------------------------------------------

    async def _get_following_same_host(self, url: str) -> tuple[Optional[HttpResponse], Optional[dict], Optional[str]]:
        """GET ``url``, following same-host redirects like the SDK does. Returns
        ``(response, cross_host_redirect, transport_error)``."""
        cur = url
        for _ in range(MAX_REDIRECTS + 1):
            self.usage.live_fetch_requests += 1
            try:
                resp = await self.http.get(
                    cur, headers=FETCH_HEADERS, timeout_s=self.fetch_timeout_s,
                    max_bytes=self.fetch_max_bytes, allow_redirects=False,
                )
            except HttpTransportError as e:
                return None, None, str(e)
            loc = resp.header("Location")
            if 300 <= resp.status < 400 and loc:
                target = urljoin(cur, loc)
                if _is_same_host_redirect(cur, target):
                    cur = target
                    continue
                return None, {"to": target, "status": resp.status}, None
            return resp, None, None
        return None, None, "too many redirects"

    async def _wayback(self, url: str, limit: int) -> tuple[Optional[dict], str]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        self.usage.wayback_requests += 1
        try:
            resp = await self.http.get(
                f"{WAYBACK_PREFIX}{stamp}id_/{url}", headers=FETCH_HEADERS,
                timeout_s=self.fetch_timeout_s, max_bytes=self.fetch_max_bytes, allow_redirects=True,
            )
        except HttpTransportError as e:
            return None, f"Wayback Machine: {e}"
        if resp.status != 200:
            return None, f"no Wayback Machine capture (HTTP {resp.status})"
        text, problem = await _extract(resp, limit)
        if problem or not text.strip():
            return None, f"Wayback Machine capture unusable ({problem or 'empty'})"
        m = re.search(r"/web/(\d{8})(\d{0,6})id_/", resp.url)
        archived = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}" if m else None
        return {"kind": "page", "text": text, "source": "wayback", "archived": archived}, ""

    async def _fetch_uncached(self, url: str, limit: int) -> dict[str, Any]:
        resp, redirect, transport = await self._get_following_same_host(url)
        if redirect is not None:
            return {"kind": "redirect", "redirect": redirect}
        reason: Optional[str] = None
        if transport is not None:
            reason = transport
        else:
            assert resp is not None
            if resp.status == 200:
                text, problem = await _extract(resp, limit)
                if problem and problem.startswith("unsupported content type"):
                    return {"kind": "error", "code": "fetch-error", "message": f"Cannot read {url}: {problem}."}
                if problem:
                    reason = problem
                elif _looks_like_challenge(resp, text):
                    reason = "bot challenge page"
                elif not text.strip():
                    reason = "empty page"
                else:
                    return {"kind": "page", "text": text, "source": "live"}
            elif resp.status in _FALLBACK_STATUSES or resp.status >= 500:
                reason = f"HTTP {resp.status}"
            else:
                return {"kind": "error", "code": "fetch-error",
                        "message": f"Fetching {url} failed with HTTP {resp.status}."}
        page, wb_problem = await self._wayback(url, limit)
        if page is not None:
            page["live_failure"] = reason
            return page
        return {
            "kind": "error", "code": "fetch-error",
            "message": (f"Could not retrieve {url}: the live page failed ({reason}) and {wb_problem}. "
                        f"Do not retry this URL - choose a different result from web_search instead."),
        }

    def _blocked(self, url: str, rule: str, extra: dict[str, Any]) -> ToolOutput:
        self.usage.blocked += 1
        self.leak.record("fetch", url, rule)
        return self._output(
            WebToolResult.error("domain-blocked", f"Access to {url} is blocked in this environment. "
                                                  f"Choose a different source."),
            extra,
        )

    async def fetch(self, url: str, prompt: str, max_chars: Optional[int] = None) -> ToolOutput:
        self.usage.fetch_calls += 1
        limit = max_chars or MAX_FETCH_TEXT_CHARS
        base = {"live_fetch": True, "cache_hit": False}
        if not url:
            return self._output(WebToolResult.error("missing-url", "`url` is required"), base)
        if not prompt:
            return self._output(WebToolResult.error("missing-prompt", "`prompt` is required"), base)
        clean, problem = validate_fetch_url(url)
        if problem is not None:
            return self._output(WebToolResult.error(*problem), base)
        url = clean
        rule = self.leak.url_rule(url)
        if rule:
            return self._blocked(url, rule, base)

        key = cache_key({"url": normalise_url(url)})
        before = (self.usage.live_fetch_requests, self.usage.wayback_requests)
        outcome, level = await self.fetch_cache.get(key)
        shared = False
        if outcome is None:
            async def make() -> dict[str, Any]:
                o = await self._fetch_uncached(url, limit)
                if o["kind"] == "page":
                    await self.fetch_cache.put(key, o, {"url": normalise_url(url)})
                return o

            outcome, shared = await self.fetch_cache.single_flight(key, make)
        cache_hit = level is not None or shared
        if cache_hit:
            self.usage.fetch_cache_hits += 1
        meta = {
            "live_fetch": True, "cache_hit": cache_hit, "cache_level": level or ("inflight" if shared else "miss"),
            "live_requests": self.usage.live_fetch_requests - before[0],
            "wayback_requests": self.usage.wayback_requests - before[1],
            "fetch_source": outcome.get("source"),
        }

        kind = outcome["kind"]
        if kind == "redirect":
            r = outcome["redirect"]
            if self.leak.url_rule(r["to"]):
                return self._blocked(r["to"], self.leak.url_rule(r["to"]) or "", meta)
            result = WebToolResult.success(
                _redirect_template(original_url=url, redirect_url=r["to"], status=int(r["status"]), prompt=prompt),
                {"redirect": r, "url": url},
            )
        elif kind == "error":
            result = WebToolResult.error(outcome["code"], outcome["message"])
        else:
            text = outcome["text"]
            rule = self.leak.text_rule(text)
            if rule:
                return self._blocked(url, rule, meta)
            if outcome.get("source") == "wayback":
                self.usage.wayback_served += 1
                when = outcome.get("archived") or "an earlier date"
                text = (f"[Archived copy: the live page could not be retrieved "
                        f"({outcome.get('live_failure') or 'unavailable'}), so this is the Wayback "
                        f"Machine capture from {when}.]\n\n{text}")
            if len(text) > limit:
                text = text[:limit] + "\n... (truncated)"
            hostname = (urlsplit(url).hostname or "").lower()
            result = WebToolResult.success(
                render_fetch_text(url, text, hostname, self.preapproved_hosts, limit),
                {"url": url, "prompt": prompt},
            )
        self._emit({"event": "live_fetch", "url": url, "cache": meta["cache_level"],
                    "source": outcome.get("source"), "kind": kind,
                    "live_requests": meta["live_requests"], "wayback_requests": meta["wayback_requests"]})
        return self._output(result, meta)
