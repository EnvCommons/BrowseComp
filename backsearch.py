"""Backdated web_search / web_fetch, ported from the EnvCommons/obscurefacts env."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from openreward.environments import ToolOutput, tool
from openreward.toolsets import BackSearchToolset
from openreward.toolsets._web_common import (
    SearchBackendUnavailable,
    WebFetchParams,
    WebSearchParams,
    to_tool_output,
)
from openreward.tools.web import FETCH_DESCRIPTION, SEARCH_DESCRIPTION, WebToolResult, run_fetch, run_search
from openreward.web_service import (
    AiohttpTransport,
    RawResponse,
    WebServiceClient,
    WebServiceConfig,
    WebTransport,
    WebTransportError,
)


def today_utc_iso() -> str:
    """Today's date in UTC as ISO ``YYYY-MM-DD`` — the backsearch cutoff.

    UTC rather than the server's local date so every replica of the env agrees
    on the cutoff regardless of the timezone it happens to run in.
    """
    return datetime.now(timezone.utc).date().isoformat()


def _url_variants(url: str) -> list[str]:
    """Cosmetic respellings of ``url`` worth retrying after a 404.

    The archive matches URLs byte-for-byte, so a page it holds under one
    spelling 404s under another. Measured on 84 real-but-mangled URLs: 89% of
    cosmetic variations 404, and this ladder recovered 100% of them
    (percent-encoding 9/9, trailing slash 41/41, ``www.`` 25/25).

    Percent-decoding is applied first and the toggles build on the decoded
    form, so a URL that is both re-encoded *and* slash-mangled is still
    repaired inside the three-attempt budget. Capped at three so a genuine miss
    costs at most three extra calls.
    """
    parts = urlsplit(url)
    decoded = unquote(parts.path)
    variants: list[str] = []
    base = url
    if decoded != parts.path:
        base = urlunsplit((parts.scheme, parts.netloc, decoded, parts.query, parts.fragment))
        variants.append(base)
    p = urlsplit(base)
    if p.path and p.path != "/":
        flipped = p.path[:-1] if p.path.endswith("/") else p.path + "/"
        variants.append(urlunsplit((p.scheme, p.netloc, flipped, p.query, p.fragment)))
    host = p.netloc[4:] if p.netloc.startswith("www.") else "www." + p.netloc
    variants.append(urlunsplit((p.scheme, host, p.path, p.query, p.fragment)))
    seen = {url}
    return [v for v in variants if not (v in seen or seen.add(v))]


def _is_not_archived(result: WebToolResult) -> bool:
    """True for "the archive has no capture of this URL", not for a real outage."""
    return (
        not result.ok
        and result.error_code == "web-service-error"
        and "HTTP 404" in (result.output or "")
    )


def _not_archived_result(url: str, as_of: Optional[str]) -> WebToolResult:
    """Replace the backend's raw 404 envelope with something the agent can act on.

    The stock message reads as an infrastructure failure and leaks internal
    corpus names, so agents re-fetch the same dead URL or guess neighbouring
    ones until the turn cap. Half the failed fetches observed in real rollouts
    were URLs the model invented rather than ones search returned, which is
    exactly the behaviour this wording is meant to stop.
    """
    return WebToolResult.error(
        "page-not-archived",
        f"No archived capture of {url} on or before {as_of or 'today'}. The archive "
        f"only serves pages it captured, so this URL may never have been captured, or "
        f"may not exist. Do not retry this URL or guess variations of it - choose a "
        f"different result from web_search instead.",
    )


def _drop_content_mirror(out: ToolOutput) -> ToolOutput:
    """Remove ``metadata["content"]``, a byte-identical copy of the text already
    in ``blocks``.

    The SDK's ``build_fetch_output`` puts the whole page into ``data["content"]``
    and ``to_tool_output`` copies that into the ToolOutput metadata, so every
    fetch ships the page twice. Nothing reads the copy. The training harness
    caps environment tool output at 32,768 bytes and replaces the *entire*
    result with "[env tool output exceeded cap before content rendering]" when
    it is exceeded, so the duplicate alone can cost the agent the page it just
    fetched. Measured on a real rollout: a 65,550-byte fetch was exactly half
    mirror, and dropping it brought the payload under the cap.
    """
    if not out.metadata or "content" not in out.metadata:
        return out
    slim = {k: v for k, v in out.metadata.items() if k != "content"}
    return ToolOutput(
        blocks=out.blocks,
        metadata=slim or None,
        reward=out.reward,
        finished=out.finished,
    )


def _no_reward(out: ToolOutput) -> ToolOutput:
    """Searching and fetching are not graded: report no reward rather than 0.0.

    A 0.0 on every web call makes the harness's cumulative return a sum of zeros, and scorers
    that prefer the cumulative over the terminal grade then report 0 for a correct answer.
    """
    return ToolOutput(blocks=out.blocks, metadata=out.metadata, reward=None, finished=out.finished)


# Statuses that mean "the backend is busy or broken right now", not "this request is wrong".
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# For fetch, only the service saying it is over capacity. A 5xx or timeout there is
# usually about one page (slow, broken), and the agent can pick another result, so it
# stays a soft error rather than stalling the call and then discarding the rollout.
_CAPACITY_STATUSES = frozenset({429, 503})


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _retry_after_s(headers: Mapping[str, str]) -> Optional[float]:
    """``Retry-After`` as seconds, or None. Only the delta-seconds form is parsed; the
    web service never sends an HTTP-date."""
    for k, v in headers.items():
        if k.lower() == "retry-after":
            try:
                return max(0.0, float(v))
            except (TypeError, ValueError):
                return None
    return None


def _is_degraded(resp: RawResponse) -> bool:
    """A 200 search whose body says some requested corpora contributed nothing."""
    if resp.status != 200:
        return False
    try:
        body = json.loads(resp.text)
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("degraded") is True


class PatientTransport:
    """Wraps the SDK transport so a busy or partially failing backend is waited out, not
    scored.

    The SDK retries 429/5xx three times over about three seconds and ignores
    ``Retry-After``, then hands the agent a soft "web-service-error". The agent carries
    on without results and the trial is graded as though it had searched. Under load
    shedding that turns an overloaded search service into wrong answers. The search
    service also returns 200 with ``degraded: true`` when some corpora timed out; the SDK
    drops that field, so a search over one corpus of six reads as a full one.

    Searches are retried on 429, 5xx, transport errors and ``degraded``; every other
    request (fetch, domain preflight) only on 429/503, since its other failures are
    usually about one page. Retries use capped,
    jittered exponential backoff, never waiting less than ``Retry-After``, until
    ``budget_s`` runs out. It then records ``gave_up`` and returns the last response
    unchanged, and every later request on this transport returns immediately. The
    toolset checks ``gave_up`` after the call and raises ``SearchBackendUnavailable``,
    so the rollout ends as an environment fault and is discarded rather than scored.

    One instance per tool call: ``gave_up`` must describe that call only, and the
    short-circuit stops the SDK's own retry loop from spending the budget again.
    """

    def __init__(
        self,
        inner: WebTransport,
        *,
        search_url: str,
        budget_s: float,
        base_s: float = 2.0,
        cap_s: float = 60.0,
        sleep=None,
        clock=None,
    ) -> None:
        self._inner = inner
        self._search_url = search_url
        self._budget_s = budget_s
        self._base_s = base_s
        self._cap_s = cap_s
        # Resolved per call, not bound at import, so tests can patch asyncio.sleep.
        self._sleep = sleep or (lambda s: asyncio.sleep(s))
        self._clock = clock or time.monotonic
        self._last: Optional[RawResponse] = None
        self.gave_up: Optional[str] = None
        self.retries = 0

    async def request(self, method: str, url: str, **kwargs: Any) -> RawResponse:
        if self.gave_up is not None:
            if self._last is None:
                raise WebTransportError(self.gave_up)
            return self._last
        deadline = self._clock() + self._budget_s
        attempt = 0
        while True:
            error: Optional[WebTransportError] = None
            resp: Optional[RawResponse] = None
            is_search = url == self._search_url
            try:
                resp = await self._inner.request(method, url, **kwargs)
            except WebTransportError as e:
                if not is_search:
                    raise
                error = e
            if resp is not None:
                self._last = resp
                if resp.status in (_RETRY_STATUSES if is_search else _CAPACITY_STATUSES):
                    reason = f"HTTP {resp.status}"
                elif is_search and _is_degraded(resp):
                    reason = "degraded search (corpora missing)"
                else:
                    return resp
            else:
                reason = f"transport error: {error}"

            # Equal jitter: half fixed, half random, so a crowd of rollouts spreads out
            # without any of them retrying almost immediately.
            step = min(self._cap_s, self._base_s * (2 ** attempt))
            delay = step / 2 + random.uniform(0, step / 2)
            hint = _retry_after_s(resp.headers) if resp is not None else None
            if hint is not None:
                delay = max(delay, hint)
            if self._clock() + delay > deadline:
                self.gave_up = f"{reason}; still failing after {self.retries} retries"
                if resp is None:
                    raise error  # type: ignore[misc]
                return resp
            await self._sleep(delay)
            attempt += 1
            self.retries += 1

    async def aclose(self) -> None:
        await self._inner.aclose()


class BrowseCompBackSearch(BackSearchToolset):
    """BackSearchToolset with six adjustments, all backdating-preserving.

    First, the session's secrets (``api_key`` / ``openreward_api_key``) are
    consulted when building the web-service config, falling back to the process
    environment's ``OPENREWARD_API_KEY`` — the stock toolset reads the process
    env only. Second, ``web_search`` passes ``include_snippets=True`` so results
    carry text snippets instead of bare titles and URLs, letting the agent
    triage hits without a fetch per candidate. Third, fatal backend errors (a
    missing key, an exhausted quota) raise ``SearchBackendUnavailable`` instead
    of becoming tool output: handed back as text, the agent would re-issue a
    dead call until the turn cap and the rollout would score 0.0 as though the
    model had answered wrongly, rather than being discarded as an
    infrastructure failure. Fourth, a fetch that 404s is retried against
    cosmetic respellings of the URL (see ``_url_variants``), and a genuine miss
    returns an actionable ``page-not-archived`` message. Fifth, the redundant
    ``metadata["content"]`` mirror is dropped (see ``_drop_content_mirror``).
    Sixth, every call goes through a ``PatientTransport``: an overloaded or degraded
    backend (for fetch, only an overloaded one) is waited out for up to ``BROWSECOMP_WEB_RETRY_BUDGET_S``
    (default 600 s) per tool call, and if it has not recovered by then the call
    raises ``SearchBackendUnavailable`` so the rollout is discarded, not graded
    on missing search results.

    The cutoff still resolves through the parent's ``_current_as_of``
    (``env.web_as_of``, the UTC date the session was created) on every call.
    No ``corpus`` is pinned, so the backend fans out over its default corpora
    (news, SEC filings, Wikipedia, general web, live captures, arXiv) — naming
    corpora *replaces* that set rather than extending it. Unlike the SDK's
    swappable web toolset, this one cannot be switched to a live-web provider
    by an environment variable.
    """

    def __init__(self, env: Optional[Any] = None, **kwargs: Any) -> None:
        if kwargs.get("config") is None:
            secrets = getattr(env, "search_secrets", None)
            kwargs["config"] = WebServiceConfig.from_env(secrets)
        super().__init__(env, **kwargs)
        self.retry_budget_s = _float_env("BROWSECOMP_WEB_RETRY_BUDGET_S", 600.0)
        self._inner_transport = AiohttpTransport  # overridable in tests

    def _patient_client(self) -> tuple[Optional[WebServiceClient], Optional[PatientTransport]]:
        """A client for ONE tool call, and its transport. (None, None) when no key is
        configured, so the SDK's own "not-configured" fatal path runs unchanged."""
        if self.config is None:
            return None, None
        transport = PatientTransport(
            self._inner_transport(),
            search_url=self.config.search_url,
            budget_s=self.retry_budget_s,
        )
        return WebServiceClient(self.config, transport=transport), transport

    @staticmethod
    def _raise_if_gave_up(transport: Optional[PatientTransport]) -> None:
        if transport is not None and transport.gave_up is not None:
            raise SearchBackendUnavailable(
                "provider-unavailable",
                f"Web service unavailable: {transport.gave_up}. The rollout is an "
                f"infrastructure failure, not an answer.",
            )

    @tool
    async def web_search(self, params: WebSearchParams) -> ToolOutput:
        client, transport = self._patient_client()
        try:
            result = await run_search(
                query=params.query,
                as_of=self._current_as_of(),
                allowed_domains=params.allowed_domains,
                blocked_domains=params.blocked_domains,
                config=self.config,
                client=client,
                include_snippets=True,
            )
        finally:
            if client is not None:
                await client.aclose()
        self._raise_if_gave_up(transport)
        return _no_reward(to_tool_output(result, raise_on_fatal=True))

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        as_of = self._current_as_of()
        client, transport = self._patient_client()
        try:
            result = await run_fetch(
                url=params.url, prompt=params.prompt, as_of=as_of, config=self.config,
                client=client,
            )
            if _is_not_archived(result):
                for candidate in _url_variants(params.url):
                    retry = await run_fetch(
                        url=candidate, prompt=params.prompt, as_of=as_of,
                        config=self.config, client=client,
                    )
                    if retry.ok:
                        result = retry
                        break
                else:
                    result = _not_archived_result(params.url, as_of)
        finally:
            if client is not None:
                await client.aclose()
        self._raise_if_gave_up(transport)
        return _no_reward(_drop_content_mirror(to_tool_output(result, raise_on_fatal=True)))


# The environment framework reads ``fn.__doc__`` for each tool's description.
BrowseCompBackSearch.web_search.__doc__ = SEARCH_DESCRIPTION
BrowseCompBackSearch.web_fetch.__doc__ = FETCH_DESCRIPTION

