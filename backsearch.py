"""Backdated web_search / web_fetch, ported from the EnvCommons/obscurefacts env."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from openreward.environments import ToolOutput, tool
from openreward.toolsets import BackSearchToolset
from openreward.toolsets._web_common import WebFetchParams, WebSearchParams, to_tool_output
from openreward.tools.web import FETCH_DESCRIPTION, SEARCH_DESCRIPTION, WebToolResult, run_fetch, run_search
from openreward.web_service import WebServiceConfig


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


class BrowseCompBackSearch(BackSearchToolset):
    """BackSearchToolset with five adjustments, all backdating-preserving.

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

    @tool
    async def web_search(self, params: WebSearchParams) -> ToolOutput:
        result = await run_search(
            query=params.query,
            as_of=self._current_as_of(),
            allowed_domains=params.allowed_domains,
            blocked_domains=params.blocked_domains,
            config=self.config,
            include_snippets=True,
        )
        return to_tool_output(result, raise_on_fatal=True)

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        as_of = self._current_as_of()
        result = await run_fetch(
            url=params.url, prompt=params.prompt, as_of=as_of, config=self.config
        )
        if _is_not_archived(result):
            for candidate in _url_variants(params.url):
                retry = await run_fetch(
                    url=candidate, prompt=params.prompt, as_of=as_of, config=self.config
                )
                if retry.ok:
                    result = retry
                    break
            else:
                result = _not_archived_result(params.url, as_of)
        return _drop_content_mirror(to_tool_output(result, raise_on_fatal=True))


# The environment framework reads ``fn.__doc__`` for each tool's description.
BrowseCompBackSearch.web_search.__doc__ = SEARCH_DESCRIPTION
BrowseCompBackSearch.web_fetch.__doc__ = FETCH_DESCRIPTION

