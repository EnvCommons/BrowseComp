"""Write-only archive of what the live (SerpAPI) web tools paid for or downloaded.

The cache (web_cache.py) keeps a parsed copy for reuse and expires it. The archive keeps
everything, forever, for later analysis and indexing: every SerpAPI response that cost a
search (the full JSON, including ``search_metadata.id``, so SerpAPI's own Searches
Archive API can be queried for 31 days), and every live or Wayback fetch, with the raw
response bytes stored once by content hash. Nothing here is ever read back by the env,
and nothing here changes what the agent sees.

Configuration (process environment, read once per process):

``BROWSECOMP_WEB_ARCHIVE_URL``
    ``gs://bucket/prefix``, ``file:///abs/dir`` or an absolute path. Unset means no archive.
``BROWSECOMP_WEB_ARCHIVE_GCS_SA_JSON``
    Optional service-account key for ``gs://`` (the JSON itself, or base64 of it).
    Without it, Application Default Credentials are used.
``BROWSECOMP_WEB_ARCHIVE_TAG``
    Optional free-form label stored in every record (e.g. the eval run name).

Layout under the archive root::

    serpapi/<YYYY-MM-DD>/<task_id>/<epoch_ms>-<cache_key[:16]>.json
    fetch/<YYYY-MM-DD>/<task_id>/<epoch_ms>-<sha256(url)[:16]>.json
    fetch-raw/<sha256[:2]>/<sha256>            (raw response body, content-addressed)

Writes run as background tasks so a tool call never waits on them; a failed write is
logged and dropped. The API key is stripped from every SerpAPI body before it is stored.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

from web_cache import PersistentStore, store_from_url

log = logging.getLogger("browsecomp.web_archive")

ARCHIVE_FORMAT_VERSION = 1


def _safe(part: Optional[str]) -> str:
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(part or "unknown"))
    return s[:80] or "unknown"


class WebArchive:
    def __init__(self, store: PersistentStore, *, tag: Optional[str] = None, clock=time.time) -> None:
        self.store = store
        self.tag = tag
        self._clock = clock
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:  # no running loop: nothing to archive into
            coro.close()
            return
        self._tasks.add(task)  # keep a strong reference until the write finishes
        task.add_done_callback(self._tasks.discard)

    async def _put(self, path: str, data: bytes, content_type: str) -> None:
        try:
            await self.store.put(path, data, content_type=content_type)
        except Exception as e:  # noqa: BLE001 - archiving must never fail a tool call
            log.warning("web_archive.write_failed path=%s err=%s", path.split("/", 1)[0], type(e).__name__)

    def _record_path(self, kind: str, task_id: Optional[str], digest: str) -> tuple[str, float]:
        now = self._clock()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        return f"{kind}/{day}/{_safe(task_id)}/{int(now * 1000)}-{digest[:16]}.json", now

    def _doc(self, kind: str, now: float, task_id: Optional[str], fields: dict[str, Any]) -> bytes:
        doc = {"v": ARCHIVE_FORMAT_VERSION, "kind": kind, "archived_at": now, "tag": self.tag,
               "task_id": task_id, **fields}
        # ensure_ascii keeps U+2028/29 escaped, so line-based readers of exports never split a record
        return json.dumps(doc, ensure_ascii=True, default=str).encode("utf-8")

    def serpapi(self, *, task_id: Optional[str], query: str, params: dict[str, Any], cache_key: str,
                body: dict[str, Any], requests: int) -> None:
        """One paid SerpAPI search. ``body`` must already have its api_key stripped."""
        path, now = self._record_path("serpapi", task_id, cache_key)
        meta = body.get("search_metadata") if isinstance(body.get("search_metadata"), dict) else {}
        data = self._doc("serpapi", now, task_id, {
            "query": query, "params": {k: v for k, v in params.items() if k != "api_key"},
            "cache_key": cache_key, "search_id": meta.get("id"), "serpapi_requests": requests,
            "body": body,
        })
        self._spawn(self._put(path, data, "application/json"))

    def fetch(self, *, task_id: Optional[str], url: str, outcome: dict[str, Any],
              responses: list[Any]) -> None:
        """One uncached fetch: the outcome the env built plus every HTTP response behind it
        (live page, redirect, Wayback capture), each body stored once by content hash."""
        path, now = self._record_path("fetch", task_id, hashlib.sha256(url.encode()).hexdigest())
        recs = []
        for r in responses:
            body = r.body or b""
            sha = hashlib.sha256(body).hexdigest()
            ctype = r.header("Content-Type") or "application/octet-stream"
            recs.append({"url": r.url, "status": r.status, "content_type": ctype,
                         "bytes": len(body), "truncated": r.truncated,
                         "body_sha256": sha if body else None})
            if body:
                self._spawn(self._put(f"fetch-raw/{sha[:2]}/{sha}", body, ctype))
        self._spawn(self._put(path, self._doc("fetch", now, task_id,
                                              {"url": url, "outcome": outcome, "responses": recs}),
                              "application/json"))

    async def drain(self) -> None:
        """Wait for pending writes (tests and clean shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


_ARCHIVE: dict[str, Optional[WebArchive]] = {}


def get_archive() -> Optional[WebArchive]:
    """Process-wide archive built from the environment on first use; None when unset."""
    if "archive" not in _ARCHIVE:
        store = store_from_url(os.environ.get("BROWSECOMP_WEB_ARCHIVE_URL"),
                               sa_json=os.environ.get("BROWSECOMP_WEB_ARCHIVE_GCS_SA_JSON"),
                               env_name="BROWSECOMP_WEB_ARCHIVE_URL")
        _ARCHIVE["archive"] = (WebArchive(store, tag=os.environ.get("BROWSECOMP_WEB_ARCHIVE_TAG") or None)
                               if store is not None else None)
        log.info("web_archive.configured store=%s",
                 store.describe() if store is not None else "none")
    return _ARCHIVE["archive"]


def reset_archive() -> None:
    """Tests only."""
    _ARCHIVE.clear()
