"""Two-level cache for the live (SerpAPI) web tools.

Level 1 is an in-process LRU, shared by every session in one env process. Level 2 is
an optional persistent store shared across sessions, pods and separate eval runs, so
repeated identical queries are paid for once. Only the live provider uses this
module; the default backsearch path never touches it.

Configuration (process environment, read once per process):

``BROWSECOMP_WEB_CACHE_URL``
    Persistent location. Unset means in-process only. ``gs://bucket/prefix`` uses
    Google Cloud Storage through its JSON API. ``file:///abs/dir`` or a bare absolute
    path uses a local or mounted directory.
``BROWSECOMP_WEB_CACHE_TTL_S``
    Entry lifetime in seconds for both levels. Default 30 days.
``BROWSECOMP_WEB_CACHE_GCS_SA_JSON``
    Optional service-account key for ``gs://`` (the JSON itself, or base64 of it).
    Without it, Application Default Credentials are used (workload identity or
    ``GOOGLE_APPLICATION_CREDENTIALS``).
``BROWSECOMP_WEB_CACHE_MEM_MB``
    In-process LRU budget per namespace. Default 32.

Keys never contain credentials: callers hash the normalised request (query plus the
SerpAPI parameters that change the result, without ``api_key``) or the normalised URL.
Entries are JSON documents ``{"v", "stored_at", "key_material", "payload"}``. A
persistent-store failure is logged and treated as a miss; it never fails a tool call.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import quote, urlsplit

log = logging.getLogger("browsecomp.web_cache")

CACHE_FORMAT_VERSION = 1
DEFAULT_TTL_S = 30 * 24 * 3600
DEFAULT_MEM_MB = 32
GCS_TIMEOUT_S = 10.0


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def cache_key(material: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of ``material`` (sorted keys, no whitespace)."""
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class PersistentStore(Protocol):
    async def get(self, path: str) -> Optional[bytes]: ...
    async def put(self, path: str, data: bytes, content_type: str = "application/json") -> None: ...


class LocalDirStore:
    """Persistent store on a local or mounted directory (also used by tests)."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def describe(self) -> str:
        return f"file://{self.root}"

    async def get(self, path: str) -> Optional[bytes]:
        p = self.root / path
        try:
            return await asyncio.to_thread(p.read_bytes)
        except FileNotFoundError:
            return None

    async def put(self, path: str, data: bytes, content_type: str = "application/json") -> None:
        p = self.root / path

        def _write() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, p)  # atomic, so a concurrent reader never sees half a file

        await asyncio.to_thread(_write)


class GCSStore:
    """Persistent store on ``gs://bucket/prefix`` via the GCS JSON API.

    Uses google-auth (already installed with the openreward SDK) for an OAuth token,
    refreshed in a worker thread, and aiohttp for the object reads and writes.
    """

    _SCOPES = ["https://www.googleapis.com/auth/devstorage.read_write"]

    def __init__(self, bucket: str, prefix: str, sa_json: Optional[str] = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._sa_json = sa_json
        self._creds: Any = None
        self._lock = asyncio.Lock()
        self._session: Any = None

    def describe(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}"

    def _load_creds(self) -> Any:
        import google.auth
        from google.oauth2 import service_account

        if self._sa_json:
            raw = self._sa_json.strip()
            if not raw.startswith("{"):
                raw = base64.b64decode(raw).decode("utf-8")
            return service_account.Credentials.from_service_account_info(
                json.loads(raw), scopes=self._SCOPES
            )
        creds, _ = google.auth.default(scopes=self._SCOPES)
        return creds

    async def _token(self) -> str:
        async with self._lock:
            if self._creds is None:
                self._creds = await asyncio.to_thread(self._load_creds)
            if not self._creds.valid:
                import google.auth.transport.requests

                await asyncio.to_thread(self._creds.refresh, google.auth.transport.requests.Request())
            return self._creds.token

    async def _http(self) -> Any:
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=GCS_TIMEOUT_S), trust_env=True
            )
        return self._session

    def _object(self, path: str) -> str:
        return f"{self.prefix}/{path}" if self.prefix else path

    async def get(self, path: str) -> Optional[bytes]:
        token = await self._token()
        session = await self._http()
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/"
            f"{quote(self._object(path), safe='')}?alt=media"
        )
        async with session.get(url, headers={"Authorization": f"Bearer {token}"}) as resp:
            if resp.status == 404:
                return None
            if resp.status != 200:
                raise RuntimeError(f"GCS read HTTP {resp.status}")
            return await resp.read()

    async def put(self, path: str, data: bytes, content_type: str = "application/json") -> None:
        token = await self._token()
        session = await self._http()
        url = (
            f"https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o"
            f"?uploadType=media&name={quote(self._object(path), safe='')}"
        )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}
        async with session.post(url, data=data, headers=headers) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"GCS write HTTP {resp.status}")


_FROM_ENV = object()


def store_from_url(url: Optional[str], *, sa_json: Any = _FROM_ENV,
                   env_name: str = "BROWSECOMP_WEB_CACHE_URL") -> Optional[PersistentStore]:
    """Build the persistent store named by ``url``; None for in-process only.

    ``sa_json`` defaults to BROWSECOMP_WEB_CACHE_GCS_SA_JSON; the archive passes its own."""
    url = (url or "").strip()
    if not url:
        return None
    if url.startswith("gs://"):
        parts = urlsplit(url)
        if sa_json is _FROM_ENV:
            sa_json = os.environ.get("BROWSECOMP_WEB_CACHE_GCS_SA_JSON")
        return GCSStore(parts.netloc, parts.path, sa_json)
    if url.startswith("file://"):
        return LocalDirStore(urlsplit(url).path)
    if url.startswith("/"):
        return LocalDirStore(url)
    raise ValueError(
        f"{env_name} must be gs://bucket/prefix, file:///dir or an absolute "
        f"path, got {url!r}"
    )


class _MemLRU:
    """Byte-budgeted LRU of (stored_at, payload, nbytes)."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._d: "OrderedDict[str, tuple[float, Any, int]]" = OrderedDict()
        self._total = 0

    def get(self, key: str) -> Optional[tuple[float, Any]]:
        hit = self._d.get(key)
        if hit is None:
            return None
        self._d.move_to_end(key)
        return hit[0], hit[1]

    def put(self, key: str, stored_at: float, payload: Any, nbytes: int) -> None:
        old = self._d.pop(key, None)
        if old is not None:
            self._total -= old[2]
        if nbytes > self.max_bytes:
            return
        while self._total + nbytes > self.max_bytes and self._d:
            _, ev = self._d.popitem(last=False)
            self._total -= ev[2]
        self._d[key] = (stored_at, payload, nbytes)
        self._total += nbytes

    def drop(self, key: str) -> None:
        old = self._d.pop(key, None)
        if old is not None:
            self._total -= old[2]

    def clear(self) -> None:
        self._d.clear()
        self._total = 0


class TwoLevelCache:
    """In-process LRU in front of an optional persistent store, for one namespace.

    ``get`` returns ``(payload, level)`` with level ``"memory"`` or ``"persistent"``,
    or ``(None, None)`` on a miss or an expired entry. ``single_flight`` collapses
    concurrent identical misses in this process into one upstream call.
    """

    def __init__(
        self,
        namespace: str,
        *,
        store: Optional[PersistentStore],
        ttl_s: float,
        mem_bytes: int,
        clock=time.time,
    ) -> None:
        self.namespace = namespace
        self.store = store
        self.ttl_s = ttl_s
        self.mem = _MemLRU(mem_bytes)
        self._clock = clock
        self._inflight: dict[str, asyncio.Future] = {}

    def _path(self, key: str) -> str:
        return f"{self.namespace}/v{CACHE_FORMAT_VERSION}/{key[:2]}/{key}.json"

    def _fresh(self, stored_at: float) -> bool:
        return self._clock() - stored_at <= self.ttl_s

    async def get(self, key: str) -> tuple[Optional[Any], Optional[str]]:
        hit = self.mem.get(key)
        if hit is not None:
            if self._fresh(hit[0]):
                return hit[1], "memory"
            self.mem.drop(key)
        if self.store is None:
            return None, None
        try:
            raw = await self.store.get(self._path(key))
        except Exception as e:  # noqa: BLE001 - a cache outage is a miss, never a failure
            log.warning("web_cache.read_failed ns=%s err=%s", self.namespace, type(e).__name__)
            return None, None
        if raw is None:
            return None, None
        try:
            entry = json.loads(raw)
            stored_at = float(entry["stored_at"])
            payload = entry["payload"]
        except (ValueError, KeyError, TypeError):
            return None, None
        if entry.get("v") != CACHE_FORMAT_VERSION or not self._fresh(stored_at):
            return None, None
        self.mem.put(key, stored_at, payload, len(raw))
        return payload, "persistent"

    async def put(self, key: str, payload: Any, key_material: dict[str, Any]) -> None:
        now = self._clock()
        doc = json.dumps(
            {
                "v": CACHE_FORMAT_VERSION,
                "stored_at": now,
                "key_material": key_material,
                "payload": payload,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.mem.put(key, now, payload, len(doc))
        if self.store is None:
            return
        try:
            await self.store.put(self._path(key), doc)
        except Exception as e:  # noqa: BLE001
            log.warning("web_cache.write_failed ns=%s err=%s", self.namespace, type(e).__name__)

    async def single_flight(self, key: str, make):
        """Run ``make()`` once per key at a time in this process; concurrent callers
        await the same result. Returns ``(result, shared)``."""
        fut = self._inflight.get(key)
        if fut is not None:
            return await asyncio.shield(fut), True
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            result = await make()
        except BaseException as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved so a lone caller does not warn
            raise
        else:
            fut.set_result(result)
            return result, False
        finally:
            self._inflight.pop(key, None)


_CACHES: dict[str, TwoLevelCache] = {}
_STORE_SENTINEL: dict[str, Any] = {}


def get_cache(namespace: str) -> TwoLevelCache:
    """Process-wide cache for ``namespace``, built from the environment on first use."""
    cache = _CACHES.get(namespace)
    if cache is None:
        if "store" not in _STORE_SENTINEL:
            _STORE_SENTINEL["store"] = store_from_url(os.environ.get("BROWSECOMP_WEB_CACHE_URL"))
            store = _STORE_SENTINEL["store"]
            log.info(
                "web_cache.configured persistent=%s ttl_s=%s",
                store.describe() if store is not None else "none",
                _float_env("BROWSECOMP_WEB_CACHE_TTL_S", DEFAULT_TTL_S),
            )
        cache = TwoLevelCache(
            namespace,
            store=_STORE_SENTINEL["store"],
            ttl_s=_float_env("BROWSECOMP_WEB_CACHE_TTL_S", DEFAULT_TTL_S),
            mem_bytes=int(_float_env("BROWSECOMP_WEB_CACHE_MEM_MB", DEFAULT_MEM_MB) * 1024 * 1024),
        )
        _CACHES[namespace] = cache
    return cache


def reset_caches() -> None:
    """Tests only: forget every cache and the configured store."""
    _CACHES.clear()
    _STORE_SENTINEL.clear()
