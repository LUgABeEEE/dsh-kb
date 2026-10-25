"""A tiny, dependency-free WebDAV client (stdlib only).

Deliberately vendor-neutral: it speaks plain RFC 4918 to any WebDAV server
(Nextcloud, Synology, ownCloud, a NAS appliance, Apache mod_dav …).  Optional
client-certificate support covers servers that require mutual TLS.

Percent-encoding rule (a classic source of bugs)
-----------------------------------------------
Callers always pass **decoded, bundle-relative POSIX paths** (``知识库/a b.md``).
:meth:`WebDAVClient._url` quotes each segment exactly once.  Never pre-quote a
path before handing it over — doing so produces the infamous
``%E7%9F%A5...`` double-encoded directory.
"""
from __future__ import annotations

import base64
import http.client
import ssl
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

DAV = "{DAV:}"
PROPFIND_BODY = (
    b'<?xml version="1.0" encoding="utf-8"?>'
    b'<D:propfind xmlns:D="DAV:"><D:prop>'
    b"<D:resourcetype/><D:getcontentlength/><D:getlastmodified/>"
    b"</D:prop></D:propfind>"
)


class WebDAVError(RuntimeError):
    """Raised when the server answers with an unexpected status."""

    def __init__(self, method: str, path: str, status: int, body: bytes = b""):
        self.method, self.path, self.status, self.body = method, path, status, body
        super().__init__(f"{method} {path} -> HTTP {status}: {body[:200]!r}")


@dataclass
class DavEntry:
    """One PROPFIND response row."""

    href: str
    is_dir: bool
    size: int
    modified: str


class WebDAVClient:
    """Minimal WebDAV client bound to one base collection."""

    def __init__(
        self,
        base_url: str,
        *,
        username: str = "",
        password: str = "",
        cert: str | None = None,
        key: str | None = None,
        ca: str | None = None,
        host_header: str | None = None,
        connect_host: str | None = None,
        verify: bool = True,
        timeout: float = 30.0,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"base_url must be http(s): {base_url!r}")
        self.scheme = parsed.scheme
        self.host = parsed.hostname or ""
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # The collection path must be percent-encoded before it can be written
        # into a request line: http.client encodes request targets as ASCII, so a
        # non-ASCII path raises UnicodeEncodeError inside the request and the
        # whole transport fails silently.  `unquote` then `quote` makes this
        # idempotent, so an already-encoded base_url is not double-encoded.
        segments = [urllib.parse.unquote(seg) for seg in parsed.path.split("/") if seg]
        self.base_path = "/" + "/".join(urllib.parse.quote(seg, safe="") for seg in segments) if segments else ""
        self.host_header = host_header or parsed.netloc
        # Some appliances serve a certificate whose CN differs from the address
        # used to reach them (typically an IP).  `connect_host` separates the two.
        self.connect_host = connect_host or self.host
        self.timeout = timeout
        self.username = username
        self.password = password
        self._verify = verify
        self._cert, self._key, self._ca = cert, key, ca

    # ---- plumbing ------------------------------------------------------
    def _ssl_context(self) -> ssl.SSLContext:
        if self._cert and self._key:
            ctx = ssl.create_default_context(cafile=self._ca)
            ctx.load_cert_chain(certfile=self._cert, keyfile=self._key)
        elif self._ca:
            ctx = ssl.create_default_context(cafile=self._ca)
        else:
            ctx = ssl.create_default_context()
        if not self._verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        # When a server certificate carries a CN that differs from the address
        # used to reach it, TLS verification must target the CN, not the IP.
        self._sni = (self.host_header.split(":")[0] or self.connect_host)
        return ctx

    def _url(self, path: str) -> str:
        """Build the request target from a **decoded** bundle-relative path.

        Idempotent by construction (unquote then quote), so passing an
        already-encoded path is harmless rather than producing ``%25`` escapes.
        """
        rel = path.strip("/")
        segments = [urllib.parse.unquote(seg) for seg in rel.split("/") if seg]
        quoted = "/".join(urllib.parse.quote(seg, safe="") for seg in segments)
        if not quoted:
            return self.base_path or "/"
        return f"{self.base_path}/{quoted}" if self.base_path else f"/{quoted}"

    def display_path(self, path: str = "") -> str:
        """Human-readable (decoded) form of a request target."""
        return urllib.parse.unquote(self._url(path))

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        hdrs = {"Host": self.host_header, "Connection": "close", "User-Agent": "dsh-kb/0.1"}
        if self.username or self.password:
            token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            hdrs["Authorization"] = "Basic " + token
        if body is not None:
            hdrs["Content-Length"] = str(len(body))
        hdrs.update(headers or {})

        ctx = self._ssl_context()
        if self.scheme == "https":
            conn = http.client.HTTPSConnection(self.connect_host, self.port, context=ctx, timeout=self.timeout)
            conn.sock = ctx.wrap_socket(
                __import__("socket").create_connection((self.connect_host, self.port), timeout=self.timeout),
                server_hostname=getattr(self, "_sni", self.connect_host),
            )
        else:
            conn = http.client.HTTPConnection(self.connect_host, self.port, timeout=self.timeout)
        try:
            conn.request(method, self._url(path), body=body, headers=hdrs)
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, {k: v for k, v in resp.getheaders()}, data
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ---- operations ----------------------------------------------------
    def listdir(self, path: str = "") -> list[DavEntry]:
        """PROPFIND ``Depth: 1``; the collection itself is filtered out."""
        status, _, data = self.request(
            "PROPFIND",
            path,
            body=PROPFIND_BODY,
            headers={"Content-Type": "application/xml; charset=utf-8", "Depth": "1"},
        )
        if status not in (207, 200):
            raise WebDAVError("PROPFIND", path, status, data)

        out: list[DavEntry] = []
        root = ET.fromstring(data)
        for resp in root.findall(f"{DAV}response"):
            href = (resp.findtext(f"{DAV}href") or "").strip()
            if not href:
                continue
            prop = resp.find(f"{DAV}propstat/{DAV}prop")
            if prop is None:
                continue
            is_dir = prop.find(f"{DAV}resourcetype/{DAV}collection") is not None
            size_text = prop.findtext(f"{DAV}getcontentlength") or "0"
            out.append(
                DavEntry(
                    href=href,
                    is_dir=is_dir,
                    size=int(size_text) if size_text.isdigit() else 0,
                    modified=prop.findtext(f"{DAV}getlastmodified") or "",
                )
            )
        # Drop the self entry so callers only see children.
        want = urllib.parse.unquote(self._url(path)).rstrip("/")
        return [e for e in out if urllib.parse.unquote(e.href).rstrip("/") != want]

    def get(self, path: str) -> bytes:
        status, _, data = self.request("GET", path)
        if status != 200:
            raise WebDAVError("GET", path, status, data)
        return data

    def put(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> int:
        status, _, body = self.request("PUT", path, body=data, headers={"Content-Type": content_type})
        if status not in (200, 201, 204):
            raise WebDAVError("PUT", path, status, body)
        return status

    def mkcol(self, path: str) -> int:
        status, _, body = self.request("MKCOL", path)
        if status not in (200, 201, 405):  # 405 = already exists
            raise WebDAVError("MKCOL", path, status, body)
        return status

    def delete(self, path: str) -> int:
        status, _, body = self.request("DELETE", path)
        if status not in (200, 204, 404):
            raise WebDAVError("DELETE", path, status, body)
        return status

    def probe(self) -> dict:
        """Cheap capability check used by ``dsh-kb doctor``."""
        result = {"base_url": self.display_path(), "reachable": False, "read": False, "write": False}
        try:
            status, _, _ = self.request("PROPFIND", "", body=PROPFIND_BODY, headers={"Depth": "0"})
            result["reachable"] = status in (207, 200)
            result["status"] = status
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result
