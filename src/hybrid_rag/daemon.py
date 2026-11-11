"""A small long-lived retrieval daemon (stdlib only).

Loads the models once, then answers newline-delimited JSON over a Unix socket
(or TCP on a loopback address). One request per connection::

    -> {"op": "query", "query": "...", "k": 5, "historical": null, "force_sparse": false}
    <- {"ok": true, "result": {...RetrievalResult...}}
    -> {"op": "ping"}
    <- {"ok": true, "op": "ping", "dense_ready": true, "heavy_busy": false, "served": 12}

Model inference is admitted through one "heavy slot": a request that cannot get
it within ``admit_wait_s``, or arrives while ``admit_max_waiters`` requests are
already queued, is answered by keyword search at once (``mode = "degraded"``).
So a busy or slow model never makes a client wait for minutes.

Access control: the Unix socket is 0600 inside a 0700 folder owned by the user.
TCP mode binds loopback only and requires a random token that the daemon writes
(0600, as ``"<host> <port> <token>"``) into that same private folder; every TCP request
must carry it as ``"token"``, so other local users cannot query the daemon. The
client sends the token only to the loopback host and port recorded with it, adds a
random ``"nonce"``, and accepts the reply only if its ``"proof"`` is
HMAC-SHA256(token, nonce), so a process that grabbed the port cannot fake answers.
The token file is removed when the daemon stops; each start writes a new token.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import signal
import socketserver
import stat
import threading
from pathlib import Path

from .config import Config
from .retriever import DenseUnavailable, Retriever

log = logging.getLogger(__name__)


class _State:
    def __init__(self, cfg: Config, retriever: Retriever) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.heavy = threading.Semaphore(1)
        self.waiters = 0
        self.waiters_lock = threading.Lock()
        self.served = 0
        self.busy = False
        self.token = ""      # set in TCP mode; then every request must carry it


def _handle_query(state: _State, req: dict) -> dict:
    cfg = state.cfg
    query = str(req.get("query") or "")[: cfg.daemon.max_query_chars]
    k = max(1, min(20, int(req.get("k") or 5)))
    historical = req.get("historical")
    if bool(req.get("force_sparse")):
        return state.retriever.retrieve(query, k, historical=historical, force_sparse=True).to_dict()
    with state.waiters_lock:
        crowded = state.waiters >= cfg.daemon.admit_max_waiters
        if not crowded:
            state.waiters += 1
    if crowded:
        res = state.retriever.retrieve(query, k, historical=historical, force_sparse=True)
        res.reason = "admission queue full"
        return res.to_dict()
    try:
        got = state.heavy.acquire(timeout=cfg.daemon.admit_wait_s)
    finally:
        with state.waiters_lock:
            state.waiters -= 1
    if not got:
        res = state.retriever.retrieve(query, k, historical=historical, force_sparse=True)
        res.reason = f"heavy slot busy for {cfg.daemon.admit_wait_s:.0f}s"
        return res.to_dict()
    try:
        state.busy = True
        return state.retriever.retrieve(query, k, historical=historical).to_dict()
    finally:
        state.busy = False
        state.heavy.release()


def token_proof(token: str, nonce: str) -> str:
    """What the daemon sends back so a TCP client knows it reached the real daemon."""
    return hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest()


class _Handler(socketserver.StreamRequestHandler):
    server: "_UnixServer | _TCPServer"
    timeout = 30.0   # a silent or slow sender cannot hold a thread forever

    def handle(self) -> None:
        state: _State = self.server.state  # type: ignore[attr-defined]
        authed = False
        req: dict = {}
        try:
            raw = self.rfile.readline(state.cfg.daemon.max_query_chars * 4 + 4096)   # bounded read
            if not raw:
                return
            if not raw.endswith(b"\n"):
                raise ValueError("request too long or not newline-terminated")
            req = json.loads(raw.decode("utf-8"))
            op = req.get("op", "query")
            authed = not state.token or hmac.compare_digest(str(req.get("token", "")).encode(), state.token.encode())
            if not authed:
                payload = {"ok": False, "error": "missing or wrong token"}
            elif op == "ping":
                try:
                    state.retriever.dense_index()
                    dense_ready = True
                except DenseUnavailable:
                    dense_ready = False
                payload = {"ok": True, "op": "ping", "dense_ready": dense_ready,
                           "heavy_busy": state.busy, "served": state.served}
            elif op == "query":
                payload = {"ok": True, "result": _handle_query(state, req)}
                state.served += 1
            else:
                payload = {"ok": False, "error": f"unknown op '{op}'"}
        except Exception as exc:  # noqa: BLE001 - report the cause to the client
            log.exception("request failed")
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if state.token and authed:
            payload["proof"] = token_proof(state.token, str(req.get("nonce", "")))
        try:
            self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.wfile.flush()
        except OSError as exc:
            log.warning("client went away before the answer (%s)", exc)


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class _TCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def is_loopback(host: str) -> bool:
    """True for localhost and loopback IP literals (127.0.0.0/8, ::1)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_private_dir(folder: Path) -> None:
    """Raise unless ``folder`` belongs to this user and nobody else can write to it."""
    st = folder.lstat()
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"socket folder {folder} is not a plain directory")
    if st.st_uid != os.getuid():
        raise RuntimeError(f"socket folder {folder} belongs to another user")
    if st.st_mode & 0o022:
        raise RuntimeError(f"socket folder {folder} is writable by other users")


def _prepare_socket_dir(folder: Path) -> None:
    """Create the socket folder 0700, or check an existing one is ours and private."""
    if not folder.exists():
        folder.mkdir(mode=0o700, parents=True)
        return
    check_private_dir(folder)


def token_path(cfg: Config) -> Path:
    """Where the TCP token lives: next to the socket, in the private folder."""
    return Path(cfg.socket_path()).parent / "tcp.token"


def _write_token(cfg: Config, host: str, port: int) -> str:
    path = token_path(cfg)
    _prepare_socket_dir(path.parent)
    token = secrets.token_hex(32)
    if path.exists() or path.is_symlink():
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as fh:
        fh.write(f"{host} {port} {token}")
    return token


def make_server(cfg: Config, retriever: Retriever | None = None, *, tcp: tuple[str, int] | None = None,
                warm: bool = True):
    """Build (not start) the server; ``serve_forever()`` on the result runs it."""
    retriever = retriever or Retriever(cfg)
    if warm:
        retriever.warm()
    if tcp is not None:
        if not is_loopback(tcp[0]):
            raise ValueError(f"TCP mode listens on loopback only, not '{tcp[0]}'")
        srv = _TCPServer(tcp, _Handler)
        token = _write_token(cfg, tcp[0], srv.server_address[1])
    else:
        token = ""
        path = Path(cfg.socket_path())
        _prepare_socket_dir(path.parent)
        if path.exists() or path.is_symlink():
            st = path.lstat()
            if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
                raise RuntimeError(f"{path} exists and is not this user's socket; refusing to replace it")
            path.unlink()   # stale socket from a crashed run
        srv = _UnixServer(str(path), _Handler)
        os.chmod(path, 0o600)   # owner only
    srv.state = _State(cfg, retriever)  # type: ignore[attr-defined]
    srv.state.token = token  # type: ignore[attr-defined]
    return srv


def serve(cfg: Config, *, tcp: tuple[str, int] | None = None) -> None:
    srv = make_server(cfg, tcp=tcp)
    where = f"tcp {tcp[0]}:{tcp[1]}" if tcp else cfg.socket_path()
    log.info("hrag daemon listening on %s", where)

    def _stop(signum: int, _frame: object) -> None:   # SIGTERM cleans up like Ctrl-C
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _stop)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if tcp is None:
            Path(cfg.socket_path()).unlink(missing_ok=True)
        else:
            token_path(cfg).unlink(missing_ok=True)
