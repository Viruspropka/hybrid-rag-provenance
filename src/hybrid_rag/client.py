"""Client for the daemon, with an in-process keyword fallback.

``query`` tries the daemon first; on any failure (no daemon, stale socket,
timeout, bad JSON) it answers in-process. By default the fallback is
keyword-only, so a client never loads the heavy models just because the daemon
is down.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import socket
import stat
from pathlib import Path

from .config import Config
from .daemon import check_private_dir, is_loopback, token_path, token_proof
from .retriever import Retriever


class DaemonError(RuntimeError):
    pass


def call_daemon(cfg: Config, payload: dict, *, connect_timeout: float = 1.0, timeout: float = 30.0,
                tcp: tuple[str, int] | None = None) -> dict:
    # The private folder (ours, not writable by others) is what makes the checks
    # below hold until connect(): nobody else can swap the socket or the token.
    token = nonce = ""
    path = Path(cfg.socket_path())
    try:
        check_private_dir(path.parent)
    except (OSError, RuntimeError) as exc:
        raise DaemonError(f"not trusting the socket folder: {exc}") from exc
    if tcp is not None:
        if not is_loopback(tcp[0]):
            raise DaemonError(f"TCP mode is loopback only, not '{tcp[0]}'")
        try:
            fields_ = token_path(cfg).read_text(encoding="ascii").split()
        except OSError as exc:
            raise DaemonError(f"no TCP token (is the daemon running in TCP mode?): {exc}") from exc
        host_s, port_s, token = (fields_ + ["", "", ""])[:3]
        if not token or (host_s, port_s) != (tcp[0], str(tcp[1])):
            raise DaemonError(f"the running daemon listens on {host_s}:{port_s}, not {tcp[0]}:{tcp[1]}; "
                              "token not sent")
        nonce = secrets.token_hex(16)
        payload = {**payload, "token": token, "nonce": nonce}
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address: object = tcp
    else:
        try:
            st = path.lstat()
        except OSError as exc:
            raise DaemonError(f"daemon unreachable: {exc}") from exc
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise DaemonError(f"{path} is not a socket owned by this user; not trusting it")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = str(path)
    try:
        sock.settimeout(connect_timeout)
        sock.connect(address)
        sock.settimeout(timeout)
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    except OSError as exc:
        raise DaemonError(f"daemon unreachable: {exc}") from exc
    finally:
        sock.close()
    try:
        reply = json.loads(buf.decode("utf-8"))
    except ValueError as exc:
        raise DaemonError("daemon sent invalid JSON") from exc
    if not reply.get("ok"):
        raise DaemonError(reply.get("error", "daemon error"))
    if tcp is not None and not hmac.compare_digest(str(reply.get("proof", "")), token_proof(token, nonce)):
        raise DaemonError("TCP reply without a valid proof; not the daemon that wrote the token")
    return reply


def query(cfg: Config, text: str, k: int = 5, *, historical: bool | None = None,
          use_daemon: bool = True, full_fallback: bool = False, tcp: tuple[str, int] | None = None) -> dict:
    """Result dict from the daemon, else from an in-process retriever.

    ``full_fallback`` = run the whole hybrid pipeline in-process (loads models);
    otherwise the fallback is keyword-only.
    """
    if use_daemon:
        try:
            reply = call_daemon(cfg, {"op": "query", "query": text, "k": k, "historical": historical}, tcp=tcp)
            result = reply["result"]
            result["via"] = "daemon"
            return result
        except DaemonError:
            pass
    res = Retriever(cfg).retrieve(text, k, historical=historical,
                                  force_sparse=use_daemon and not full_fallback)
    out = res.to_dict()
    out["via"] = "in-process"
    if use_daemon and not full_fallback:
        out["reason"] = "daemon unreachable: keyword-only fallback"
    return out
