import os
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

from hybrid_rag.client import DaemonError, call_daemon, query
from hybrid_rag.daemon import make_server
from hybrid_rag.embedders import HashingEmbedder
from hybrid_rag.retriever import Retriever


@pytest.fixture
def server(built):
    d = Path(tempfile.mkdtemp(prefix="hr-", dir="/tmp"))   # Unix socket paths must stay short
    built.daemon.socket = str(d / "s.sock")
    built.daemon.admit_wait_s = 0.2
    srv = make_server(built, Retriever(built, embedder=HashingEmbedder(256), memory_probe=lambda: None))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield built, srv
    srv.shutdown()
    srv.server_close()
    shutil.rmtree(d, ignore_errors=True)


def test_ping_and_query(server):
    cfg, _ = server
    ping = call_daemon(cfg, {"op": "ping"})
    assert ping["dense_ready"] is True and ping["served"] == 0
    reply = call_daemon(cfg, {"op": "query", "query": "How long are nightly backups kept?", "k": 2})
    assert reply["result"]["mode"] == "hybrid" and reply["result"]["hits"][0]["note_id"] == "backup"
    assert call_daemon(cfg, {"op": "ping"})["served"] == 1
    out = query(cfg, "Team plan cost per seat", 1)
    assert out["via"] == "daemon" and out["hits"][0]["note_id"] == "price-2026"


def test_socket_is_owner_only(server):
    cfg, _ = server
    assert (Path(cfg.socket_path()).stat().st_mode & 0o777) == 0o600


def test_busy_heavy_slot_degrades(server):
    cfg, srv = server
    srv.state.heavy.acquire()
    try:
        res = call_daemon(cfg, {"op": "query", "query": "backups kept", "k": 1})["result"]
    finally:
        srv.state.heavy.release()
    assert res["mode"] == "degraded" and "busy" in res["reason"] and res["hits"]


def test_errors_are_reported(server):
    cfg, _ = server
    with pytest.raises(DaemonError, match="unknown op"):
        call_daemon(cfg, {"op": "explode"})


def test_client_falls_back_without_daemon(built, tmp_path):
    built.daemon.socket = str(tmp_path / "none.sock")
    with pytest.raises(DaemonError):
        call_daemon(built, {"op": "ping"})
    out = query(built, "How long are nightly backups kept?", 2)
    assert out["via"] == "in-process" and out["mode"] == "degraded" and out["hits"][0]["note_id"] == "backup"


def test_tcp_mode_refuses_non_loopback_hosts(built):
    from hybrid_rag.daemon import is_loopback
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.10")
    with pytest.raises(ValueError):
        make_server(built, tcp=("0.0.0.0", 0), warm=False)


def test_default_socket_lives_in_a_private_folder(built, monkeypatch):
    d = Path(tempfile.mkdtemp(prefix="hr-", dir="/tmp"))
    monkeypatch.setenv("TMPDIR", str(d))
    built.daemon.socket = ""
    path = Path(built.socket_path())
    assert path.parent.name == f"hrag-{os.getuid()}"
    srv = make_server(built, warm=False)
    try:
        assert (path.parent.stat().st_mode & 0o777) == 0o700
        assert (path.stat().st_mode & 0o777) == 0o600
    finally:
        srv.server_close()
        shutil.rmtree(d, ignore_errors=True)


def test_client_refuses_a_socket_owned_by_someone_else(server, monkeypatch):
    cfg, _ = server
    real = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real + 1)
    with pytest.raises(DaemonError, match="another user"):
        call_daemon(cfg, {"op": "ping"})


def test_tcp_mode_requires_the_token(built):
    import socket as so
    d = Path(tempfile.mkdtemp(prefix="hr-", dir="/tmp"))
    built.daemon.socket = str(d / "s.sock")
    srv = make_server(built, Retriever(built, embedder=HashingEmbedder(256), memory_probe=lambda: None),
                      tcp=("127.0.0.1", 0), warm=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        addr = srv.server_address[:2]
        tok = d / "tcp.token"
        host, port, token = tok.read_text().split()
        assert (tok.stat().st_mode & 0o777) == 0o600 and (host, port) == ("127.0.0.1", str(addr[1]))
        assert len(token) == 64
        assert call_daemon(built, {"op": "ping"}, tcp=addr)["ok"] is True   # client sends the token
        with pytest.raises(DaemonError, match="token not sent"):           # wrong port: no token leaves
            call_daemon(built, {"op": "ping"}, tcp=(addr[0], addr[1] + 1))
        with pytest.raises(DaemonError, match="loopback"):
            call_daemon(built, {"op": "ping"}, tcp=("192.0.2.1", addr[1]))
        with pytest.raises(DaemonError, match="token not sent"):           # other loopback host
            call_daemon(built, {"op": "ping"}, tcp=("127.0.0.2", addr[1]))
        with so.create_connection(addr, timeout=2) as raw:                  # a caller without it
            raw.sendall(b'{"op": "ping"}\n')
            reply = raw.makefile().readline()
        assert '"ok": false' in reply and "token" in reply and "proof" not in reply
        good = call_daemon(built, {"op": "ping"}, tcp=addr)
        assert len(good["proof"]) == 64                                     # HMAC-SHA256 over the nonce
    finally:
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(d, ignore_errors=True)


def test_oversized_request_is_rejected_before_parsing(server):
    import socket as so
    cfg, _ = server
    with so.socket(so.AF_UNIX, so.SOCK_STREAM) as raw:
        raw.settimeout(5)
        raw.connect(cfg.socket_path())
        raw.sendall(b"x" * (cfg.daemon.max_query_chars * 4 + 5000) + b"\n")
        reply = raw.makefile().readline()
    assert '"ok": false' in reply and "too long" in reply
