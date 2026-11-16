"""Synchronous PP Cache reset requests to worker-owned Cache domains."""

import hmac
import secrets
import socket
import socketserver
import threading
from typing import Callable


class CacheResetServer:
    """Expose one worker's optional reset hook through PP handshake metadata."""

    def __init__(self, reset: Callable[[], bool], host: str | None = None):
        if host is None:
            try:
                from vllm.utils.network_utils import get_ip
            except ImportError:
                from vllm.utils import get_ip

            host = get_ip()
        token = secrets.token_bytes(32)

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(5)
                try:
                    received = bytearray()
                    while len(received) < len(token):
                        part = self.request.recv(len(token) - len(received))
                        if not part:
                            return
                        received.extend(part)
                    if not hmac.compare_digest(received, token):
                        self.request.sendall(b"0")
                        return
                    self.request.sendall(b"1" if reset() else b"0")
                except Exception:
                    # Closing without a success reply makes the caller fail.
                    return

        class Server(socketserver.TCPServer):
            address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

        self._server = Server((host, 0), Handler)
        self.endpoint = {
            "host": host,
            "port": self._server.server_address[1],
            "token": token.hex(),
        }
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="ucm-cache-reset",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


def reset_cache_endpoints(endpoints: list[dict], timeout: float = 30) -> bool:
    """Wait for every domain, attempting later domains even after a failure."""
    if not endpoints:
        return False
    success = True
    for endpoint in endpoints:
        try:
            with socket.create_connection(
                (endpoint["host"], endpoint["port"]), timeout=timeout
            ) as client:
                client.sendall(bytes.fromhex(endpoint["token"]))
                completed = client.recv(1) == b"1"
        except Exception:
            completed = False
        success = completed and success
    return success
