"""Owned process groups, readiness and checkout exclusion."""

import asyncio
from contextlib import contextmanager
import fcntl
import os
import signal
import socket
import subprocess
import time
import httpx
from qed.lib.common import workdir


def ensure_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


class Services:

    def __init__(self):
        self.processes = []
        self.logs = []

    def launch(self, command, logfile, env=None):
        file = logfile.open("w")
        self.logs.append(file)
        process = subprocess.Popen(
            command,
            stdout=file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            cwd=workdir(),
        )
        self.processes.append(process)
        return process

    async def close(self):
        for process in reversed(self.processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.perf_counter() + 15
        while (
            any((p.poll() is None for p in self.processes))
            and time.perf_counter() < deadline
        ):
            await asyncio.sleep(0.1)
        for process in self.processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for file in self.logs:
            file.close()


async def ready(client, url, timeout, process=None):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Service exited ({process.returncode}); inspect attempt service logs"
            )
        try:
            response = await client.get(url, timeout=2)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Service not ready: {url}")


@contextmanager
def attempt_lock():
    with (workdir() / ".attempt.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another canonical attempt owns this working directory") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
