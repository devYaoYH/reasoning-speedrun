"""GPU admission checks and optional device sampling."""

import asyncio
import json
import threading
import time
from reasoning_speedrun.lib.common import utc_now


def append_json(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()


class GPUSampler:
    """Device-level NVML samples, shared by concurrent rollouts (not attribution)."""

    def __init__(self, path, interval=0.2, device=0):
        self.path, self.interval, self.device = (path, interval, device)
        self.samples = []
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            with self.path.open("w") as file:
                while not self.stop_event.is_set():
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    row = {
                        "monotonic_s": time.perf_counter(),
                        "timestamp_utc": utc_now(),
                        "vram_used_mib": memory.used / 2**20,
                        "vram_total_mib": memory.total / 2**20,
                        "gpu_util_pct": util.gpu,
                    }
                    self.samples.append(row)
                    append_json(file, row)
                    self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    async def start(self):
        self.thread.start()
        for _ in range(100):
            if self.error:
                raise RuntimeError(f"GPU telemetry failed: {self.error}")
            if self.samples:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("GPU telemetry did not produce a sample")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def window(self, start, end):
        snapshot = list(self.samples)
        before = [s for s in snapshot if s["monotonic_s"] <= start]
        rows = ([before[-1]] if before else []) + [
            s for s in snapshot if start < s["monotonic_s"] <= end
        ]
        return {
            "sample_count": len(rows),
            "start_vram_mib": rows[0]["vram_used_mib"] if rows else None,
            "end_vram_mib": rows[-1]["vram_used_mib"] if rows else None,
            "observed_peak_vram_mib": max(
                (s["vram_used_mib"] for s in rows), default=None
            ),
            "scope": "shared GPU device; sampled peak, not per-request allocation",
            "error": self.error,
        }


def assert_gpu_idle(device):
    """Fail before warmup if another CUDA workload is present; never terminate it."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if processes:
            raise RuntimeError(
                f"GPU {device} occupied by PIDs {[p.pid for p in processes]}; defer the sweep"
            )
    finally:
        pynvml.nvmlShutdown()
