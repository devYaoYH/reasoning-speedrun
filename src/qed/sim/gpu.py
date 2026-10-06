"""Simulated NVML samples: same schema and window summary as the real sampler."""

import time

from qed.lib.common import utc_now
from qed.lib.gpu import GPUSampler, append_json


class SimulatedGPUSampler(GPUSampler):
    """vLLM preallocates its KV pool, so VRAM is constant at the configured fraction;
    utilization is 100% while any sequence is running. Finer models plug in here."""

    def __init__(self, path, engine, interval=0.2):
        super().__init__(path, interval, device=0)
        self.engine = engine

    def _run(self):
        c, e = self.engine.config, self.engine
        used = c.vram_total_mib * e.gpu_memory_utilization
        try:
            with self.path.open("w") as file:
                while not self.stop_event.is_set():
                    row = {
                        "monotonic_s": time.perf_counter(),
                        "timestamp_utc": utc_now(),
                        "vram_used_mib": used,
                        "vram_total_mib": c.vram_total_mib,
                        "gpu_util_pct": 100 if e.running else 0,
                    }
                    self.samples.append(row)
                    append_json(file, row)
                    self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
