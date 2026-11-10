"""Run every experiment in order and record the computing environment."""
import json
import platform
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["01_synthetic_1d.py", "02_jura_estimation.py", "03_jura_latent_magnitude.py",
         "04_jura_anisotropy.py", "05_jura_crossvalidation.py", "06_figure4.py",
         "07_controlled_simulation.py"]


def environment():
    import numpy, scipy, matplotlib, pandas
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    import os
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "pandas": pandas.__version__,
            "platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count()}


def main():
    times = {}
    for step in STEPS:
        t0 = time.perf_counter()
        print(f"\n=== {step} ===", flush=True)
        runpy.run_path(str(ROOT / "scripts" / step), run_name="__main__")
        times[step] = round(time.perf_counter() - t0, 2)
    out = {"environment": environment(), "wall_clock_s": times, "total_s": round(sum(times.values()), 2)}
    (ROOT / "results" / "run_record.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
