"""Run a Python script and append its wall-clock time to data/timings.csv.

Usage: python scripts/timed.py SCRIPT [ARGS...]. The Makefile runs every data generator
through this wrapper, so `make data` records how long each experiment takes, and on what
hardware. `make timings` prints the log.
"""

import csv
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "data" / "timings.csv"
FIELDS = ["finished", "script", "arguments", "minutes", "exit_status", "cpu", "cores", "python"]


def cpu_name() -> str:
    if sys.platform == "darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main() -> None:
    script, *arguments = sys.argv[1:]
    start = time.perf_counter()
    status = subprocess.run([sys.executable, script, *arguments]).returncode
    minutes = (time.perf_counter() - start) / 60
    new_log = not LOG.exists()
    with LOG.open("a", newline="") as log:
        writer = csv.writer(log, lineterminator="\n")
        if new_log:
            writer.writerow(FIELDS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), script, " ".join(arguments),
            f"{minutes:.2f}", status, cpu_name(), os.cpu_count(), platform.python_version(),
        ])
    print(f"{script}: {minutes:.1f} min (logged to {LOG.name})", file=sys.stderr)
    sys.exit(status)


if __name__ == "__main__":
    main()
