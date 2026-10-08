"""Sample what the Streamlit process costs while a load test runs.

Writes one row every INTERVAL seconds to results/resources.csv. The machine-wide
columns are there to answer the question that decides whether a run is worth
reading at all: is the app the bottleneck, or are the thirty test browsers?

    python loadtest_monitor.py [seconds]
"""

from __future__ import annotations

import csv
import os
import sys
import time

import psutil

INTERVAL = 5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "resources.csv")
COLUMNS = ["t", "elapsed_s", "app_cpu_pct", "app_rss_mb", "app_threads",
           "system_cpu_pct", "system_mem_pct", "browser_count",
           "browser_rss_mb", "browser_cpu_pct"]


PORT = 8501


def find_app() -> psutil.Process | None:
    """The Streamlit server process - the one actually listening on PORT.

    Matching on the command line alone is not enough: the shell that launched
    it carries the same text, so several processes match and the smallest of
    them is usually the wrapper. The listening socket identifies the server
    itself; the command-line match is only the fallback, and then the largest
    resident process wins, which is never the wrapper.
    """
    try:
        for conn in psutil.net_connections(kind="inet"):
            if (conn.laddr and conn.laddr.port == PORT
                    and conn.status == psutil.CONN_LISTEN and conn.pid):
                return psutil.Process(conn.pid)
    except (psutil.AccessDenied, psutil.Error):
        pass

    candidates = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        line = " ".join(proc.info.get("cmdline") or [])
        if "streamlit" in line and "app.py" in line:
            try:
                candidates.append((proc.memory_info().rss, proc.info["pid"]))
            except psutil.Error:
                continue
    if candidates:
        return psutil.Process(max(candidates)[1])
    return None


def browsers() -> list[psutil.Process]:
    out = []
    for proc in psutil.process_iter(["pid", "name"]):
        if (proc.info.get("name") or "").lower().startswith("chrome"):
            out.append(proc)
    return out


def keep_awake() -> None:
    """Ask Windows not to sleep while the test runs.

    A ten-minute unattended run is exactly long enough for an idle timer to
    fire, and a machine that suspends mid-test takes its numbers with it - the
    first attempt here slept at 94% RAM and came back five hours later with a
    dead run. This is a request for the lifetime of THIS process only: Windows
    forgets it the moment the monitor exits, so no power setting is changed.
    """
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception as e:                  # noqa: BLE001 - never block the run
        print(f"could not hold off sleep ({e}); carrying on")


def main() -> None:
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    keep_awake()
    app = find_app()
    if app is None:
        print("no streamlit process found - start the app first")
        sys.exit(1)
    print(f"watching streamlit pid {app.pid}, writing {OUT}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # Prime the counters: the first cpu_percent() call after attaching always
    # reports 0.0, because there is no previous sample to compare against.
    app.cpu_percent()
    psutil.cpu_percent()
    for b in browsers():
        try:
            b.cpu_percent()
        except psutil.Error:
            pass

    started = time.time()
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        while time.time() - started < limit:
            time.sleep(INTERVAL)
            now = time.time()
            try:
                with app.oneshot():
                    app_cpu = app.cpu_percent()
                    app_rss = app.memory_info().rss / 1e6
                    app_threads = app.num_threads()
            except psutil.Error:
                print("streamlit process is gone - stopping")
                break

            b_rss, b_cpu, b_n = 0.0, 0.0, 0
            for b in browsers():
                try:
                    b_rss += b.memory_info().rss / 1e6
                    b_cpu += b.cpu_percent()
                    b_n += 1
                except psutil.Error:
                    continue

            writer.writerow([
                time.strftime("%H:%M:%S"), round(now - started, 1),
                round(app_cpu, 1), round(app_rss, 1), app_threads,
                round(psutil.cpu_percent(), 1),
                round(psutil.virtual_memory().percent, 1),
                b_n, round(b_rss, 1), round(b_cpu, 1),
            ])
            fh.flush()


if __name__ == "__main__":
    main()
