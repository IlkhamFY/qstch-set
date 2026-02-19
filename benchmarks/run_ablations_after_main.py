#!/usr/bin/env python3
"""Wait for main benchmark to finish, then run ablations."""
import sys, os, time, subprocess
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
PYTHON = r"C:\Users\zolot\miniforge3\python.exe"
BENCHDIR = Path(__file__).parent

def wait_for_main():
    """Wait until summary.json exists (main benchmark done)."""
    print("Waiting for main benchmark to complete...", flush=True)
    while not (RESULTS / "summary.json").exists():
        time.sleep(300)  # check every 5 min
        log = RESULTS / "benchmark_stdout.log"
        if log.exists():
            age = time.time() - log.stat().st_mtime
            print(f"  Still running... log updated {age:.0f}s ago", flush=True)
    print("Main benchmark complete!", flush=True)

def run_script(name):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BENCHDIR.parent / "src")
    print(f"\n{'='*50}\nRunning {name}\n{'='*50}", flush=True)
    subprocess.run([PYTHON, "-u", str(BENCHDIR / name)], env=env)

if __name__ == "__main__":
    wait_for_main()
    run_script("ablation_mu.py")
    run_script("ablation_K.py")
    print("\nAll ablations complete!", flush=True)
