#!/usr/bin/env python3
"""Quick benchmark status check. Agent-friendly: minimal output, exit code = state."""
import json, sys, os
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

def main():
    # Check if process is running
    pid_running = False
    try:
        import psutil
        for p in psutil.process_iter(['pid', 'cmdline']):
            if p.info['cmdline'] and 'run_all_benchmarks.py' in ' '.join(p.info['cmdline']):
                pid_running = True
                print(f"STATUS: RUNNING (pid {p.info['pid']})")
                break
    except ImportError:
        # fallback: check log freshness
        log = RESULTS / "benchmark_stdout.log"
        if log.exists():
            import time
            age = time.time() - log.stat().st_mtime
            pid_running = age < 120
            if pid_running:
                print(f"STATUS: LIKELY RUNNING (log updated {age:.0f}s ago)")

    if not pid_running:
        summary = RESULTS / "summary.json"
        if summary.exists():
            print("STATUS: COMPLETE")
            data = json.loads(summary.read_text())
            for k, v in data.get("results", {}).items():
                print(f"\n  {k}:")
                for method, hv in v.items():
                    print(f"    {method:20s} HV = {hv}")
            print(f"\n  Total time: {data.get('elapsed_hours', '?'):.1f}h")
            sys.exit(0)
        else:
            print("STATUS: NOT STARTED")
            sys.exit(2)

    # Show progress from log
    log = RESULTS / "benchmark_stdout.log"
    if log.exists():
        lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
        # Find completed methods
        completed = [l.strip() for l in lines if 'Final HV=' in l or 'DTLZ2 m=' in l or '--- ' in l]
        print(f"\nProgress ({len(completed)} milestones):")
        for l in completed[-15:]:
            print(f"  {l}")
        # Last activity
        for l in reversed(lines):
            if l.strip():
                print(f"\nLast line: {l.strip()}")
                break
    sys.exit(1 if pid_running else 2)

if __name__ == "__main__":
    main()
