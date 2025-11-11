#!/usr/bin/env python3
"""
run_parallel.py — run multiple Scrapy crawls concurrently
Each crawl uses its own URLs file, log, jobdir, and outputs data while printing live colored output to console.
"""
import subprocess, sys, os
from datetime import datetime
from pathlib import Path

URL_FILES = [
    "test_urls.txt",
    "test_urls_2.txt",
    "test_urls_3.txt",
]

def run_crawl(url_file):
    base = Path(url_file).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logs = Path("logs")
    crawls = Path("crawls")
    for d in (logs, crawls):
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs / f"{base}_{ts}.log"
    job_dir = crawls / f"{base}_{ts}"

    env = dict(**os.environ, PYTHONUNBUFFERED="1")

    cmd = [
        sys.executable, "-m", "scrapy", "crawl", "forums_hybrid",
        "-a", f"urls_file={url_file}",
        "-s", f"JOBDIR={job_dir}",
        "-s", "LOG_STDOUT=True",   # ✅ print to console
        "-s", f"LOG_FILE={log_file}",
        "-s", "LOG_LEVEL=INFO",
    ]

    print(f"🚀 Starting crawl for {url_file} → log: {log_file}")
    return subprocess.Popen(cmd, env=env)

def main():
    procs = [run_crawl(f) for f in URL_FILES]
    for p in procs:
        p.wait()
    print("\n✅ All crawls complete.\n")

if __name__ == "__main__":
    main()
