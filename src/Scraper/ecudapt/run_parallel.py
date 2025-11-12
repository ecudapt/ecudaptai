#!/usr/bin/env python3
"""
run_parallel.py

Runs multiple Scrapy forum crawls in parallel, each with its own:
  • URL file (in the urls/ folder)
  • log file
  • JOBDIR (persistent, so crawls can resume)

✅ Auto-discovers URL files under urls/
✅ Streams Scrapy output to the terminal
✅ Writes logs to logs/<base>_<timestamp>.log
✅ Uses stable JOBDIR per URL file so you can resume after stopping
✅ Each spider's console output is colorized differently
"""

import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init

init(autoreset=True)

SPIDER_NAME = "forums_hybrid"

# This file lives in: src/Scraper/ecudapt/run_parallel.py
# Scrapy project root (where scrapy.cfg is) is: src/Scraper
BASE_DIR = Path(__file__).resolve().parents[1]  # /src/Scraper
URL_DIR = BASE_DIR / "urls"                     # /src/Scraper/urls

LOG_DIR = BASE_DIR / "logs"
CRAWL_DIR = BASE_DIR / "crawls"

for d in (LOG_DIR, CRAWL_DIR, URL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Colors we’ll cycle through for spider prefixes
SPIDER_COLORS = [
    Fore.CYAN,
    Fore.GREEN,
    Fore.MAGENTA,
    Fore.YELLOW,
    Fore.BLUE,
    Fore.RED,
    Fore.WHITE,
]


def discover_url_files(pattern: str = "*.txt"):
    """
    Find all URL list files matching the pattern in URL_DIR.
    Example: urls_1.txt, vw_forums.txt, bmw_ecu.txt, etc.
    """
    files = sorted(URL_DIR.glob(pattern))
    return files


def start_crawl(url_path: Path, color: str):
    base = url_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = LOG_DIR / f"{base}_{ts}.log"

    # 🔑 Stable JOBDIR: one per URL file, no timestamp
    job_dir = CRAWL_DIR / base
    job_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{color}🚀 Starting crawl for {url_path.relative_to(BASE_DIR)} "
        f"{Style.RESET_ALL}→ log: {log_file} | JOBDIR: {job_dir}"
    )

    cmd = [
        sys.executable,
        "-m",
        "scrapy",
        "crawl",
        SPIDER_NAME,
        "-a",
        f"urls_file={str(url_path)}",
        "-s",
        f"JOBDIR={str(job_dir)}",
        "-s",
        "LOG_LEVEL=WARNING",  # spider uses logger.warning for the lines you care about
    ]

    # Capture stdout/stderr so we can tee to console + log file
    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    def reader():
        with log_file.open("w", encoding="utf-8") as f:
            for line in proc.stdout:
                # print to terminal with colored prefix per spider
                print(f"{color}[{base}]{Style.RESET_ALL} {line.rstrip()}")
                f.write(line)
        proc.stdout.close()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return proc, t


def main():
    # 🔍 Auto-discover URL files under urls/
    url_files = discover_url_files("*.txt")

    if not url_files:
        print(
            f"{Fore.RED}❌ No URL files found in {URL_DIR} (expected *.txt){Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}👉 Create files like {URL_DIR / 'my_forums.txt'} "
            f"with one forum URL per line.{Style.RESET_ALL}"
        )
        return

    print(
        f"{Fore.MAGENTA}Found {len(url_files)} URL file(s) in {URL_DIR.name}: "
        f"{', '.join(p.name for p in url_files)}{Style.RESET_ALL}"
    )

    procs = []

    # Assign colors in a round-robin fashion across all spiders
    for idx, path in enumerate(url_files):
        color = SPIDER_COLORS[idx % len(SPIDER_COLORS)]
        procs.append(start_crawl(path, color))

    try:
        # Wait for all crawls to finish
        for proc, t in procs:
            proc.wait()
            t.join()
    except KeyboardInterrupt:
        print(
            f"{Fore.RED}⏹ Interrupted by user. Terminating all crawls...{Style.RESET_ALL}"
        )
        for proc, _ in procs:
            proc.terminate()
    finally:
        print(f"{Fore.GREEN}✅ All crawls complete.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
