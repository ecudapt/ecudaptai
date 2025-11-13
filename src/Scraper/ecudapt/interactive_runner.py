#!/usr/bin/env python3
"""
interactive_runner.py

Interactive controller for running multiple Scrapy forum crawls.

Naming conventions
------------------
Logs:      logs/crawl-<urls_stem>-<YYYYMMDD-HHMMSS>.log
JOBDIR:    crawls/<urls_stem>               (stable; used for resume)
Data:      data/raw/forums-<urls_stem>-<YYYYMMDD-HHMMSS>.jsonl (set in spider)

Commands
--------
  help, ?           Show commands
  list              List spiders and status
  start <name>      Start spider for given URL file (by stem)
  startall          Start all spiders
  stop <name>       Stop a running spider
  stopall           Stop all running spiders
  resume <name>     Same as start (will resume via JOBDIR)
  addurl <URL>      Create a new urls/manual_*.txt with that URL and start it
  reload            Rescan urls/ folder for new *.txt files
  quit, exit        Stop all spiders and exit
"""

import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init

init(autoreset=True)

SPIDER_NAME = "forums_hybrid"

# This file lives in: src/Scraper/ecudapt/interactive_runner.py
# Scrapy project root (where scrapy.cfg is) is: src/Scraper
BASE_DIR = Path(__file__).resolve().parents[1]   # /src/Scraper
URL_DIR = BASE_DIR / "urls"                      # /src/Scraper/urls
LOG_DIR = BASE_DIR / "logs"
CRAWL_DIR = BASE_DIR / "crawls"

for d in (URL_DIR, LOG_DIR, CRAWL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPIDER_COLORS = [
    Fore.CYAN,
    Fore.GREEN,
    Fore.MAGENTA,
    Fore.YELLOW,
    Fore.BLUE,
    Fore.RED,
    Fore.WHITE,
]


class SpiderJob:
    """Tracks one spider process associated with one URL file."""

    def __init__(self, url_path: Path, color: str):
        self.url_path = url_path
        self.base = url_path.stem  # used as name / prefix
        self.color = color
        self.job_dir = CRAWL_DIR / self.base
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.status: str = "stopped"  # "running", "stopped", "stopping"

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        """Start or resume this spider."""
        if self.is_running():
            print(f"{self.color}[{self.base}]{Style.RESET_ALL} already running.")
            return

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        # New, clearer log name: crawl-<urls_stem>-<timestamp>.log
        log_file = LOG_DIR / f"crawl-{self.base}-{ts}.log"

        print(
            f"{self.color}🚀 Starting crawl for {self.url_path.relative_to(BASE_DIR)} "
            f"{Style.RESET_ALL}→ log: {log_file} | JOBDIR: {self.job_dir}"
        )

        cmd = [
            sys.executable,
            "-m",
            "scrapy",
            "crawl",
            SPIDER_NAME,
            "-a",
            f"urls_file={str(self.url_path)}",
            "-s",
            f"JOBDIR={str(self.job_dir)}",
            "-s",
            "LOG_LEVEL=WARNING",
        ]

        self.proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        self.status = "running"

        def reader():
            with log_file.open("w", encoding="utf-8") as f:
                assert self.proc is not None and self.proc.stdout is not None
                for line in self.proc.stdout:
                    print(f"{self.color}[{self.base}]{Style.RESET_ALL} {line.rstrip()}")
                    f.write(line)
            if self.proc:
                self.proc.stdout and self.proc.stdout.close()
            self.status = "stopped"

        self.thread = threading.Thread(target=reader, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop this spider if it is running."""
        if not self.is_running():
            print(f"{self.color}[{self.base}]{Style.RESET_ALL} not running.")
            return
        print(f"{self.color}⏹ Stopping {self.base}...{Style.RESET_ALL}")
        self.status = "stopping"
        self.proc.terminate()


def discover_url_files(pattern: str = "*.txt") -> list[Path]:
    return sorted(URL_DIR.glob(pattern))


def build_jobs(existing: dict[str, SpiderJob]) -> dict[str, SpiderJob]:
    jobs = dict(existing)
    existing_bases = set(jobs.keys())

    url_files = discover_url_files("*.txt")
    next_color_index = len(existing_bases) % len(SPIDER_COLORS)

    for path in url_files:
        base = path.stem
        if base in existing_bases:
            continue
        color = SPIDER_COLORS[next_color_index % len(SPIDER_COLORS)]
        next_color_index += 1
        jobs[base] = SpiderJob(path, color)

    return jobs


def print_help():
    print(
        f"""\nCommands:
  help, ?           Show this help
  list              List spiders (name, status, URL file)
  start <name>      Start spider by URL file stem (e.g. 'test_urls')
  startall          Start all spiders
  stop <name>       Stop a running spider
  stopall           Stop all running spiders
  resume <name>     Same as 'start' (will resume via JOBDIR)
  addurl <URL>      Create a new urls/manual_*.txt containing URL and start its spider
  reload            Rescan urls/ folder for new *.txt URL files
  quit, exit        Stop all spiders and exit

Notes:
  • 'name' is the stem of the URL file, e.g. urls/test_urls.txt → name 'test_urls'
  • JOBDIR is stable per URL file, so 'start'/'resume' will pick up where it left off.
"""
    )


def main():
    jobs: dict[str, SpiderJob] = build_jobs({})
    if not jobs:
        print(
            f"{Fore.YELLOW}No URL files found in {URL_DIR} (expected *.txt)."
            f"{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}Create something like {URL_DIR / 'test_urls.txt'} "
            f"with one forum URL per line, then run again.{Style.RESET_ALL}"
        )

    print(
        f"{Fore.MAGENTA}Found {len(jobs)} URL file(s) in {URL_DIR.name}: "
        f"{', '.join(j.url_path.name for j in jobs.values())}{Style.RESET_ALL}"
    )

    print_help()

    while True:
        try:
            cmd_line = input(f"{Fore.WHITE}cmd>{Style.RESET_ALL} ").strip()
        except KeyboardInterrupt:
            print(
                f"\n{Fore.YELLOW}Use 'stopall' to stop spiders, or 'quit' to exit.{Style.RESET_ALL}"
            )
            continue

        if not cmd_line:
            continue

        parts = cmd_line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("help", "?"):
            print_help()

        elif cmd == "list":
            if not jobs:
                print("No spiders configured.")
                continue
            for name, job in jobs.items():
                status = job.status
                running = "running" if job.is_running() else status
                print(
                    f"{job.color}{name:<20}{Style.RESET_ALL}"
                    f" status={running:<9} file={job.url_path.relative_to(BASE_DIR)}"
                )

        elif cmd == "startall":
            if not jobs:
                print("No spiders to start.")
                continue
            for job in jobs.values():
                job.start()

        elif cmd == "start":
            if not args:
                print("Usage: start <name>")
                continue
            name = args[0]
            job = jobs.get(name)
            if not job:
                print(f"No spider with name '{name}'. Use 'list' to see names.")
                continue
            job.start()

        elif cmd == "resume":
            if not args:
                print("Usage: resume <name>")
                continue
            name = args[0]
            job = jobs.get(name)
            if not job:
                print(f"No spider with name '{name}'. Use 'list' to see names.")
                continue
            job.start()

        elif cmd == "stop":
            if not args:
                print("Usage: stop <name>")
                continue
            name = args[0]
            job = jobs.get(name)
            if not job:
                print(f"No spider with name '{name}'. Use 'list' to see names.")
                continue
            job.stop()

        elif cmd == "stopall":
            if not jobs:
                print("No spiders to stop.")
                continue
            for job in jobs.values():
                job.stop()

        elif cmd == "addurl":
            if not args:
                print("Usage: addurl <URL>")
                continue
            url = " ".join(args).strip()
            if not (url.startswith("http://") or url.startswith("https://")):
                print("URL should start with http:// or https://")
                continue

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f"manual_{ts}.txt"
            path = URL_DIR / file_name
            with path.open("w", encoding="utf-8") as f:
                f.write(url + "\n")

            color = SPIDER_COLORS[len(jobs) % len(SPIDER_COLORS)]
            job = SpiderJob(path, color)
            jobs[job.base] = job

            print(
                f"Created {path.relative_to(BASE_DIR)} with URL, starting spider '{job.base}'."
            )
            job.start()

        elif cmd == "reload":
            jobs = build_jobs(jobs)
            print(
                f"Reloaded URL files. Now tracking {len(jobs)} spider(s): "
                f"{', '.join(jobs.keys())}"
            )

        elif cmd in ("quit", "exit"):
            print("Stopping all spiders and exiting...")
            for job in jobs.values():
                job.stop()
            for job in jobs.values():
                if job.thread and job.thread.is_alive():
                    job.thread.join(timeout=1.0)
            break

        else:
            print(f"Unknown command '{cmd}'. Type 'help' for options.")


if __name__ == "__main__":
    main()
