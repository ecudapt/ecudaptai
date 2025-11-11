#!/usr/bin/env python3
"""
forum_scraper.py

Async web scraper using pyppeteer + stealth that is adapted for the ECUDAPT-AI project layout.

Default behavior:
  - Reads URLs from PROJECT_ROOT/data/raw/urls.txt
  - Writes JSONL to PROJECT_ROOT/data/raw/forum_posts_raw.jsonl

Usage examples:
  python src/forum_scraper.py
  python src/forum_scraper.py --urls data/raw/urls.txt --output data/raw/forum_posts_raw.jsonl --concurrency 4
"""

import asyncio
import json
import random
import argparse
from pathlib import Path
from typing import Optional
import aiofiles
from user_agents import parse as parse_ua
import logging
from pyppeteer import launch
from pyppeteer_stealth import stealth

# --- Detect project root and default paths ---
THIS_FILE = Path(__file__).resolve()
# We assume this file lives in PROJECT_ROOT/src/; project root is parent of src
PROJECT_ROOT = THIS_FILE.parents[1]
DEFAULT_URLS = PROJECT_ROOT / "data" / "raw" / "urls.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "forum_posts_raw.jsonl"
DEFAULT_SCREENSHOT_DIR = PROJECT_ROOT / "data" / "raw" / "screenshots"
logging.getLogger('pyppeteer.connection').setLevel(logging.ERROR)

# --- Small UA rotation (extend as needed) ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]


async def random_sleep(min_s=0.4, max_s=1.2):
    await asyncio.sleep(random.uniform(min_s, max_s))


async def _apply_basic_stealth(page, ua: str):
    """
    Apply stealth plugin + a few early navigator overrides.
    """
    await stealth(page)

    # Some lightweight navigator overrides
    await page.evaluateOnNewDocument(
        """() => {
            // webdriver
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
            // languages
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            // plugins stub
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4] });
        }"""
    )
    await page.setUserAgent(ua)


async def scrape_one(
    url: str,
    browser,
    selector: Optional[str],
    timeout: int,
    take_screenshot: bool,
    max_wait: int,
    screenshot_dir: Optional[Path],
):
    page = await browser.newPage()
    ua = random.choice(USER_AGENTS)
    await _apply_basic_stealth(page, ua)

    # realistic viewport
    await page.setViewport({"width": random.choice([1200, 1366, 1440]), "height": random.choice([720, 800, 900])})

    result = {"url": url, "status": None, "error": None, "content": None, "selector": selector or None, "ua": ua}

    try:
        resp = await page.goto(url, {"timeout": timeout, "waitUntil": "networkidle2"})
        if resp:
            result["status"] = resp.status
        # wait for selector (optional)
        if selector:
            try:
                await page.waitForSelector(selector, {"timeout": max_wait})
            except Exception:
                # selector didn't appear; we'll fall back
                pass

        if selector:
            try:
                elements = await page.querySelectorAll(selector)
                texts = []
                for el in elements:
                    txt = await page.evaluate("(e) => e.innerText", el)
                    texts.append(txt)
                result["content"] = texts
            except Exception as ex:
                result["error"] = f"selector-extract-error: {ex}"
        else:
            html = await page.content()
            result["content"] = html

        if take_screenshot and screenshot_dir:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            safe_name = url.replace("://", "_").replace("/", "_")[:120]
            out_path = screenshot_dir / f"{safe_name}.png"
            await page.screenshot({"path": str(out_path), "fullPage": True})
            result["screenshot"] = str(out_path)

    except Exception as e:
        result["error"] = repr(e)
    finally:
        try:
            await page.close()
        except Exception:
            pass

    return result


async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    browser,
    out_fpath: Path,
    selector: Optional[str],
    timeout: int,
    retries: int,
    take_screenshot: bool,
    max_wait: int,
    screenshot_dir: Optional[Path],
):
    # open file in append mode per worker (keeps things simple)
    async with aiofiles.open(out_fpath, "a", encoding="utf-8") as afp:
        while True:
            try:
                url = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            attempt = 0
            res = None
            while attempt <= retries:
                attempt += 1
                try:
                    res = await scrape_one(url, browser, selector, timeout, take_screenshot, max_wait, screenshot_dir)
                    break
                except Exception as e:
                    res = {"url": url, "status": None, "error": f"worker-exception:{e}", "content": None}
                    await asyncio.sleep(1 + attempt * 0.5)
            await afp.write(json.dumps(res, ensure_ascii=False) + "\n")
            await afp.flush()
            await random_sleep(0.2, 0.8)
            queue.task_done()


async def main(
    urls_file: Path,
    output_file: Path,
    concurrency: int,
    headless: bool,
    selector: Optional[str],
    timeout: int,
    retries: int,
    screenshot_dir: Optional[Path],
    max_wait: int,
):
    # Ensure directories exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if screenshot_dir:
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Read URL list
    if not urls_file.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}\nCreate it with one URL per line (use '#' for comments).")
    raw = urls_file.read_text(encoding="utf-8").strip().splitlines()
    urls = [u.strip() for u in raw if u.strip() and not u.strip().startswith("#")]
    if not urls:
        raise ValueError(f"No URLs found in {urls_file}")

    # prepare queue
    queue = asyncio.Queue()
    for u in urls:
        queue.put_nowait(u)

    launch_args = {
        "headless": headless,
        "executablePath": r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # <-- your Chrome path here
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-blink-features=AutomationControlled",
        ],
        "handleSIGINT": False,
        "handleSIGTERM": False,
    }

    browser = await launch(**launch_args)

    # start workers
    workers = [
        asyncio.create_task(
            worker(i, queue, browser, output_file, selector, timeout, retries, bool(screenshot_dir), max_wait, screenshot_dir)
        )
        for i in range(concurrency)
    ]

    await asyncio.gather(*workers)
    await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECUDAPT-AI adapted pyppeteer stealth scraper")
    parser.add_argument("--urls", type=Path, default=DEFAULT_URLS, help=f"Text file with one URL per line (default: {DEFAULT_URLS})")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help=f"JSONL output (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--concurrency", "-c", type=int, default=3, help="Concurrent pages")
    parser.add_argument("--headless", action="store_true", help="Run headless (default False)")
    parser.add_argument("--selector", "-s", type=str, default=None, help="CSS selector to extract innerText from")
    parser.add_argument("--timeout", type=int, default=30000, help="Navigation timeout in ms")
    parser.add_argument("--retries", type=int, default=1, help="Retries per URL on failure")
    parser.add_argument("--screenshot-dir", type=Path, default=None, help="Directory to save screenshots (optional)")
    parser.add_argument("--max-wait", type=int, default=5000, help="ms to wait for selector before falling back")

    args = parser.parse_args()

    # create or clear output file if it doesn't exist; if it exists, append by default
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text("", encoding="utf-8")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading URLs from: {args.urls}")
    print(f"Writing results to: {args.output}")
    if args.screenshot_dir:
        print(f"Saving screenshots to: {args.screenshot_dir}")

    asyncio.get_event_loop().run_until_complete(
        main(
            urls_file=args.urls,
            output_file=args.output,
            concurrency=args.concurrency,
            headless=args.headless,
            selector=args.selector,
            timeout=args.timeout,
            retries=args.retries,
            screenshot_dir=args.screenshot_dir,
            max_wait=args.max_wait,
        )
    )
    print("Done.")
