#!/usr/bin/env python3
"""
forum_spider_hybrid.py

ECU Forum Crawler (Forum Pages Only, Per-Spider Output, Minimal Logging)
-----------------------------------------------------------------------
✅ Starts from forum SECTION URLs (like /forums/performance-tuning.13/)
✅ Walks ONLY the section's pagination (page 1, page 2, page 3, ...)
✅ From each forum page, finds real thread URLs and scrapes posts
✅ Canonicalizes thread URLs so the same thread isn't queued multiple times
✅ Each spider instance writes to its own JSONL file based on urls_file:
       data/raw/forums-<urls_stem>-<YYYYMMDD-HHMMSS>.jsonl
✅ Skips member/profile/etc. URLs
✅ Threads with 0 posts are NOT stored
✅ Console logging:
   • when a spider moves to a new site
   • when a thread with posts is appended to the output
   • when a site is finished (no more forum pages)
"""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode

import scrapy
from colorama import Fore, Style, init
from scrapy_playwright.page import PageMethod

init(autoreset=True)

# ---------------- Keywords & Config ----------------
INCLUDE_KEYWORDS = [
    "ecu", "tune", "tuning", "flash", "remap", "map", "boost", "turbo",
    "intake", "exhaust", "reflash", "chip", "performance",
    "power", "hp", "torque", "stage", "efi", "cobb", "hondata",
    "mhd", "link", "haltech", "megasquirt", "sensor", "afr", "engine",
]

EXCLUDE_KEYWORDS = [
    "wheels", "tires", "audio", "off-topic", "classified", "sale",
    "suspension", "meet", "body", "paint", "detailing", "show",
]

# Known JS-heavy forums that benefit from Playwright
PLAYWRIGHT_SITES = [
    "hondata.com", "bimmerboost.com", "hpacademy.com",
    "audizine.com", "ft86club.com",
]

# Domains where we allow looser thread detection (their URL patterns are weird)
LOOSE_THREAD_DOMAINS = {
    "hpacademy.com",
    "www.hpacademy.com",
    "jb4tech.com",
    "www.jb4tech.com",
}


class ForumSpider(scrapy.Spider):
    name = "forums_hybrid"

    # FEEDS is set dynamically in from_crawler
    custom_settings = {
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_MAX_DELAY": 5,
        "DOWNLOAD_DELAY": 0.5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "CONCURRENT_REQUESTS": 1,
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }

    # --------------------------------------------------
    # Per-spider FEEDS config based on urls_file
    # --------------------------------------------------
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)

        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        if getattr(spider, "urls_file", None):
            base_name = Path(spider.urls_file).stem
        else:
            base_name = "forums"

        # New, clearer filename: forums-<urls_stem>-<YYYYMMDD-HHMMSS>.jsonl
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        feed_path = data_dir / f"forums-{base_name}-{ts}.jsonl"

        crawler.settings.set(
            "FEEDS",
            {
                str(feed_path): {
                    "format": "jsonlines",
                    "encoding": "utf8",
                    "overwrite": True,
                }
            },
            priority="spider",
        )

        return spider

    # --------------------------------------------------
    # Init
    # --------------------------------------------------
    def __init__(self, urls_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.urls_file = urls_file  # used by from_crawler

        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.start_urls = []
        self.summary = {}                 # domain -> {threads, posts, status}
        self.seen_threads = set()         # canonical thread URLs
        self.visited_forum_pages = set()  # normalized forum-section pages
        self.domain_pages = defaultdict(int)  # domain -> number of forum pages crawled

        if urls_file:
            path = Path(urls_file)
            if not path.exists():
                path = base_dir / path
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.start_urls = [u.strip() for u in f if u.strip()]
            else:
                self.logger.error(f"URLs file not found: {path}")
        else:
            self.logger.error("Missing urls_file argument.")

    # --------------------------------------------------
    # Scrapy 2.13+ entrypoint (no start_requests deprecation)
    # --------------------------------------------------
    async def start(self):
        """Entry point for Scrapy 2.13+. Replaces deprecated start_requests()."""
        total = len(self.start_urls)
        for i, url in enumerate(self.start_urls, 1):
            domain = urlparse(url).netloc
            use_pw = any(d in domain for d in PLAYWRIGHT_SITES)

            self.summary.setdefault(domain, {"threads": 0, "posts": 0, "status": "⚠️"})

            # Site-start log
            self.logger.warning(
                f"{Fore.BLUE}🌍 [{i}/{total}] Starting site: {domain}{Style.RESET_ALL}"
            )

            yield scrapy.Request(
                url=url,
                callback=self.parse_forum_page,
                meta={
                    "playwright": use_pw,
                    "playwright_include_page": use_pw,
                    "playwright_page_methods": [PageMethod("wait_for_timeout", 1500)],
                    "domain": domain,
                    "forum_root": url,
                    "page": 1,
                },
                dont_filter=True,
            )

    # --------------------------------------------------
    # Helper: canonicalize a thread URL so duplicates collapse
    # --------------------------------------------------
    def canonical_thread_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or ""
        query = parsed.query or ""

        # phpBB / vBulletin-style
        if "viewtopic.php" in path or "showthread.php" in path:
            params = dict(parse_qsl(query))
            keep = {}
            if "t" in params:
                keep["t"] = params["t"]
            if "f" in params:
                keep["f"] = params["f"]
            new_query = urlencode(keep)
            return parsed._replace(query=new_query, fragment="").geturl()

        # XenForo / similar: /threads/title.12345/...
        if "/threads/" in path:
            path = re.sub(r"/(page-\d+|latest|post-\d+[^/]*)/?$", "", path)
            return parsed._replace(path=path, query="", fragment="").geturl()

        # Discourse: /t/slug/12345/..., canonicalize to /t/slug/12345
        if re.search(r"/t/[^/]+/\d+/?", path):
            m = re.search(r"(/t/[^/]+/\d+)", path)
            path = m.group(1) if m else path
            return parsed._replace(path=path, query="", fragment="").geturl()

        # Generic fallback: strip query + fragment
        return parsed._replace(query="", fragment="").geturl()

    # --------------------------------------------------
    # Helper: is this URL a real thread URL?
    # --------------------------------------------------
    def is_thread_url(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()

        # Hard excludes – clearly not threads
        if any(seg in path for seg in [
            "/members", "/member", "/user", "/users", "/profile",
            "/login", "/register", "/account", "/attachments",
            "/conversations", "/inbox", "/help", "/about", "/contact",
        ]):
            return False

        # Positive patterns (generic)
        if "/threads/" in path:
            return True
        if "showthread.php" in path:
            return True
        if "viewtopic.php" in path:
            return True
        if "/topic/" in path:
            return True
        if "/discussion/" in path:
            return True
        # Discourse-style: /t/slug/12345
        if re.search(r"/t/[^/]+/\d+/?", path):
            return True

        # phpBB-style with t= in query
        if ("t=" in query) and ("viewtopic" in path or "showthread" in path):
            return True

        # Looser detection for tricky domains (hpacademy, jb4tech)
        if domain in LOOSE_THREAD_DOMAINS:
            if "/forum" in path and not any(seg in path for seg in [
                "/members", "/user", "/users", "/profile", "/login", "/register"
            ]):
                return True

        return False

    # --------------------------------------------------
    # Helper: keyword filter
    # --------------------------------------------------
    def is_relevant(self, text: str) -> bool:
        t = text.lower()
        if any(bad in t for bad in EXCLUDE_KEYWORDS):
            return False
        return any(k in t for k in INCLUDE_KEYWORDS)

    # --------------------------------------------------
    # Forum section page parser (pagination only)
    # --------------------------------------------------
    def parse_forum_page(self, response):
        domain = response.meta["domain"]
        page_num = response.meta.get("page", 1)
        forum_root = response.meta.get("forum_root", response.url)

        norm_url = response.url.split("#")[0].rstrip("/")
        if norm_url in self.visited_forum_pages:
            return
        self.visited_forum_pages.add(norm_url)
        self.domain_pages[domain] += 1

        if response.status != 200:
            self.summary[domain]["status"] = "❌"
            return

        links = response.css("a::attr(href)").getall()
        thread_links = []

        for href in links:
            if not href:
                continue

            full = urljoin(response.url, href)
            norm = full.split("#")[0].rstrip("/")

            parsed = urlparse(full)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != urlparse(response.url).netloc:
                continue

            if not self.is_thread_url(norm):
                continue

            canon = self.canonical_thread_url(norm)
            if canon in self.seen_threads:
                continue

            anchor_text = (
                " ".join(response.css(f"a[href='{href}']::attr(title)").getall())
                + " "
                + " ".join(response.css(f"a[href='{href}']::text").getall())
            ).lower()

            if not self.is_relevant(anchor_text + " " + href):
                continue

            self.seen_threads.add(canon)
            thread_links.append(canon)

        self.summary[domain]["threads"] += len(thread_links)

        # Queue thread pages (canonical URLs)
        for link in thread_links:
            yield scrapy.Request(
                link,
                callback=self.parse_thread,
                meta={"forum_domain": domain},
                dont_filter=True,
            )

        # -------------------------------
        # Forum pagination ONLY
        # -------------------------------
        next_href = response.css(
            "a[rel='next']::attr(href), "
            "a.next::attr(href), "
            "a.pagination-next::attr(href), "
            "a.pageNav-jump--next::attr(href), "
            "a[aria-label='Next']::attr(href)"
        ).get()

        if not next_href:
            for href in links:
                if not href:
                    continue
                if re.search(r"(?:\?|&|/)(page[-=/]?%d)\b" % (page_num + 1), href, re.I):
                    next_href = href
                    break
                if re.search(r"(?:\?|&)start=\d+", href, re.I) and str(page_num * 20) in href:
                    next_href = href
                    break

        if next_href:
            full_next = urljoin(response.url, next_href)
            next_norm = full_next.split("#")[0].rstrip("/")

            root_path = urlparse(forum_root).path.rstrip("/")
            next_path = urlparse(full_next).path.rstrip("/")
            if not next_path.startswith(root_path):
                return
            if next_norm in self.visited_forum_pages:
                return

            yield scrapy.Request(
                full_next,
                callback=self.parse_forum_page,
                meta={
                    "domain": domain,
                    "forum_root": forum_root,
                    "page": page_num + 1,
                },
                dont_filter=True,
            )
        else:
            # Domain done – log once
            self.summary[domain]["status"] = "✅"
            self.logger.warning(
                f"{Fore.MAGENTA}[DONE]{Style.RESET_ALL} {domain} | "
                f"forum_pages={self.domain_pages[domain]} | "
                f"threads={self.summary[domain]['threads']} | "
                f"posts={self.summary[domain]['posts']}"
            )

    # --------------------------------------------------
    # Thread parser (skip 0-post threads; log only when yielding)
    # --------------------------------------------------
    def parse_thread(self, response):
        domain = response.meta.get("forum_domain")
        title = (response.css("title::text").get() or "").strip()
        posts = []

        for sel in [
            "div.post",
            "div.message",
            "article",
            "div.postbody",
            "li.message",
            ".bbWrapper",
        ]:
            for e in response.css(sel):
                text = " ".join(e.css("::text").getall()).strip()
                if len(text) < 30:
                    continue

                author = (
                    e.css(".username::text, .author::text, .user::text").get()
                    or "unknown"
                ).strip()
                date = (
                    e.css("time::attr(datetime)").get()
                    or e.css(".date::text, .postdate::text").get()
                    or ""
                ).strip()

                posts.append(
                    {
                        "author": author,
                        "date": date,
                        "content": re.sub(r"\s+", " ", text),
                    }
                )

            if posts:
                break

        count = len(posts)

        # Threads with no posts should NOT be stored and not logged
        if count == 0:
            return

        self.summary[domain]["posts"] += count

        # Log only when we actually output a thread
        safe_title = title.replace("\n", " ").strip()
        if len(safe_title) > 80:
            safe_title = safe_title[:77] + "..."

        self.logger.warning(
            f"{Fore.GREEN}[THREAD]{Style.RESET_ALL} {domain} | {safe_title} | posts={count}"
        )

        yield {
            "url": response.url,
            "domain": domain,
            "title": title,
            "num_posts": count,
            "posts": posts,
        }
