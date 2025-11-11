#!/usr/bin/env python3
"""
forums_hybrid_seq.py

Final Sequential ECU Forum Crawler
----------------------------------
✅ One domain crawled at a time
✅ Colorized live console output
✅ Playwright support for JS-heavy sites
✅ Per-domain summary table at end
"""
import scrapy, re, asyncio
from urllib.parse import urljoin, urlparse
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style, init
from scrapy_playwright.page import PageMethod

init(autoreset=True)

# ---------------- Keywords & Config ----------------
INCLUDE_KEYWORDS = [
    "ecu","tune","tuning","flash","remap","map","boost","turbo",
    "intake","exhaust","reflash","chip","performance",
    "power","hp","torque","stage","efi","cobb","hondata",
    "mhd","link","haltech","megasquirt","sensor","afr","engine"
]
EXCLUDE_KEYWORDS = [
    "wheels","tires","audio","off-topic","classified","sale",
    "suspension","meet","body","paint","detailing","show"
]
THREAD_PATTERNS = re.compile(
    r"(showthread\.php|viewtopic\.php|/threads?/|/topic/|/discussion/|-t\d{2,6}|"
    r"\.\d{2,6}/|/forums/[^/]+\.\d+/)", re.I
)
ENGINE_HINTS = {
    "ecuedit.com":"vbulletin","eectuning.org":"phpbb","hpacademy.com":"discourse",
    "db-tuned.com":"xenforo","tristatetuners.com":"vbulletin","n54tech.com":"vbulletin",
    "hondata.com":"phpbb","bimmerboost.com":"vbulletin","audizine.com":"vbulletin",
    "ft86club.com":"xenforo","turbobricks.com":"vbulletin","explorerst.org":"xenforo",
    "golfmk6.com":"vbulletin","golfmk7.com":"vbulletin","evolutionm.net":"vbulletin"
}
PLAYWRIGHT_SITES = [
    "hondata.com","bimmerboost.com","hpacademy.com","audizine.com","ft86club.com"
]


class ForumSpider(scrapy.Spider):
    name = "forums_hybrid"

    custom_settings = {
        "FEEDS": {
            f"data/raw/forum_posts_{datetime.now():%Y%m%d_%H%M%S}.jsonl": {
                "format": "jsonlines",
                "encoding": "utf8",
                "overwrite": True,
            }
        },
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

    def __init__(self, urls_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls_file = urls_file
        self.start_urls = []
        self.summary = {}
        self.seen_threads = set()
        if urls_file:
            path = Path(urls_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.start_urls = [u.strip() for u in f if u.strip()]
                self.logger.info(f"{Fore.CYAN}📂 Loaded {len(self.start_urls)} start URLs from {path}{Style.RESET_ALL}")
            else:
                self.logger.error(f"{Fore.RED}❌ URLs file not found: {path}{Style.RESET_ALL}")
        else:
            self.logger.error(f"{Fore.RED}❌ Missing urls_file argument.{Style.RESET_ALL}")

    # -----------------------------------------------------
    # Sequential start (Scrapy 2.13+)
    # -----------------------------------------------------
    async def start(self):
        """Run each forum sequentially before moving to the next."""
        self.logger.info(f"{Fore.MAGENTA}🚀 Sequential crawl of {len(self.start_urls)} forums begins.{Style.RESET_ALL}")

        for i, url in enumerate(self.start_urls, 1):
            domain = urlparse(url).netloc
            use_pw = any(d in url for d in PLAYWRIGHT_SITES)
            self.logger.info(f"\n{Fore.BLUE}🌍 [{i}/{len(self.start_urls)}] {domain}{Style.RESET_ALL}")
            self.seen_threads.clear()
            self.summary[domain] = {"threads": 0, "posts": 0, "status": "⚠️"}

            req = scrapy.Request(
                url=url,
                callback=self.parse_forum,
                meta={
                    "playwright": use_pw,
                    "playwright_include_page": use_pw,
                    "playwright_page_methods": [PageMethod("wait_for_timeout", 1500)],
                    "forum_root": url,
                    "page": 1,
                    "domain": domain,
                },
                dont_filter=True,
            )
            yield req

            # Wait until crawler queue drains
            while len(self.crawler.engine.slot.inprogress) > 0:
                await asyncio.sleep(0.5)
            await asyncio.sleep(1.5)

        # Print domain summary
        print(f"\n{Fore.MAGENTA}🧾 Crawl Summary:{Style.RESET_ALL}")
        print(f"{'Domain':<25} {'Threads':>8} {'Posts':>8} {'Status':>8}")
        print("-"*50)
        for d, v in self.summary.items():
            color = Fore.GREEN if v["status"] == "✅" else (Fore.YELLOW if v["status"] == "⚠️" else Fore.RED)
            print(f"{color}{d:<25} {v['threads']:>8} {v['posts']:>8} {v['status']:>8}{Style.RESET_ALL}")

    # -----------------------------------------------------
    # Forum parser
    # -----------------------------------------------------
    def parse_forum(self, response):
        domain = response.meta.get("domain")
        engine = ENGINE_HINTS.get(domain, "generic")
        page = response.meta.get("page", 1)
        self.logger.info(f"{Fore.CYAN}📖 Parsing {domain} page {page}{Style.RESET_ALL}")

        if response.status != 200:
            self.summary[domain]["status"] = "❌"
            self.logger.warning(f"{Fore.RED}⚠️ HTTP {response.status} on {domain}{Style.RESET_ALL}")
            return

        links = response.css("a::attr(href)").getall()
        thread_links = []

        for href in links:
            if not href or not THREAD_PATTERNS.search(href):
                continue
            full = urljoin(response.url, href)
            norm = full.split("#")[0].rstrip("/")
            if norm in self.seen_threads:
                continue

            anchor_text = (
                " ".join(response.css(f"a[href='{href}']::attr(title)").getall()) + " " +
                " ".join(response.css(f"a[href='{href}']::text").getall())
            ).lower()

            if not self.is_relevant(anchor_text + " " + href):
                continue

            thread_links.append(norm)
            self.seen_threads.add(norm)

        self.summary[domain]["threads"] += len(thread_links)

        if thread_links:
            self.logger.info(f"{Fore.GREEN}✅ {len(thread_links)} threads on {domain} (page {page}){Style.RESET_ALL}")
        else:
            self.logger.info(f"{Fore.YELLOW}😴 No relevant threads on {domain} (page {page}){Style.RESET_ALL}")

        for link in thread_links:
            yield scrapy.Request(
                link,
                callback=self.parse_thread,
                meta={"forum_domain": domain, "engine": engine},
                dont_filter=True,
            )

        next_page = response.css("a[rel='next']::attr(href), a.next::attr(href)").get()
        if not next_page:
            for a in links:
                if re.search(r"(page[-=]\d+|start=\d+)", a) and str(page + 1) in a:
                    next_page = urljoin(response.url, a)
                    break

        if next_page:
            yield scrapy.Request(
                next_page,
                callback=self.parse_forum,
                meta={"domain": domain, "page": page + 1},
                dont_filter=True,
            )
        else:
            self.logger.info(f"{Fore.GREEN}🏁 Finished crawling {domain}{Style.RESET_ALL}")
            if self.summary[domain]["threads"] > 0:
                self.summary[domain]["status"] = "✅"

    # -----------------------------------------------------
    # Thread parser
    # -----------------------------------------------------
    def parse_thread(self, response):
        domain = response.meta.get("forum_domain")
        title = (response.css("title::text").get() or "").strip()
        posts = []
        for sel in ["div.post","div.message","article","div.postbody","li.message",".bbWrapper"]:
            for e in response.css(sel):
                text = " ".join(e.css("::text").getall()).strip()
                if len(text) < 30:
                    continue
                author = (e.css(".username::text, .author::text, .user::text").get() or "unknown").strip()
                date = (
                    e.css("time::attr(datetime)").get() or
                    e.css(".date::text, .postdate::text").get() or ""
                ).strip()
                posts.append({"author": author, "date": date, "content": re.sub(r"\s+", " ", text)})
            if posts:
                break

        count = len(posts)
        self.summary[domain]["posts"] += count

        if count:
            self.logger.info(f"{Fore.GREEN}💬 {count} posts parsed from {domain}{Style.RESET_ALL}")
        else:
            self.logger.info(f"{Fore.YELLOW}📭 0 posts parsed from {domain}{Style.RESET_ALL}")

        yield {
            "url": response.url,
            "domain": domain,
            "title": title,
            "num_posts": count,
            "posts": posts,
        }

    # -----------------------------------------------------
    # Keyword filter
    # -----------------------------------------------------
    def is_relevant(self, text: str):
        t = text.lower()
        if any(bad in t for bad in EXCLUDE_KEYWORDS):
            return False
        return any(k in t for k in INCLUDE_KEYWORDS)
