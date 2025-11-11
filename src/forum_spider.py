import scrapy, re, os
from urllib.parse import urljoin

INCLUDE_KEYWORDS = [
    "ecu", "tune", "tuning", "flash", "remap", "chip", "engine", "boost", "ignition",
    "fuel", "air", "knock", "turbo", "stage", "hp", "torque", "efi", "map", "mhd",
    "cobb", "hondata", "megasquirt", "link", "haltech", "obd", "sensor", "injector", "afr"
]
EXCLUDE_KEYWORDS = [
    "off", "topic", "wheels", "tires", "audio", "classified", "sale",
    "suspension", "body", "paint", "detailing", "events"
]

THREAD_PATTERNS = re.compile(
    r"(showthread\.php|viewtopic\.php|/thread|/topic|/threads?/|/t/|/posts/|/discussion/)",
    re.IGNORECASE,
)
FORUM_PATTERNS = re.compile(
    r"(forumdisplay\.php|viewforum\.php|/forums?/|/categories?/|/f/|/c/)",
    re.IGNORECASE,
)


class ForumSpider(scrapy.Spider):
    name = "forums_hybrid"

    custom_settings = {
        "FEEDS": {"data/raw/forum_posts_raw.jsonl": {"format": "jsonlines"}},
        "LOG_LEVEL": "INFO",
        "DOWNLOAD_DELAY": 0.5,
        "CONCURRENT_REQUESTS": 8,
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {"headless": True},
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36",
    }

    def start_requests(self):
        urls_file = "data/raw/urls.txt"
        if not os.path.exists(urls_file):
            self.logger.error("❌ Missing data/raw/urls.txt")
            return
        with open(urls_file) as f:
            urls = [u.strip() for u in f if u.strip()]
        self.logger.info(f"🚀 Starting crawl for {len(urls)} forums")

        for i, url in enumerate(urls, 1):
            use_playwright = any(
                x in url.lower()
                for x in ["hpacademy", "ft86club", "nissanzclub", "rx7club", "clublexus"]
            )
            self.logger.info(f"\n🌐 [{i}/{len(urls)}] Visiting {url} (playwright={use_playwright})")
            yield scrapy.Request(
                url,
                callback=self.parse_forum,
                meta={"forum_url": url, "depth": 0, "playwright": use_playwright},
            )

    # ---------- Keyword relevance ----------
    def is_relevant(self, text):
        t = text.lower()
        if any(bad in t for bad in EXCLUDE_KEYWORDS):
            return False
        return any(k in t for k in INCLUDE_KEYWORDS)

    # ---------- Forum parsing ----------
    async def parse_forum(self, response):
        forum = response.meta["forum_url"]
        depth = response.meta.get("depth", 0)
        if response.status >= 400:
            self.logger.warning(f"⚠️ Skipping {forum} — HTTP {response.status}")
            return

        links = response.css("a::attr(href)").getall()
        threads, subforums = set(), set()

        for href in links:
            full = urljoin(response.url, href)
            text = " ".join(response.css(f"a[href='{href}']::text").getall())
            if THREAD_PATTERNS.search(href):
                if self.is_relevant(text + href):
                    threads.add(full)
            elif depth < 2 and FORUM_PATTERNS.search(href):
                subforums.add(full)

        if threads:
            self.logger.info(f"✅ Found {len(threads)} ECU-related threads on {forum}")
        else:
            self.logger.info(f"😴 No relevant threads on {forum} ({len(links)} links)")

        # Parse threads
        for idx, link in enumerate(list(threads)[:60]):
            yield scrapy.Request(
                link,
                callback=self.parse_thread,
                meta={"forum_url": forum, "thread_num": idx + 1, "total_threads": len(threads), "playwright": False},
            )

        # Recurse subforums
        for f_url in list(subforums)[:10]:
            yield scrapy.Request(
                f_url,
                callback=self.parse_forum,
                meta={"forum_url": forum, "depth": depth + 1, "playwright": response.meta.get("playwright", False)},
            )

        # Pagination detection (supports "Next", icons, and rel attributes)
        next_page = response.css(
            "a[rel='next']::attr(href), a.next::attr(href), a:contains('Next')::attr(href), a:contains('›')::attr(href)"
        ).get()
        if next_page:
            next_url = urljoin(response.url, next_page)
            self.logger.info(f"➡️  Next page for {forum}: {next_url}")
            yield scrapy.Request(
                next_url,
                callback=self.parse_forum,
                meta={"forum_url": forum, "depth": depth, "playwright": response.meta.get("playwright", False)},
            )

    # ---------- Thread parsing ----------
    async def parse_thread(self, response):
        forum = response.meta.get("forum_url")
        idx = response.meta.get("thread_num", 0)
        total = response.meta.get("total_threads", "?")
        title = response.css("title::text").get() or "Untitled"
        posts = []

        post_blocks = response.css(
            "div.post, div.postbody, article.message, div.message-content, div.message, div.messageText"
        )
        for post in post_blocks:
            author = post.css(".username::text, .author::text, a.user::text").get(default="Unknown").strip()
            content = " ".join(post.css("::text").getall()).strip()
            if len(content) < 40:
                continue
            posts.append({"author": author, "content": content})

        if posts:
            self.logger.info(f"💬 [{forum}] ({idx}/{total}) '{title[:50]}' → {len(posts)} posts")
            yield {"url": response.url, "title": title.strip(), "posts": posts}
        else:
            self.logger.info(f"⚪ [{forum}] ({idx}/{total}) '{title[:50]}' → no valid posts found")
