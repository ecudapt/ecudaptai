BOT_NAME = "ecudapt"

SPIDER_MODULES = ["ecudapt.spiders"]
NEWSPIDER_MODULE = "ecudapt.spiders"

# Crawl responsibly
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"
ROBOTSTXT_OBEY = False

# Concurrency
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 4
DOWNLOAD_DELAY = 0.25
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 5.0

# Output
FEEDS = {
    "data/raw/forum_posts_raw.jsonl": {"format": "jsonlines"},
}

LOG_LEVEL = "INFO"

# Enable Playwright for hybrid spider
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {"headless": True}
