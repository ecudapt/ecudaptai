#!/usr/bin/env python3
"""
forum_crawler_plus.py
Enhanced ECU-related forum crawler:
• Concurrent multi-forum crawling
• Pagination handling
• Per-forum progress & dedup
• Richer post metadata
• Cloudflare bypass with visible Chrome retry
• Cookie persistence per domain
"""

import asyncio, aiofiles, json, random, re, sys, traceback
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pyppeteer import launch
from pyppeteer_stealth import stealth

# ---------------- Configuration ----------------
INCLUDE = [
    "ecu","tune","tuning","flash","remap","chip","engine","boost","ignition",
    "fuel","air","knock","turbo","stage","hp","torque","efi","map","mhd","cobb",
    "hondata","megasquirt","link","haltech","obd","sensor","injector","afr"
]
EXCLUDE = [
    "off","topic","wheels","tires","audio","show","meet","classified","sale",
    "suspension","body","paint","detailing","events","appearance"
]
OUT_DIR = Path("data/raw/threads")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_FILE = OUT_DIR / "seen_threads.json"
COOKIE_DIR = Path("data/cookies")
COOKIE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/122 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) Chrome/122 Safari/537.36",
]

MAX_PAGES = 5
CONCURRENCY = 4

# ---------------- Utilities ----------------
def load_seen():
    if DEDUP_FILE.exists():
        return set(json.loads(DEDUP_FILE.read_text()))
    return set()

def save_seen(seen):
    DEDUP_FILE.write_text(json.dumps(list(seen), indent=2))

async def random_sleep(a=1.0,b=3.0):
    await asyncio.sleep(random.uniform(a,b))

def is_relevant(text, href):
    t = f"{text or ''} {href or ''}".lower()
    if any(x in t for x in EXCLUDE): return False
    return any(k in t for k in INCLUDE)

# ---------------- Extraction ----------------
THREAD_KEYWORDS = [
    "showthread", "/thread", "/threads", "/t/", "/topic", "viewtopic",
    "/discussion", "forumdisplay.php?f="
]

async def extract_links(page, base):
    anchors = await page.querySelectorAll("a[href]")
    links=set()
    for a in anchors:
        href = await page.evaluate("(a)=>a.getAttribute('href')", a)
        txt  = await page.evaluate("(a)=>a.innerText", a)
        if not href:
            continue
        if any(k in href for k in THREAD_KEYWORDS):
            full=urljoin(base,href)
            if urlparse(full).netloc!=urlparse(base).netloc:
                continue
            if is_relevant(txt,href):
                links.add(full.split("#")[0])
    return list(links)

def extract_next_page(soup, base):
    """Find next page link across various forum engines"""
    candidates = soup.select('a[href], nav a[href], li.pageNav-jump--next a')
    for c in candidates:
        text = (c.get_text() or "").lower().strip()
        if any(x in text for x in ["next", "›", "→"]):
            href = c.get("href")
            if href:
                return urljoin(base, href)
    rel_next = soup.find("a", rel="next")
    if rel_next and rel_next.get("href"):
        return urljoin(base, rel_next["href"])
    return None

# ---------------- Browser Management ----------------
async def launch_browser(domain, headless=True):
    """Launch Chrome and load cookies for a given domain"""
    launch_args = {
        "headless": headless,
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
        "executablePath": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    }
    browser = await launch(**launch_args)
    page = await browser.newPage()
    await stealth(page)
    await page.setViewport({'width': 1280, 'height': 900})
    await page.setUserAgent(random.choice(USER_AGENTS))
    await page.setExtraHTTPHeaders({'Accept-Language': 'en-US,en;q=0.9'})
    await page.evaluateOnNewDocument("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    """)

    cookie_file = COOKIE_DIR / f"{domain}.json"
    if cookie_file.exists():
        cookies = json.loads(cookie_file.read_text())
        await page.setCookie(*cookies)
    return browser, page

async def save_cookies(page, domain):
    cookies = await page.cookies()
    cookie_file = COOKIE_DIR / f"{domain}.json"
    cookie_file.write_text(json.dumps(cookies))

# ---------------- Scraping ----------------
async def scrape_thread(browser, url):
    page=await browser.newPage()
    await stealth(page)
    await page.setUserAgent(random.choice(USER_AGENTS))
    await page.setViewport({'width': 1280, 'height': 900})
    await page.setExtraHTTPHeaders({'Accept-Language': 'en-US,en;q=0.9'})
    await page.evaluateOnNewDocument("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    """)
    res={"thread_url":url,"title":None,"posts":[]}
    try:
        await page.goto(url, {"timeout":90000,"waitUntil":"networkidle2"})
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await asyncio.sleep(2)

        html=await page.content()
        soup=BeautifulSoup(html,"lxml")
        title=soup.find(["h1","title"])
        res["title"]=title.get_text(strip=True) if title else url

        posts=soup.select("div.post,div.postbody,article.message,div.message,div.message-content")
        for p in posts:
            text=" ".join(p.get_text(" ",strip=True).split())
            if len(text)<25: continue
            author=(p.find_previous(class_=re.compile("user|author",re.I))
                    or p.find_parent(class_=re.compile("user|author",re.I)))
            date=p.find(class_=re.compile("date|time",re.I))
            res["posts"].append({
                "author": author.get_text(strip=True) if author else "Unknown",
                "date": date.get_text(strip=True) if date else "",
                "content": text
            })
    except Exception as e:
        print(f"[!] Error {url}: {e}")
        traceback.print_exc()
    finally:
        await page.close()
    return res

# ---------------- Crawl Controller ----------------
async def crawl_forum(base_url, seen, limit=25):
    domain = urlparse(base_url).netloc
    print(f"\n🌐 Crawling {domain}")
    browser, page = await launch_browser(domain, headless=True)
    new_threads=set()
    cur_url=base_url
    cloudflare_triggered=False

    for i in range(MAX_PAGES):
        try:
            await page.goto(cur_url,{"waitUntil":"domcontentloaded","timeout":90000})
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            html=await page.content()

            # Detect Cloudflare challenge
            if "cf-error" in html or "Attention Required" in html:
                print(f"⚠️ Cloudflare challenge detected on {domain}, retrying non-headless...")
                cloudflare_triggered=True
                break

            soup=BeautifulSoup(html,"lxml")
            links=await extract_links(page,base_url)
            for l in links:
                if l not in seen:
                    new_threads.add(l)

            nextp=extract_next_page(soup,base_url)
            if not nextp:
                break
            cur_url=nextp
            await random_sleep()
        except Exception as e:
            print(f"Pagination stop {e}")
            break

    await save_cookies(page, domain)
    await page.close()
    await browser.close()

    # Retry in visible Chrome if Cloudflare blocked
    if cloudflare_triggered:
        browser, page = await launch_browser(domain, headless=False)
        await page.goto(base_url)
        print("⚠️ Please complete Cloudflare verification in Chrome window...")
        await asyncio.sleep(15)
        await save_cookies(page, domain)
        await browser.close()
        print("✅ Cookies saved, next run will continue headless.")
        return seen

    print(f"Found {len(new_threads)} new thread links")
    if not new_threads:
        print(f"⚠️ No threads found for {domain}")
        return seen

    browser, _ = await launch_browser(domain)
    sem=asyncio.Semaphore(CONCURRENCY)
    results=[]

    async def worker(link):
        async with sem:
            d=await scrape_thread(browser,link)
            if d["posts"]:
                results.append(d)
                seen.add(link)
                print(f"✅ {d['title']} ({len(d['posts'])} posts)")
            await random_sleep()

    await asyncio.gather(*(worker(l) for l in list(new_threads)[:limit]))
    await browser.close()

    if results:
        fpath=OUT_DIR/f"{domain.replace('.','_')}.jsonl"
        async with aiofiles.open(fpath,"a",encoding="utf-8") as f:
            for r in results:
                await f.write(json.dumps(r,ensure_ascii=False)+"\n")
        print(f"[✓] Saved {len(results)} → {fpath}")
    else:
        print(f"⚠️ No threads saved for {domain}")
    return seen

# ---------------- Entry ----------------
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--urls","-u",default="data/raw/urls.txt")
    parser.add_argument("--limit","-l",type=int,default=25)
    args=parser.parse_args()

    urls=[u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip()]
    seen=load_seen()

    async def main():
        for u in urls:
            seen.update(await crawl_forum(u,seen,args.limit))
            save_seen(seen)
            await random_sleep(3,7)
        print(f"\n✅ Done. Total seen: {len(seen)}")

    asyncio.run(main())
