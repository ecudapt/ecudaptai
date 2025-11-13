# ECUdapt AI

ECUdapt AI is an intelligent data collection and processing system focused on automotive ECU tuning forums and related performance discussions. It automatically crawls tuning forums, extracts relevant technical posts, and stores them in a Supabase database for later use in vector-based semantic search and AI-assisted query systems.

## Features

- **Multi-Forum Support**: Crawls 30+ major automotive tuning forums
- **Smart Filtering**: Keyword-based relevance filtering for ECU/tuning content
- **Hybrid Crawling**: Supports both static and JavaScript-rendered sites using Playwright
- **Database Integration**: Automatic storage to Supabase with deduplication
- **Progress Tracking**: Real-time colorized console output and per-domain statistics
- **Respectful Crawling**: Auto-throttling and configurable delays to respect server resources

## Architecture

### Components

1. **Scrapy Spider** (`src/Scraper/ecudapt/spiders/forum_spider_hybrid.py`)
   - Crawls forum threads based on keyword relevance
   - Supports multiple forum engines (vBulletin, phpBB, XenForo, Discourse)
   - Uses Playwright for JavaScript-heavy sites

2. **Supabase Pipeline** (`src/Scraper/ecudapt/pipelines.py`)
   - Stores threads and posts in Supabase database
   - Tracks scrape runs with statistics
   - Handles deduplication via URL uniqueness

3. **Database Schema**
   - `forum_threads`: Thread metadata and URLs
   - `forum_posts`: Individual posts with content
   - `scrape_runs`: Scraping session statistics

## Setup

### Prerequisites

- Python 3.8+
- Supabase account
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ecudapt/ecudaptai.git
   cd ecudaptai
   ```

2. **Install dependencies**
   ```bash
   cd src
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers** (first time only)
   ```bash
   playwright install chromium
   ```

4. **Configure environment variables**

   The `.env` file should already exist in the project root with your Supabase credentials:
   ```
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

### Database Setup

The database schema has been automatically created in your Supabase instance with the following tables:

- **forum_threads**: Stores thread metadata (URL, domain, title, post count)
- **forum_posts**: Stores individual posts (author, date, content)
- **scrape_runs**: Tracks scraping sessions and statistics

## Usage

### Basic Usage

Run the scraper with the default URL list:

```bash
cd src/Scraper
./run_scraper.sh
```

### Custom URL List

Run with a custom list of forum URLs:

```bash
./run_scraper.sh test_urls.txt
```

### Manual Scrapy Command

You can also run Scrapy directly:

```bash
cd src/Scraper/ecudapt
scrapy crawl forums_hybrid -a urls_file="../urls.txt"
```

## Configuration

### Keyword Filtering

Edit `src/Scraper/ecudapt/spiders/forum_spider_hybrid.py` to modify:

- **INCLUDE_KEYWORDS**: Terms that indicate relevant threads (e.g., "ecu", "tune", "boost")
- **EXCLUDE_KEYWORDS**: Terms that indicate off-topic threads (e.g., "wheels", "audio")

### Crawl Settings

Modify `src/Scraper/ecudapt/settings.py` to adjust:

- **CONCURRENT_REQUESTS**: Number of parallel requests (default: 16)
- **DOWNLOAD_DELAY**: Delay between requests in seconds (default: 0.25)
- **AUTOTHROTTLE_ENABLED**: Automatic throttling based on server load

### Forum List

Add or remove forum URLs in `src/Scraper/urls.txt`. The scraper currently targets 31 automotive tuning forums including:

- ECU Edit, EEC Tuning, HP Academy
- Hondata, Bimmerboost, Audizine
- Golf Mk6/Mk7, N54Tech, FT86Club
- And many more...

## Output

### Database

All scraped data is stored in Supabase:

- Query threads: `SELECT * FROM forum_threads WHERE domain = 'ecuedit.com'`
- Query posts: `SELECT * FROM forum_posts WHERE thread_id = '...'`
- View scrape history: `SELECT * FROM scrape_runs ORDER BY started_at DESC`

### Backup Files

JSON Lines files are also created in `data/raw/` as backup:
- `forum_posts_raw.jsonl`: Contains all scraped threads and posts

## Monitoring

The scraper provides real-time feedback:

- **Console Output**: Colorized progress updates per domain
- **Summary Table**: Final statistics showing threads/posts per domain
- **Database Tracking**: `scrape_runs` table tracks each session

## Troubleshooting

### Missing Dependencies

If you see import errors:
```bash
pip install -r src/requirements.txt
```

### Playwright Not Installed

If JavaScript-heavy sites fail:
```bash
playwright install chromium
```

### Database Connection Errors

Verify your `.env` file contains valid Supabase credentials:
```bash
cat .env
```

### Permission Errors

Make sure the run script is executable:
```bash
chmod +x src/Scraper/run_scraper.sh
```

## Future Enhancements

- Vector embedding generation for semantic search
- AI-powered query interface
- Real-time monitoring dashboard
- Incremental updates (only scrape new threads)
- Multi-language support
- Advanced deduplication algorithms

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions, please open a GitHub issue or contact the maintainers.
