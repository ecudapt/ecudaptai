#!/bin/bash

# ECUdapt AI Forum Scraper Runner
# Usage: ./run_scraper.sh [urls_file]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}   ECUdapt AI Forum Scraper${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"

# Default to urls.txt if no argument provided
URLS_FILE="${1:-urls.txt}"

# Check if urls file exists
if [ ! -f "$URLS_FILE" ]; then
    echo -e "${RED}❌ Error: URLs file not found: $URLS_FILE${NC}"
    echo -e "${YELLOW}Usage: ./run_scraper.sh [urls_file]${NC}"
    exit 1
fi

# Check if .env exists
if [ ! -f "../../.env" ]; then
    echo -e "${RED}❌ Error: .env file not found at project root${NC}"
    echo -e "${YELLOW}Please create .env with VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY${NC}"
    exit 1
fi

# Count URLs
URL_COUNT=$(grep -v '^#' "$URLS_FILE" | grep -v '^$' | wc -l)
echo -e "${GREEN}📂 Loading $URL_COUNT URLs from $URLS_FILE${NC}"

# Install playwright browsers if needed
if [ ! -d "$HOME/.cache/ms-playwright" ]; then
    echo -e "${YELLOW}📥 Installing Playwright browsers (first time only)...${NC}"
    playwright install chromium
fi

# Run the spider
echo -e "${BLUE}🚀 Starting scraper...${NC}"
echo ""

cd ecudapt
scrapy crawl forums_hybrid -a urls_file="../$URLS_FILE"

echo ""
echo -e "${GREEN}✅ Scraping complete!${NC}"
