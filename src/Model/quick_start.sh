#!/bin/bash
# Quick Start Script for ECUdapt AI LLM Training

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║         ECUdapt AI - LLM Training Quick Start         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python found: $(python3 --version)${NC}"

# Check if in correct directory
if [ ! -f "pipeline.py" ]; then
    echo -e "${RED}❌ Please run this script from the Model directory${NC}"
    exit 1
fi

# Check for scraped data
if [ ! -d "../Scraper/data/raw" ] || [ -z "$(ls -A ../Scraper/data/raw/*.jsonl 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠️  No scraped forum data found${NC}"
    echo -e "${YELLOW}   Run the scraper first:${NC}"
    echo -e "${YELLOW}   cd ../Scraper && ./run_scraper.sh urls.txt${NC}"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo ""
echo -e "${BLUE}📦 Installing dependencies...${NC}"
read -p "Install required packages? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m pip install -q -r requirements_llm.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
fi

# Choose configuration
echo ""
echo -e "${BLUE}⚙️  Choose configuration:${NC}"
echo "   1. Full (Mistral-7B, requires 12GB+ VRAM)"
echo "   2. Lightweight (Phi-2 2.7B, requires 6GB VRAM)"
echo "   3. CPU-friendly (TinyLlama 1.1B)"
echo ""
read -p "Select option (1-3): " -n 1 -r
echo ""

case $REPLY in
    1)
        CONFIG_FLAG=""
        echo -e "${GREEN}Selected: Full configuration (Mistral-7B)${NC}"
        ;;
    2)
        CONFIG_FLAG="--lightweight"
        echo -e "${GREEN}Selected: Lightweight (Phi-2)${NC}"
        ;;
    3)
        CONFIG_FLAG="--model TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        echo -e "${GREEN}Selected: CPU-friendly (TinyLlama)${NC}"
        ;;
    *)
        CONFIG_FLAG="--lightweight"
        echo -e "${YELLOW}Invalid option, using Lightweight${NC}"
        ;;
esac

# Choose what to run
echo ""
echo -e "${BLUE}🚀 What would you like to do?${NC}"
echo "   1. Run complete pipeline (clean → train → evaluate)"
echo "   2. Prepare data only"
echo "   3. Train model only (requires prepared data)"
echo "   4. Test trained model"
echo ""
read -p "Select option (1-4): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo -e "${GREEN}Running complete pipeline...${NC}"
        python3 pipeline.py $CONFIG_FLAG
        ;;
    2)
        echo -e "${GREEN}Preparing training data...${NC}"
        python3 pipeline.py --step prepare
        ;;
    3)
        echo -e "${GREEN}Training model...${NC}"
        python3 train_llm.py $CONFIG_FLAG
        ;;
    4)
        if [ -d "models/ecutuning-llm" ]; then
            echo -e "${GREEN}Testing trained model...${NC}"
            python3 inference.py
        else
            echo -e "${RED}❌ No trained model found${NC}"
            echo -e "${YELLOW}   Train a model first with option 1 or 3${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Complete!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  • Test your model: ${YELLOW}python3 inference.py${NC}"
echo -e "  • Evaluate performance: ${YELLOW}python3 evaluate.py${NC}"
echo -e "  • View training logs: ${YELLOW}tensorboard --logdir logs/${NC}"
echo ""
