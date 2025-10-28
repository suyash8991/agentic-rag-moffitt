#!/bin/bash

# Cross-platform setup verification script
# This script checks if the environment is properly configured

set -e

echo "================================================"
echo "  Moffitt Agentic RAG - Setup Verification"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
echo -n "Checking .env file... "
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo -e "${YELLOW}Please copy .env.docker.example to .env and configure it${NC}"
    exit 1
fi

# Check if Docker is installed
echo -n "Checking Docker installation... "
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d ' ' -f3 | cut -d ',' -f1)
    echo -e "${GREEN}✓ Docker ${DOCKER_VERSION}${NC}"
else
    echo -e "${RED}✗ Docker not found${NC}"
    echo -e "${YELLOW}Please install Docker from https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

# Check if Docker Compose is installed
echo -n "Checking Docker Compose installation... "
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | cut -d ' ' -f4 | cut -d ',' -f1)
    echo -e "${GREEN}✓ Docker Compose ${COMPOSE_VERSION}${NC}"
else
    echo -e "${RED}✗ Docker Compose not found${NC}"
    echo -e "${YELLOW}Please install Docker Compose from https://docs.docker.com/compose/install/${NC}"
    exit 1
fi

# Check if GROQ_API_KEY or OPENAI_API_KEY is set
echo -n "Checking API keys... "
if grep -q "GROQ_API_KEY=your_groq_api_key_here" .env || ! grep -q "GROQ_API_KEY=" .env; then
    if grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env || ! grep -q "OPENAI_API_KEY=" .env; then
        echo -e "${RED}✗ No valid API key found${NC}"
        echo -e "${YELLOW}Please set either GROQ_API_KEY or OPENAI_API_KEY in .env${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ API key configured${NC}"

# Check data directories
echo -n "Checking data directories... "
if [ ! -d "data" ]; then
    echo -e "${YELLOW}⚠ Creating data directories${NC}"
    mkdir -p data/vector_db data/processed data/markdown
fi
echo -e "${GREEN}✓ Data directories ready${NC}"

echo ""
echo "================================================"
echo -e "${GREEN}✅ All checks passed! Ready to run.${NC}"
echo "================================================"
echo ""
echo "To start the application, run:"
echo "  docker-compose up --build"
echo ""
