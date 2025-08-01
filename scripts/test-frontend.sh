#!/bin/bash

# Frontend Test Runner Script
# This script runs frontend tests with various options

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
FRONTEND_DIR="$PROJECT_ROOT/apps/webui-react"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
RUN_ALL=false
RUN_COLLECTIONS=false
RUN_COVERAGE=false
WATCH_MODE=false
SPECIFIC_TEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --collections)
            RUN_COLLECTIONS=true
            shift
            ;;
        --coverage)
            RUN_COVERAGE=true
            shift
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --file)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --all          Run all frontend tests"
            echo "  --collections  Run only collection component tests"
            echo "  --coverage     Run tests with coverage report"
            echo "  --watch        Run tests in watch mode"
            echo "  --file <path>  Run a specific test file"
            echo "  --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --collections                    # Run all collection tests"
            echo "  $0 --coverage --collections         # Run collection tests with coverage"
            echo "  $0 --file CreateCollectionModal     # Run specific test file"
            echo "  $0 --watch                          # Run tests in watch mode"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Change to frontend directory
cd "$FRONTEND_DIR"

# Ensure dependencies are installed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm ci
fi

# Run the appropriate test command
if [ "$RUN_COVERAGE" = true ]; then
    echo -e "${GREEN}Running tests with coverage...${NC}"
    if [ "$RUN_COLLECTIONS" = true ]; then
        npm run test:coverage -- --run \
            src/components/__tests__/CollectionCard.test.tsx \
            src/components/__tests__/CreateCollectionModal.test.tsx \
            src/components/__tests__/CollectionsDashboard.test.tsx \
            src/components/__tests__/CollectionDetailsModal.test.tsx \
            src/components/__tests__/DeleteCollectionModal.test.tsx \
            src/components/__tests__/RenameCollectionModal.test.tsx \
            src/components/__tests__/ReindexCollectionModal.test.tsx \
            src/components/__tests__/AddDataToCollectionModal.test.tsx
    else
        npm run test:coverage -- --run
    fi
elif [ "$WATCH_MODE" = true ]; then
    echo -e "${GREEN}Running tests in watch mode...${NC}"
    if [ "$RUN_COLLECTIONS" = true ]; then
        npm test -- --watch "src/components/__tests__/*Collection*.test.tsx"
    else
        npm run test:watch
    fi
elif [ -n "$SPECIFIC_TEST" ]; then
    echo -e "${GREEN}Running specific test: $SPECIFIC_TEST${NC}"
    npm test -- --run "*$SPECIFIC_TEST*"
elif [ "$RUN_COLLECTIONS" = true ]; then
    echo -e "${GREEN}Running collection component tests...${NC}"
    npm test -- --run \
        src/components/__tests__/CollectionCard.test.tsx \
        src/components/__tests__/CreateCollectionModal.test.tsx \
        src/components/__tests__/CollectionsDashboard.test.tsx \
        src/components/__tests__/CollectionDetailsModal.test.tsx \
        src/components/__tests__/DeleteCollectionModal.test.tsx \
        src/components/__tests__/RenameCollectionModal.test.tsx \
        src/components/__tests__/ReindexCollectionModal.test.tsx \
        src/components/__tests__/AddDataToCollectionModal.test.tsx
elif [ "$RUN_ALL" = true ]; then
    echo -e "${GREEN}Running all frontend tests...${NC}"
    npm test -- --run
else
    # Default: run collection tests
    echo -e "${GREEN}Running collection component tests (default)...${NC}"
    npm test -- --run "src/components/__tests__/*Collection*.test.tsx"
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tests passed successfully!${NC}"
else
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi