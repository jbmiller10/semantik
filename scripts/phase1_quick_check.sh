#!/bin/bash
# Phase 1 Quick Check Script
# Run this before committing to ensure all Phase 1 requirements are met

set -e

echo "=========================================="
echo "Phase 1: Database & Model Alignment Check"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    echo -e "${RED}✗${NC} Error: Must be run from project root"
    exit 1
fi

echo "1. Checking Python syntax in migration files..."
python scripts/verify_migrations.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All migrations are syntactically correct"
else
    echo -e "${RED}✗${NC} Migration syntax errors found"
    exit 1
fi

echo ""
echo "2. Running dry validation (no database required)..."
python scripts/phase1_validation_dry_run.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Dry validation passed"
else
    echo -e "${RED}✗${NC} Dry validation failed"
    exit 1
fi

echo ""
echo "3. Checking for required files..."
REQUIRED_FILES=(
    "packages/shared/database/models.py"
    "alembic/versions/db003_replace_trigger_with_generated_column.py"
    "scripts/phase1_validation.py"
    "PHASE1_VALIDATION_REPORT.md"
)

all_files_exist=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} Found: $file"
    else
        echo -e "${RED}✗${NC} Missing: $file"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo -e "${RED}✗${NC} Some required files are missing"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Phase 1 Quick Check PASSED${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review PHASE1_VALIDATION_REPORT.md"
echo "2. Start database: make docker-dev-up"
echo "3. Run full validation: python scripts/phase1_validation.py"
echo "4. Create PR with the suggested commit message"
echo ""