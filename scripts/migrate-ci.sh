#!/bin/bash
# CI/CD Migration Script
# This script helps migrate from the old multi-workflow setup to the new consolidated workflow

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Semantik CI/CD Migration Script ===${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ] || [ ! -d ".github/workflows" ]; then
    echo -e "${RED}Error: This script must be run from the Semantik project root${NC}"
    exit 1
fi

# Function to prompt user
confirm() {
    read -p "$1 (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 1
    fi
    return 0
}

echo -e "${YELLOW}This script will:${NC}"
echo "1. Back up existing workflows"
echo "2. Remove old workflow files"
echo "3. Activate the new consolidated workflow"
echo "4. Update your local environment"
echo ""

if ! confirm "Do you want to proceed?"; then
    echo -e "${RED}Migration cancelled${NC}"
    exit 0
fi

# Step 1: Create backup
echo -e "\n${GREEN}Step 1: Creating backup of existing workflows${NC}"
BACKUP_DIR=".github/workflows/backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

for workflow in ci.yml frontend-tests.yml test-all.yml pr-checks.yml; do
    if [ -f ".github/workflows/$workflow" ]; then
        cp ".github/workflows/$workflow" "$BACKUP_DIR/"
        echo "  ✓ Backed up $workflow"
    fi
done

# Step 2: Remove old workflows
echo -e "\n${GREEN}Step 2: Removing old workflow files${NC}"
for workflow in ci.yml frontend-tests.yml test-all.yml pr-checks.yml; do
    if [ -f ".github/workflows/$workflow" ]; then
        rm ".github/workflows/$workflow"
        echo "  ✓ Removed $workflow"
    fi
done

# Step 3: Verify new workflow exists
echo -e "\n${GREEN}Step 3: Verifying new workflow${NC}"
if [ ! -f ".github/workflows/main.yml" ]; then
    echo -e "${RED}Error: main.yml not found. Please ensure the new workflow file exists${NC}"
    exit 1
fi
echo "  ✓ main.yml is present"

# Step 4: Update git
echo -e "\n${GREEN}Step 4: Staging changes${NC}"
git add -A .github/workflows/
echo "  ✓ Changes staged"

# Step 5: Show summary
echo -e "\n${GREEN}Step 5: Migration Summary${NC}"
echo "  Old workflows backed up to: $BACKUP_DIR"
echo "  New consolidated workflow: .github/workflows/main.yml"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review the changes: git diff --staged"
echo "2. Commit the changes: git commit -m 'refactor: consolidate CI/CD workflows'"
echo "3. Push to a feature branch to test"
echo "4. Update branch protection rules in GitHub settings"
echo ""
echo -e "${GREEN}Required GitHub settings updates:${NC}"
echo "- Go to Settings → Branches → Edit protection rules"
echo "- Remove old status checks: 'lint', 'test', 'backend-tests', etc."
echo "- Add new status checks:"
echo "  • quality-checks"
echo "  • backend-tests"
echo "  • frontend-tests" 
echo "  • build-validation"
echo "  • ci-summary"
echo ""
echo -e "${YELLOW}Optional: Add these secrets in GitHub:${NC}"
echo "- CODECOV_TOKEN (for coverage reports)"
echo "- SAFETY_API_KEY (for security scanning)"
echo ""
echo -e "${GREEN}✓ Migration preparation complete!${NC}"