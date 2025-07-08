#!/bin/bash
# Fix permissions for Docker volumes
# The containers run as user 1000, so directories must be owned by UID 1000

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Fixing permissions for Docker volumes${NC}"
echo "====================================="

# Directories to fix
DIRS=("./models" "./data" "./logs")

# Create directories if they don't exist
for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}Creating directory: $dir${NC}"
        mkdir -p "$dir"
    fi
done

# Fix ownership
echo -e "\n${YELLOW}Setting ownership to UID 1000 (container user)...${NC}"

if command -v sudo >/dev/null 2>&1; then
    sudo chown -R 1000:1000 "${DIRS[@]}"
    echo -e "${GREEN}✓ Permissions fixed using sudo${NC}"
else
    # Try without sudo
    if chown -R 1000:1000 "${DIRS[@]}" 2>/dev/null; then
        echo -e "${GREEN}✓ Permissions fixed${NC}"
    else
        echo -e "${RED}ERROR: Could not fix permissions. Please run:${NC}"
        echo "  sudo chown -R 1000:1000 ./models ./data ./logs"
        exit 1
    fi
fi

# Verify permissions
echo -e "\n${GREEN}Verifying permissions:${NC}"
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        owner=$(stat -c '%u:%g' "$dir" 2>/dev/null || stat -f '%u:%g' "$dir" 2>/dev/null)
        echo "  $dir: owner=$owner"
    fi
done

echo -e "\n${GREEN}✅ Permissions fixed! You can now run: docker compose up -d${NC}"