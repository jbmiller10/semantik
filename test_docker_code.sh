#!/bin/bash
# Test if the Docker container has the updated code

echo "Checking if the datetime_to_str function exists in the container..."
docker exec semantik-webui grep -n "def datetime_to_str" /app/packages/webui/repositories/postgres/user_repository.py

echo -e "\nChecking the _user_to_dict method..."
docker exec semantik-webui grep -A 5 "def _user_to_dict" /app/packages/webui/repositories/postgres/user_repository.py

echo -e "\nChecking for debug logging..."
docker exec semantik-webui grep -n "User after refresh" /app/packages/webui/repositories/postgres/user_repository.py