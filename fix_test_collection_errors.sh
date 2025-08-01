#!/bin/bash
# Fix test collection errors in test_search_api.py and test_celery_tasks.py

echo "Fixing test collection errors..."

# Fix test_search_api.py - restore from git and apply minimal fixes
cd /home/dockertest/semantik
git checkout tests/unit/test_search_api.py

# Apply only the critical fix for the mock_get_model_info function
python3 -c "
import re

with open('tests/unit/test_search_api.py', 'r') as f:
    content = f.read()

# Fix the mock_get_model_info to accept variable arguments
content = re.sub(
    r'def mock_get_model_info\(\):\s*return',
    'def mock_get_model_info(*args, **kwargs):\\n        return',
    content
)

with open('tests/unit/test_search_api.py', 'w') as f:
    f.write(content)

print('Fixed mock_get_model_info in test_search_api.py')
"

# Check if test_celery_tasks.py has any issues
echo "Checking test_celery_tasks.py for syntax errors..."
python3 -m py_compile tests/webui/test_celery_tasks.py 2>&1 | head -10

echo "Done!"