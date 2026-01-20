"""Constants for model manager operations."""

# Celery queue for model-manager tasks
MODEL_MANAGER_QUEUE = "model-manager"

# Redis key patterns
ACTIVE_OP_KEY_PATTERN = "model-manager:active:{model_id}"
TASK_PROGRESS_KEY_PATTERN = "model-manager:task:{task_id}"

# TTLs (seconds)
PROGRESS_KEY_TTL = 86400  # 24 hours
ACTIVE_KEY_TTL = 86400  # 24 hours (refreshed on progress updates)

# HF cache scan TTL (seconds)
HF_CACHE_SCAN_TTL = 15  # 15 seconds default
