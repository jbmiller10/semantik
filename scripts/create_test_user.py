#!/usr/bin/env python
import sys
from pathlib import Path

if Path("/app").exists():
    sys.path.insert(0, "/app")
else:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "packages"))

import os

from shared.config.postgres import postgres_config
from shared.database.models import User
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from webui.auth import get_password_hash

if "JWT_SECRET_KEY" not in os.environ:
    raise RuntimeError(
        "JWT_SECRET_KEY must be set before running this script. "
        "Generate one with `uv run python scripts/generate_jwt_secret.py --write` or export it manually."
    )

engine = create_engine(postgres_config.sync_database_url)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# Check existing users
users = db.query(User).all()
print("Existing users:")
for user in users:
    print(f"  - {user.username} (is_active: {user.is_active})")

# Create a test user if needed
test_user = db.query(User).filter_by(username="testuser").first()
if not test_user:
    hashed_password = get_password_hash("testuser123")
    test_user = User(username="testuser", email="test@test.com", hashed_password=hashed_password)
    db.add(test_user)
    db.commit()
    print("\nCreated test user: testuser/testuser123")
else:
    print(f"\nTest user exists: is_active={test_user.is_active}")

db.close()
