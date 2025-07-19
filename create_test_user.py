#!/usr/bin/env python
import sys
sys.path.insert(0, '/app')

from packages.shared.database.models.user import User
from packages.shared.database.session import get_db_engine, SessionLocal
from sqlalchemy.orm import sessionmaker
import packages.webui.auth.utils as auth_utils

engine = get_db_engine()
Session = sessionmaker(bind=engine)
db = Session()

# Check existing users
users = db.query(User).all()
print('Existing users:')
for user in users:
    print(f'  - {user.username} (is_active: {user.is_active})')

# Create a test user if needed
test_user = db.query(User).filter_by(username='testuser').first()
if not test_user:
    hashed_password = auth_utils.get_password_hash('testuser123')
    test_user = User(username='testuser', email='test@test.com', hashed_password=hashed_password)
    db.add(test_user)
    db.commit()
    print('\nCreated test user: testuser/testuser123')
else:
    print(f'\nTest user exists: is_active={test_user.is_active}')
    
db.close()