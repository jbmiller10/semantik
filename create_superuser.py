#!/usr/bin/env python
"""
Script to create a superuser for Semantik.
Usage: python create_superuser.py [username] [email] [password]
If no arguments provided, will prompt interactively.
"""
import asyncio

# Handle both Docker and local environments
import sys
from datetime import UTC, datetime
from getpass import getpass
from pathlib import Path

if Path("/app").exists():
    sys.path.insert(0, "/app")
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import select

from packages.shared.database import get_db, pwd_context
from packages.shared.database.models import User


async def create_superuser(username: str, email: str, password: str):
    """Create a superuser with the given credentials."""
    # Create async session
    async for session in get_db():
        # Check if user already exists
        result = await session.execute(select(User).where((User.username == username) | (User.email == email)))
        existing_user = result.scalar_one_or_none()

        if existing_user:
            if existing_user.username == username:
                print(f"Error: User with username '{username}' already exists.")
            else:
                print(f"Error: User with email '{email}' already exists.")

            # Ask if they want to make existing user a superuser
            if existing_user.is_superuser:
                print("This user is already a superuser.")
                return False
            response = input("Would you like to make this existing user a superuser? (y/n): ")
            if response.lower() == "y":
                existing_user.is_superuser = True
                existing_user.updated_at = datetime.now(UTC)
                await session.commit()
                print(f"User '{existing_user.username}' is now a superuser!")
                return True
            return False

        # Create new superuser
        hashed_password = pwd_context.hash(password)
        new_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_active=True,
            is_superuser=True,
            full_name=f"{username} (Superuser)",
            created_at=datetime.now(UTC),
        )

        session.add(new_user)
        await session.commit()

        print("\nSuperuser created successfully!")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print("You can now log in and use the reset database functionality.")
        return True

    # Should not reach here, but return False if somehow we do
    return False


async def main():
    """Main function to handle command line arguments or interactive input."""
    print("=== Semantik Superuser Creation ===\n")

    if len(sys.argv) == 4:
        # Use command line arguments
        username = sys.argv[1]
        email = sys.argv[2]
        password = sys.argv[3]
    else:
        # Interactive mode
        print("No arguments provided. Entering interactive mode.\n")
        username = input("Enter username: ").strip()
        email = input("Enter email: ").strip()
        password = getpass("Enter password: ")
        confirm_password = getpass("Confirm password: ")

        if password != confirm_password:
            print("Error: Passwords do not match.")
            sys.exit(1)

    if not username or not email or not password:
        print("Error: All fields are required.")
        sys.exit(1)

    # Basic email validation
    if "@" not in email:
        print("Error: Invalid email address.")
        sys.exit(1)

    # Create the superuser
    success = await create_superuser(username, email, password)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
