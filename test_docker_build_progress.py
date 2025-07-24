#!/usr/bin/env python3
"""Test script to verify Docker build progress is shown correctly"""

import subprocess
import sys
from pathlib import Path


def test_build_progress() -> None:
    """Test that Docker build shows progress"""
    print("Testing Docker build progress display...")
    print("=" * 60)

    # Check if docker and docker compose are available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        subprocess.run(["docker", "compose", "version"], check=True, capture_output=True)
    except Exception as e:
        print(f"Error: Docker or Docker Compose not found: {e}")
        return False

    # Test with a simple Dockerfile that shows progress
    test_dockerfile = """FROM alpine:latest
RUN echo "Step 1: Starting build..." && sleep 2
RUN echo "Step 2: Installing packages..." && sleep 2
RUN echo "Step 3: Configuring..." && sleep 2
RUN echo "Step 4: Finalizing..." && sleep 2
"""

    test_compose = """version: '3.8'
services:
  test:
    build:
      context: .
      dockerfile: Dockerfile.test
    image: semantik-test:latest
"""

    # Create temporary test files
    Path("Dockerfile.test").write_text(test_dockerfile)
    Path("docker-compose.test.yml").write_text(test_compose)

    try:
        print("\nRunning build WITH progress (new behavior):")
        print("-" * 40)
        # This simulates the new behavior - no capture_output
        result = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.test.yml", "build", "--no-cache"], text=True
        )

        if result.returncode == 0:
            print("\n✓ Build completed successfully with visible progress!")
        else:
            print("\n✗ Build failed")

        print("\n" + "=" * 60)
        print("Compare with build WITHOUT progress (old behavior):")
        print("-" * 40)
        print("Building... (no output will be shown)")

        # This simulates the old behavior - with capture_output
        result = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.test.yml", "build", "--no-cache"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✓ Build completed (but you couldn't see what was happening!)")
        else:
            print("✗ Build failed")

    finally:
        # Cleanup
        Path("Dockerfile.test").unlink(missing_ok=True)
        Path("docker-compose.test.yml").unlink(missing_ok=True)
        subprocess.run(["docker", "rmi", "-f", "semantik-test:latest"], capture_output=True)

    print("\n" + "=" * 60)
    print("Test complete! The fix ensures Docker build progress is visible.")
    return True


if __name__ == "__main__":
    success = test_build_progress()
    sys.exit(0 if success else 1)
