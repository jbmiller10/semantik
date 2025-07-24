#!/usr/bin/env python3
"""Debug script to check login and page state."""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def debug_login():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("1. Navigating to login page...")
        await page.goto("http://localhost:8080/login")
        await page.wait_for_load_state("networkidle")
        await page.screenshot(path="login_page.png")
        print("Login page screenshot saved")

        print("2. Logging in...")
        await page.fill('input[name="username"]', "testuser")
        await page.fill('input[name="password"]', "testpassword123")
        await page.screenshot(path="login_filled.png")
        print("Login form filled screenshot saved")

        await page.click('button[type="submit"]')

        # Wait for navigation
        await page.wait_for_timeout(3000)

        print(f"3. After login - Current URL: {page.url}")
        await page.screenshot(path="after_login.png")
        print("After login screenshot saved")

        # Navigate to collections
        print("4. Navigating to collections...")
        await page.goto("http://localhost:8080/collections")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)

        print(f"Current URL: {page.url}")
        await page.screenshot(path="collections_page.png")
        print("Collections page screenshot saved")

        # Look for buttons
        buttons = await page.query_selector_all("button")
        print(f"\nFound {len(buttons)} buttons on page:")
        for i, button in enumerate(buttons):
            text = await button.text_content()
            print(f"  Button {i}: {text}")

        # Get page content
        content = await page.content()
        with Path("page_content.html").open("w") as f:
            f.write(content)
        print("\nPage content saved to page_content.html")

        # Try different selectors
        new_collection_selectors = [
            'button:has-text("New Collection")',
            'button:has-text("Create")',
            'button:has-text("Add")',
            'a:has-text("New")',
            "button",
            '[role="button"]',
        ]

        print("\nTrying different selectors:")
        for selector in new_collection_selectors:
            elements = await page.query_selector_all(selector)
            if elements:
                print(f"  {selector}: Found {len(elements)} elements")
                for elem in elements[:3]:  # Show first 3
                    text = await elem.text_content()
                    print(f"    - {text}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(debug_login())
