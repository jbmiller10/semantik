#!/usr/bin/env python3
"""Test script with better wait conditions."""

import asyncio

from playwright.async_api import async_playwright


async def test_with_proper_waits():
    async with async_playwright() as p:
        # Launch with devtools to see console errors
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Listen for console messages
        page.on("console", lambda msg: print(f"Console {msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Page error: {err}"))

        print("1. Navigating to main page...")
        await page.goto("http://localhost:8080")

        # Wait for the React app to load
        print("2. Waiting for React app to load...")
        try:
            # Wait for login form or collections page
            await page.wait_for_selector('input[name="username"], h1:has-text("Collections")', timeout=10000)
            print("App loaded!")
        except Exception:
            print("App failed to load properly")
            await page.screenshot(path="app_load_error.png")
            content = await page.content()
            print(f"Page content length: {len(content)}")
            await browser.close()
            return

        # Check if we need to login
        if await page.query_selector('input[name="username"]'):
            print("3. Login required, logging in...")
            await page.fill('input[name="username"]', "testuser")
            await page.fill('input[name="password"]', "testpassword123")
            await page.click('button[type="submit"]')

            # Wait for navigation
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)

        print(f"4. Current URL: {page.url}")

        # Wait for collections page to load
        try:
            await page.wait_for_selector('h1:has-text("Collections"), h2:has-text("Collections")', timeout=5000)
            print("Collections page loaded!")
        except Exception:
            print("Collections page not found")
            await page.screenshot(path="collections_error.png")

        # Look for the New Collection button with various selectors
        print("5. Looking for New Collection button...")
        button_selectors = [
            'button:has-text("New Collection")',
            'button:has-text("Create Collection")',
            'button:has-text("Add Collection")',
            'button >> text="New Collection"',
            'text="New Collection"',
            '[data-testid="new-collection-button"]',
            "button.btn-primary",
            'button[class*="primary"]',
        ]

        button_found = False
        for selector in button_selectors:
            try:
                button = await page.wait_for_selector(selector, timeout=1000)
                if button:
                    print(f"✓ Found button with selector: {selector}")
                    button_text = await button.text_content()
                    print(f"  Button text: {button_text}")
                    button_found = True

                    # Click the button
                    await button.click()
                    print("6. Clicked New Collection button!")

                    # Wait for modal
                    await page.wait_for_selector('input[id="name"]', timeout=5000)
                    print("7. Create Collection modal opened!")

                    break
            except Exception:
                continue

        if not button_found:
            print("✗ New Collection button not found")
            await page.screenshot(path="no_button_found.png")

            # Get all visible text
            visible_text = await page.evaluate("() => document.body.innerText")
            print(f"\nVisible text on page:\n{visible_text[:500]}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(test_with_proper_waits())
