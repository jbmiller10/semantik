#!/usr/bin/env python3
"""Test script to verify the race condition fix for collection creation with initial source."""

import asyncio
from playwright.async_api import async_playwright
import time


async def test_collection_creation_with_source():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("1. Navigating to login page...")
        await page.goto("http://localhost:8080/login")
        await page.wait_for_load_state("networkidle")

        print("2. Logging in...")
        await page.fill('input[name="username"]', "testuser")
        await page.fill('input[name="password"]', "testpassword123")
        await page.click('button[type="submit"]')

        # Wait for navigation after login
        await page.wait_for_timeout(2000)

        # Check if we're on the collections page
        if "/collections" in page.url:
            print("Login successful!")
        else:
            print(f"Login redirected to: {page.url}")
            # Try to navigate to collections page manually
            await page.goto("http://localhost:8080/collections")
            await page.wait_for_load_state("networkidle")

        print("3. Creating collection with initial source...")
        # Click "New Collection" button
        await page.click('button:has-text("New Collection")')
        await page.wait_for_selector('input[id="name"]', state="visible")

        # Fill in collection details
        collection_name = f"Test Collection {int(time.time())}"
        await page.fill('input[id="name"]', collection_name)
        await page.fill('textarea[id="description"]', "Testing race condition fix")
        await page.fill('input[id="sourcePath"]', "/mnt/docs")

        print(f"4. Creating collection: {collection_name}")
        # Click Create button
        await page.click('button:has-text("Create Collection")')

        # Wait for the creation process to complete
        print("5. Waiting for collection creation and source addition...")
        try:
            # Wait for success message or navigation
            await page.wait_for_function(
                """() => {
                    const toasts = document.querySelectorAll('[role="alert"]');
                    for (const toast of toasts) {
                        if (toast.textContent.includes('created') || toast.textContent.includes('success')) {
                            return true;
                        }
                    }
                    return window.location.pathname.includes('/collections/');
                }""",
                timeout=30000,
            )

            # Check current URL
            current_url = page.url
            print(f"Current URL: {current_url}")

            if "/collections/" in current_url:
                print("✓ Successfully navigated to collection page!")

                # Wait a bit for the page to load
                await page.wait_for_timeout(3000)

                # Check for document count
                doc_count_element = await page.query_selector("text=/[0-9]+ documents?/")
                if doc_count_element:
                    doc_count_text = await doc_count_element.text_content()
                    print(f"✓ Document count shown: {doc_count_text}")
                else:
                    print("⚠ Document count not found on page")

                # Check for any error messages
                error_elements = await page.query_selector_all('[role="alert"].error, .text-red-600')
                if error_elements:
                    for elem in error_elements:
                        error_text = await elem.text_content()
                        print(f"⚠ Error found: {error_text}")
            else:
                print("⚠ Did not navigate to collection page")

                # Check for any error messages on current page
                error_elements = await page.query_selector_all('[role="alert"]')
                for elem in error_elements:
                    alert_text = await elem.text_content()
                    print(f"Alert: {alert_text}")

        except Exception as e:
            print(f"✗ Error during test: {e}")

            # Take screenshot for debugging
            await page.screenshot(path="test_error.png")
            print("Screenshot saved as test_error.png")

            # Print any visible alerts
            alerts = await page.query_selector_all('[role="alert"]')
            for alert in alerts:
                alert_text = await alert.text_content()
                print(f"Alert found: {alert_text}")

        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(test_collection_creation_with_source())
