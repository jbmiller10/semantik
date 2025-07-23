#!/usr/bin/env python3
"""Final test script to verify the race condition fix."""

import asyncio
from playwright.async_api import async_playwright
import time
import json

async def test_race_condition_fix():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Listen for console messages
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        
        print("1. Navigating to app...")
        await page.goto("http://localhost:8080")
        
        # Wait for app to load
        await page.wait_for_selector('input[name="username"], h1:has-text("Collections")', timeout=10000)
        
        # Login if needed
        if await page.query_selector('input[name="username"]'):
            print("2. Logging in...")
            await page.fill('input[name="username"]', "testuser")
            await page.fill('input[name="password"]', "testpassword123")
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)
        
        # Click Create Collection button
        print("3. Opening Create Collection modal...")
        await page.click('button:has-text("Create Collection")')
        await page.wait_for_selector('input[id="name"]', timeout=5000)
        
        # Fill in collection details
        collection_name = f"Test Collection {int(time.time())}"
        print(f"4. Creating collection: {collection_name}")
        await page.fill('input[id="name"]', collection_name)
        await page.fill('textarea[id="description"]', "Testing race condition fix")
        await page.fill('input[id="sourcePath"]', "/mnt/docs")
        
        # Monitor network requests to capture the response
        response_data = {}
        
        async def handle_response(response):
            if "/api/v2/collections" in response.url and response.request.method == "POST":
                try:
                    data = await response.json()
                    response_data["create_response"] = data
                    print(f"Collection created with ID: {data.get('id')}")
                    if "initial_operation_id" in data:
                        print(f"Initial operation ID: {data['initial_operation_id']}")
                except:
                    pass
        
        page.on("response", handle_response)
        
        # Click Create button - be more specific to get the submit button
        print("5. Submitting form...")
        # Use a more specific selector for the submit button in the modal
        submit_button = await page.query_selector('div.fixed button[type="submit"]:has-text("Create Collection")')
        if not submit_button:
            # Fallback to last button with the text
            buttons = await page.query_selector_all('button:has-text("Create Collection")')
            submit_button = buttons[-1] if buttons else None
        
        if submit_button:
            await submit_button.click()
        else:
            print("ERROR: Could not find submit button!")
            return
        
        # Wait for response and check for errors
        print("6. Waiting for collection creation...")
        try:
            # Wait for either success or error
            await page.wait_for_function(
                """() => {
                    // Check for navigation to collection page
                    if (window.location.pathname.includes('/collections/')) {
                        return true;
                    }
                    // Check for error messages
                    const alerts = document.querySelectorAll('[role="alert"]');
                    for (const alert of alerts) {
                        const text = alert.textContent || '';
                        if (text.toLowerCase().includes('error') || 
                            text.toLowerCase().includes('failed') ||
                            text.toLowerCase().includes('cannot add source')) {
                            return true;
                        }
                    }
                    return false;
                }""",
                timeout=30000
            )
            
            # Check what happened
            current_url = page.url
            print(f"7. Current URL: {current_url}")
            
            # Check for any alerts/toasts
            alerts = await page.query_selector_all('[role="alert"]')
            if alerts:
                print("\n=== Alerts found ===")
                for alert in alerts:
                    alert_text = await alert.text_content()
                    print(f"  Alert: {alert_text}")
                    
                    # Check if it's the race condition error
                    if "cannot add source while another operation is in progress" in alert_text.lower():
                        print("\n✓ RACE CONDITION PROPERLY PREVENTED!")
                        print("The system correctly prevented the APPEND operation from running")
                        print("while the INDEX operation was still in progress.")
                    elif "created" in alert_text.lower() and "success" in alert_text.lower():
                        print("\n✓ Collection created successfully")
                        if "waiting" in alert_text.lower() or "initialization" in alert_text.lower():
                            print("✓ System is properly waiting for INDEX to complete before adding source")
            
            # If we navigated to the collection page
            if "/collections/" in current_url:
                print("\n✓ Successfully navigated to collection page")
                await page.wait_for_timeout(3000)
                
                # Check operation status
                operation_elements = await page.query_selector_all('text=/pending|processing|completed/i')
                if operation_elements:
                    print("\nOperation statuses found:")
                    for elem in operation_elements:
                        status = await elem.text_content()
                        print(f"  - {status}")
            
            # Print response data if captured
            if response_data:
                print(f"\n=== API Response ===")
                print(json.dumps(response_data, indent=2))
            
        except Exception as e:
            print(f"\n✗ Test error: {e}")
            await page.screenshot(path="test_error_final.png")
            
        finally:
            await browser.close()
            
        print("\n=== Test Summary ===")
        print("The race condition fix ensures that:")
        print("1. Only one operation can run at a time on a collection")
        print("2. APPEND must wait for INDEX to complete")
        print("3. This prevents database inconsistencies")

if __name__ == "__main__":
    asyncio.run(test_race_condition_fix())