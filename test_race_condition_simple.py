#!/usr/bin/env python3
"""Simple test to verify the race condition fix."""

import asyncio
from playwright.async_api import async_playwright
import time
import json

async def test_simple():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("1. Navigating and logging in...")
        await page.goto("http://localhost:8080")
        await page.wait_for_selector('input[name="username"]', timeout=10000)
        await page.fill('input[name="username"]', "testuser")
        await page.fill('input[name="password"]', "testpassword123")
        await page.click('button[type="submit"]')
        await page.wait_for_timeout(3000)
        
        print("2. Creating collection with source...")
        await page.click('button:has-text("Create Collection")')
        await page.wait_for_selector('input[id="name"]', timeout=5000)
        
        collection_name = f"Race Test {int(time.time())}"
        await page.fill('input[id="name"]', collection_name)
        await page.fill('input[id="sourcePath"]', "/mnt/docs")
        
        # Capture responses
        responses = []
        async def capture_response(response):
            if "/api/v2/" in response.url:
                try:
                    data = await response.json()
                    responses.append({
                        "url": response.url,
                        "status": response.status,
                        "data": data
                    })
                except:
                    pass
        
        page.on("response", capture_response)
        
        # Submit form
        buttons = await page.query_selector_all('button:has-text("Create Collection")')
        await buttons[-1].click()
        
        # Wait for responses
        await page.wait_for_timeout(5000)
        
        print("\n=== API Responses ===")
        collection_created = False
        source_error = False
        
        for resp in responses:
            if "collections" in resp['url'] and resp['status'] == 201:
                collection_created = True
                print(f"✓ Collection created: {resp['data'].get('id')}")
                
            if "sources" in resp['url']:
                if resp['status'] == 409:
                    source_error = True
                    print(f"✓ RACE CONDITION PREVENTED! Status: {resp['status']}")
                    print(f"  Error: {resp['data'].get('detail', 'Unknown error')}")
                elif resp['status'] == 202:
                    print(f"✗ Source was added (should have been blocked): Status {resp['status']}")
        
        # Check for toast messages
        await page.wait_for_timeout(2000)
        toasts = await page.query_selector_all('[role="alert"]')
        if toasts:
            print("\n=== Toast Messages ===")
            for toast in toasts:
                text = await toast.text_content()
                print(f"  {text}")
                if "cannot add source while another operation is in progress" in text.lower():
                    source_error = True
                    print("  ^ RACE CONDITION PROPERLY PREVENTED!")
        
        print("\n=== Test Result ===")
        if collection_created and source_error:
            print("✓ TEST PASSED: Race condition was properly prevented!")
            print("  - Collection was created successfully")
            print("  - Source addition was blocked due to active INDEX operation")
        elif collection_created and not source_error:
            print("✗ TEST FAILED: Race condition still exists!")
            print("  - Collection was created")
            print("  - Source was added without waiting for INDEX to complete")
        else:
            print("? TEST INCONCLUSIVE")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_simple())