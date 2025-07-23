#!/usr/bin/env python3
"""Check the operations in the database to verify the fix."""

import asyncio
from playwright.async_api import async_playwright
import time
import json

async def check_operations():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("1. Navigating to app...")
        await page.goto("http://localhost:8080")
        await page.wait_for_selector('input[name="username"]', timeout=10000)
        
        print("2. Logging in...")
        await page.fill('input[name="username"]', "testuser")
        await page.fill('input[name="password"]', "testpassword123")
        await page.click('button[type="submit"]')
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)
        
        # Monitor all API responses
        all_responses = []
        
        async def capture_response(response):
            if "/api/v2/" in response.url:
                try:
                    data = await response.json()
                    all_responses.append({
                        "url": response.url,
                        "method": response.request.method,
                        "status": response.status,
                        "data": data
                    })
                except:
                    pass
        
        page.on("response", capture_response)
        
        print("3. Opening Create Collection modal...")
        await page.click('button:has-text("Create Collection")')
        await page.wait_for_selector('input[id="name"]', timeout=5000)
        
        # Fill form
        collection_name = f"Debug Collection {int(time.time())}"
        print(f"4. Creating collection: {collection_name}")
        await page.fill('input[id="name"]', collection_name)
        await page.fill('textarea[id="description"]', "Debug test")
        await page.fill('input[id="sourcePath"]', "/mnt/docs")
        
        # Submit
        print("5. Submitting form...")
        buttons = await page.query_selector_all('button:has-text("Create Collection")')
        await buttons[-1].click()
        
        # Wait a bit to capture all responses
        await page.wait_for_timeout(5000)
        
        print("\n=== All API Responses ===")
        for resp in all_responses:
            print(f"\n{resp['method']} {resp['url']} - Status: {resp['status']}")
            if "collections" in resp['url'] and resp['method'] == "POST":
                print("Collection creation response:")
                print(json.dumps(resp['data'], indent=2))
            elif "operations" in resp['url']:
                print("Operation response:")
                print(json.dumps(resp['data'], indent=2))
        
        # Now check the operations via API
        print("\n=== Checking operations directly ===")
        
        # Get the collection ID from responses
        collection_id = None
        for resp in all_responses:
            if "collections" in resp['url'] and resp['method'] == "POST":
                collection_id = resp['data'].get('id')
                break
        
        if collection_id:
            print(f"Collection ID: {collection_id}")
            
            # Navigate to the API endpoint for operations
            api_url = f"http://localhost:8080/api/v2/collections/{collection_id}/operations"
            response = await page.request.get(api_url)
            operations = await response.json()
            
            print(f"\nOperations for collection {collection_id}:")
            print(json.dumps(operations, indent=2))
            
            # Check operation types and statuses
            if 'data' in operations:
                for op in operations['data']:
                    print(f"\n- Operation {op['id']}:")
                    print(f"  Type: {op['type']}")
                    print(f"  Status: {op['status']}")
                    print(f"  Created: {op['created_at']}")
                    if op.get('error_message'):
                        print(f"  Error: {op['error_message']}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(check_operations())