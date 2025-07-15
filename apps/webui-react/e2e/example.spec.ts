import { test, expect } from '@playwright/test';

test.describe('Semantik Frontend', () => {
  test('has title', async ({ page }) => {
    await page.goto('/');

    // Expects page to have a title containing "Semantik"
    await expect(page).toHaveTitle(/Semantik/);
  });

  test('can navigate to login page', async ({ page }) => {
    await page.goto('/');

    // Check if we're redirected to login or if login link exists
    const url = page.url();
    
    if (url.includes('/login')) {
      // We've been redirected to login
      await expect(page.getByRole('heading', { name: /sign in/i })).toBeVisible();
    } else {
      // Look for login link and click it
      const loginLink = page.getByRole('link', { name: /sign in|login/i });
      await loginLink.click();
      await expect(page).toHaveURL(/\/login/);
    }
  });

  test('app renders without errors', async ({ page }) => {
    // Listen for console errors
    const errors: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error') {
        errors.push(message.text());
      }
    });

    await page.goto('/');
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');

    // Assert no console errors
    expect(errors).toHaveLength(0);
  });
});