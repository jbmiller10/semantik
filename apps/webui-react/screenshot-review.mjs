import { chromium } from 'playwright';

const BASE_URL = 'http://localhost:5174';

async function captureScreenshots() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 }
  });
  const page = await context.newPage();

  console.log('1. Capturing login page...');
  await page.goto(`${BASE_URL}/login`);
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: '/tmp/01-login-page.png', fullPage: true });
  console.log('   Saved: /tmp/01-login-page.png');

  console.log('2. Logging in...');
  await page.fill('input[placeholder="username"]', 'threepars');
  await page.fill('input[placeholder="Enter your password"]', 'puddin123');
  await page.screenshot({ path: '/tmp/02-login-filled.png', fullPage: true });
  console.log('   Saved: /tmp/02-login-filled.png');

  await page.click('button[type="submit"]');

  // Wait for navigation after login
  await page.waitForURL('**/', { timeout: 10000 }).catch(() => {});
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1000);

  console.log('3. Capturing main dashboard...');
  await page.screenshot({ path: '/tmp/03-dashboard.png', fullPage: true });
  console.log('   Saved: /tmp/03-dashboard.png');

  // Try clicking on Search tab
  console.log('4. Capturing Search tab...');
  const searchTab = page.getByRole('button', { name: 'Search' });
  if (await searchTab.isVisible()) {
    await searchTab.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: '/tmp/04-search.png', fullPage: true });
    console.log('   Saved: /tmp/04-search.png');
  }

  // Try clicking Collections tab
  console.log('5. Capturing Collections tab...');
  const collectionsTab = page.getByRole('button', { name: 'Collections' });
  if (await collectionsTab.isVisible()) {
    await collectionsTab.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: '/tmp/05-collections.png', fullPage: true });
    console.log('   Saved: /tmp/05-collections.png');
  }

  // Navigate to settings
  console.log('6. Capturing Settings page...');
  const settingsLink = page.getByRole('link', { name: 'Settings' });
  if (await settingsLink.isVisible()) {
    await settingsLink.click();
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    await page.screenshot({ path: '/tmp/06-settings.png', fullPage: true });
    console.log('   Saved: /tmp/06-settings.png');
  }

  // Toggle to dark mode if possible
  console.log('7. Capturing dark mode...');
  await page.goto(`${BASE_URL}/`);
  await page.waitForLoadState('networkidle');
  const themeToggle = page.locator('button').filter({ has: page.locator('svg') }).first();
  // Look for theme toggle button
  const toggleButtons = await page.locator('button').all();
  for (const btn of toggleButtons) {
    const ariaLabel = await btn.getAttribute('aria-label');
    if (ariaLabel && ariaLabel.includes('theme')) {
      await btn.click();
      await page.waitForTimeout(300);
      await btn.click(); // Click again to get to dark
      await page.waitForTimeout(300);
      break;
    }
  }
  // Try clicking any button that might be the theme toggle (usually has sun/moon icon)
  await page.screenshot({ path: '/tmp/07-dark-mode.png', fullPage: true });
  console.log('   Saved: /tmp/07-dark-mode.png');

  await browser.close();
  console.log('\nDone! Screenshots saved to /tmp/');
}

captureScreenshots().catch(console.error);
