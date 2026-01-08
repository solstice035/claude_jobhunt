import { test, expect } from '@playwright/test';

// Helper to login before each test
async function login(page: import('@playwright/test').Page) {
  await page.goto('/login');
  await page.getByLabel('Password').fill('testpassword');
  await page.getByRole('button', { name: 'Login' }).click();
  await expect(page).toHaveURL('/jobs');
}

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should have working sidebar navigation', async ({ page }) => {
    // Check that sidebar navigation exists
    const nav = page.locator('nav, aside, [role="navigation"]');
    await expect(nav.first()).toBeVisible();
  });

  test('should navigate between pages', async ({ page }) => {
    // Navigate to profile
    await page.goto('/profile');
    await expect(page.getByRole('heading', { name: 'Profile' })).toBeVisible();

    // Navigate back to jobs
    await page.goto('/jobs');
    await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible();

    // Navigate to applications if available
    await page.goto('/applications');
    // Page should load (might show empty state or content)
    await page.waitForLoadState('networkidle');
  });

  test('should have responsive layout', async ({ page }) => {
    // Test desktop layout
    await page.setViewportSize({ width: 1280, height: 720 });
    await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible();

    // Test tablet layout
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible();

    // Test mobile layout
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible();
  });
});
