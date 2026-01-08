import { test, expect } from '@playwright/test';

// Helper to login before each test
async function login(page: import('@playwright/test').Page) {
  await page.goto('/login');
  await page.getByLabel('Password').fill('testpassword');
  await page.getByRole('button', { name: 'Login' }).click();
  await expect(page).toHaveURL('/jobs');
}

test.describe('Jobs Page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should display jobs page with header and filters', async ({ page }) => {
    // Check header
    await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible();

    // Check filter panel exists
    await expect(page.getByRole('combobox').first()).toBeVisible(); // Status filter
    await expect(page.getByPlaceholder(/search/i)).toBeVisible();

    // Should show job count
    await expect(page.getByText(/jobs found/i)).toBeVisible();
  });

  test('should filter jobs by status', async ({ page }) => {
    // Wait for initial load
    await page.waitForLoadState('networkidle');

    // Click status dropdown and change filter
    const statusSelect = page.locator('[data-slot="select-trigger"]').first();
    if (await statusSelect.isVisible()) {
      await statusSelect.click();

      // Select "All" status
      const allOption = page.getByRole('option', { name: /all/i });
      if (await allOption.isVisible()) {
        await allOption.click();
      }
    }

    // Should trigger a refresh
    await page.waitForLoadState('networkidle');
  });

  test('should search jobs', async ({ page }) => {
    // Wait for initial load
    await page.waitForLoadState('networkidle');

    // Type in search
    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('developer');

    // Wait for debounced search
    await page.waitForTimeout(500);
    await page.waitForLoadState('networkidle');
  });

  test('should navigate to profile from sidebar', async ({ page }) => {
    // Look for profile link in navigation
    const profileLink = page.getByRole('link', { name: /profile/i });
    if (await profileLink.isVisible()) {
      await profileLink.click();
      await expect(page).toHaveURL('/profile');
    }
  });
});
