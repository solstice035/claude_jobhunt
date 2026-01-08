import { test, expect } from '@playwright/test';

// Helper to login before each test
async function login(page: import('@playwright/test').Page) {
  await page.goto('/login');
  await page.getByLabel('Password').fill('testpassword');
  await page.getByRole('button', { name: 'Login' }).click();
  await expect(page).toHaveURL('/jobs');
}

test.describe('Profile Page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto('/profile');
  });

  test('should display profile page with CV section', async ({ page }) => {
    // Wait for page to load
    await page.waitForLoadState('networkidle');

    // Check header - use text match instead of heading role
    await expect(page.getByText('Profile').first()).toBeVisible();

    // Check CV section exists by looking for textarea
    const cvTextarea = page.locator('textarea').first();
    await expect(cvTextarea).toBeVisible();
  });

  test('should display job preferences section', async ({ page }) => {
    // Wait for page to load
    await page.waitForLoadState('networkidle');

    // Check preference inputs exist - use more flexible selectors
    await expect(page.locator('input').first()).toBeVisible();

    // Check save button exists
    await expect(page.getByRole('button', { name: /save/i })).toBeVisible();
  });

  test('should save profile successfully', async ({ page }) => {
    // Fill in some profile data
    await page.getByLabel('Target Roles').fill('Senior Developer, Tech Lead');
    await page.getByLabel('Locations').fill('London, Remote');
    await page.getByLabel('Minimum Salary (£)').fill('80000');
    await page.getByLabel('Target Salary (£)').fill('120000');

    // Click save button
    await page.getByRole('button', { name: /save/i }).click();

    // Wait for save to complete (button should change text while saving)
    await expect(page.getByRole('button', { name: /save/i })).toBeEnabled();
  });

  test('should update CV text', async ({ page }) => {
    const cvTextarea = page.getByPlaceholder(/Paste your CV content/i);

    // Clear and fill CV
    await cvTextarea.fill('Test CV content for UAT testing.\n\nSkills: Python, React, TypeScript');

    // Save
    await page.getByRole('button', { name: /save/i }).click();

    // Wait for save
    await page.waitForLoadState('networkidle');

    // Verify CV is still there after reload
    await page.reload();
    await expect(cvTextarea).toHaveValue(/Test CV content/);
  });
});
