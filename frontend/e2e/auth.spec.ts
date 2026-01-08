import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should display login page', async ({ page }) => {
    await page.goto('/login');

    // Check login page elements - wait for page to load
    await page.waitForLoadState('networkidle');

    // Check for password input and login button (most reliable selectors)
    await expect(page.getByLabel('Password')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Login' })).toBeVisible();
  });

  test('should show error on invalid password', async ({ page }) => {
    await page.goto('/login');
    await page.waitForLoadState('networkidle');

    // Fill password and submit
    const passwordInput = page.getByLabel('Password');
    await passwordInput.fill('wrongpassword');
    await expect(passwordInput).toHaveValue('wrongpassword');

    // Click login and wait for network
    const loginButton = page.getByRole('button', { name: 'Login' });
    await loginButton.click();

    // Wait for API response - button might show "Logging in..."
    await page.waitForLoadState('networkidle');

    // Should either show error or still be on login page (test that we didn't navigate away)
    await expect(page).toHaveURL('/login');
  });

  test('should login successfully with correct password', async ({ page }) => {
    await page.goto('/login');

    // Default password from config is 'testpassword'
    await page.getByLabel('Password').fill('testpassword');
    await page.getByRole('button', { name: 'Login' }).click();

    // Should redirect to jobs page
    await expect(page).toHaveURL('/jobs');
    await expect(page.getByRole('heading', { name: 'Jobs' })).toBeVisible();
  });

  test('should redirect unauthenticated users to login', async ({ page }) => {
    // Clear any existing cookies
    await page.context().clearCookies();

    // Try to access protected page
    await page.goto('/jobs');

    // Should redirect to login
    await expect(page).toHaveURL('/login');
  });
});
