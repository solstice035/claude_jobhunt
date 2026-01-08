/**
 * API Client - HTTP client for backend communication
 *
 * Features:
 * - Automatic JSON serialization/deserialization
 * - Cookie-based authentication (credentials: include)
 * - Auto-redirect to /login on 401 responses
 * - Type-safe generic methods
 *
 * @example
 * // GET request with typed response
 * const { jobs } = await api.get<{ jobs: Job[] }>('/jobs?status=new');
 *
 * // PATCH request
 * await api.patch<Job>(`/jobs/${id}`, { status: 'saved' });
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Type-safe API client with automatic auth handling.
 */
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  /**
   * Core fetch wrapper with error handling.
   *
   * @param endpoint - API endpoint (e.g., '/jobs')
   * @param options - Standard RequestInit options
   * @returns Parsed JSON response of type T
   * @throws Error on non-2xx responses
   */
  async fetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      credentials: "include", // Include httpOnly cookies for auth
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      // Redirect to login on authentication failure
      if (response.status === 401) {
        if (typeof window !== "undefined") {
          window.location.href = "/login";
        }
      }
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }

  /** GET request */
  get<T>(endpoint: string) {
    return this.fetch<T>(endpoint);
  }

  /** POST request with JSON body */
  post<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  /** PATCH request for partial updates */
  patch<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  /** PUT request for full replacements */
  put<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }
}

/** Pre-configured API client instance */
export const api = new ApiClient(API_URL);
