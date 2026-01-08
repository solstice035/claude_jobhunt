const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async fetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      if (response.status === 401) {
        if (typeof window !== "undefined") {
          window.location.href = "/login";
        }
      }
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }

  get<T>(endpoint: string) {
    return this.fetch<T>(endpoint);
  }

  post<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  patch<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  put<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }
}

export const api = new ApiClient(API_URL);
