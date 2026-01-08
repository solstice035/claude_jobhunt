/**
 * Type Definitions - Shared types matching backend API schemas
 *
 * These types mirror the Pydantic schemas from the FastAPI backend.
 * Keep in sync with backend/app/schemas/*.py
 */

/**
 * Job pipeline status - progression flow:
 * new → saved → applied → interviewing → offered/rejected → archived
 */
export type JobStatus =
  | "new"
  | "saved"
  | "applied"
  | "interviewing"
  | "offered"
  | "rejected"
  | "archived";

/** Data source for job postings */
export type JobSource = "adzuna";

/**
 * Job posting with AI match scoring.
 *
 * @property match_score - AI-calculated relevance (0-100)
 * @property match_reasons - Human-readable explanations for score
 */
export interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  salary_min?: number;
  salary_max?: number;
  description: string;
  url: string;
  source: JobSource;
  posted_at: string;
  closing_date?: string;
  /** AI match score (0-100), higher = better match */
  match_score: number;
  /** e.g., ["Good CV match", "Skills: python, react", "Location: London"] */
  match_reasons: string[];
  status: JobStatus;
  notes?: string;
  created_at: string;
  updated_at: string;
}

/**
 * User profile with job search preferences.
 *
 * @property score_weights - Customize how match scores are calculated
 */
export interface Profile {
  id: string;
  /** Full CV/resume text for AI matching */
  cv_text: string;
  /** Target job titles (e.g., ["CTO", "VP Engineering"]) */
  target_roles: string[];
  /** Preferred industries */
  target_sectors: string[];
  /** Preferred locations (e.g., ["London", "Remote"]) */
  locations: string[];
  salary_min?: number;
  salary_target?: number;
  /** Words to exclude from job matches */
  exclude_keywords: string[];
  /** Match score component weights (must sum to 1.0) */
  score_weights: {
    semantic: number; // CV-job embedding similarity (default 0.25)
    skills: number; // Keyword overlap (default 0.25)
    seniority: number; // Job level alignment (default 0.20)
    location: number; // Geographic match (default 0.15)
    salary: number; // Salary match (default 0.15)
  };
}

/** Dashboard statistics */
export interface Stats {
  total_jobs: number;
  new_jobs: number;
  saved_jobs: number;
  applied_jobs: number;
  avg_match_score: number;
  jobs_by_source: Record<string, number>;
}
