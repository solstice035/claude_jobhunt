export type JobStatus =
  | "new"
  | "saved"
  | "applied"
  | "interviewing"
  | "offered"
  | "rejected"
  | "archived";

export type JobSource = "adzuna";

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
  match_score: number;
  match_reasons: string[];
  status: JobStatus;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface Profile {
  id: string;
  cv_text: string;
  target_roles: string[];
  target_sectors: string[];
  locations: string[];
  salary_min?: number;
  salary_target?: number;
  exclude_keywords: string[];
  score_weights: {
    semantic: number;
    skills: number;
    seniority: number;
    location: number;
  };
}

export interface Stats {
  total_jobs: number;
  new_jobs: number;
  saved_jobs: number;
  applied_jobs: number;
  avg_match_score: number;
  jobs_by_source: Record<string, number>;
}
