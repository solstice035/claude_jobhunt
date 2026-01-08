"use client";

import { useEffect, useState, useCallback } from "react";
import { Header } from "@/components/layout/Header";
import { JobCard } from "@/components/jobs/JobCard";
import { FilterPanel } from "@/components/jobs/FilterPanel";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Job } from "@/types";

interface JobsResponse {
  jobs: Job[];
  total: number;
  page: number;
  per_page: number;
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);

  // Filters
  const [status, setStatus] = useState("new");
  const [minScore, setMinScore] = useState("0");
  const [search, setSearch] = useState("");

  const fetchJobs = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (status !== "all") params.set("status", status);
      if (minScore !== "0") params.set("score_min", minScore);
      if (search) params.set("search", search);
      params.set("page", String(page));

      const data = await api.get<JobsResponse>(`/jobs?${params}`);
      setJobs(data.jobs);
      setTotal(data.total);
    } catch (error) {
      console.error("Failed to fetch jobs:", error);
    } finally {
      setLoading(false);
    }
  }, [status, minScore, search, page]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  return (
    <div className="flex flex-col h-full">
      <Header title="Jobs" showRefresh />
      <div className="flex-1 overflow-auto p-6 space-y-4">
        <FilterPanel
          status={status}
          minScore={minScore}
          search={search}
          onStatusChange={(v) => {
            setStatus(v);
            setPage(1);
          }}
          onMinScoreChange={(v) => {
            setMinScore(v);
            setPage(1);
          }}
          onSearchChange={setSearch}
        />

        <div className="text-sm text-muted-foreground">{total} jobs found</div>

        {loading ? (
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-32 w-full" />
            ))}
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            No jobs found. Try adjusting your filters.
          </div>
        ) : (
          <div className="space-y-4">
            {jobs.map((job) => (
              <JobCard key={job.id} job={job} onStatusChange={fetchJobs} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
