"use client";

import { useEffect, useState, useCallback } from "react";
import { Header } from "@/components/layout/Header";
import { JobRow } from "@/components/jobs/JobRow";
import { FilterPanel, SortOption } from "@/components/jobs/FilterPanel";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { Search, AlertCircle } from "lucide-react";
import { toast } from "sonner";

interface JobsResponse {
  jobs: Job[];
  total: number;
  page: number;
  per_page: number;
}

function JobRowSkeleton() {
  return (
    <div className="flex items-center gap-4 px-4 py-3">
      <Skeleton className="h-10 w-14 rounded-md" />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-2/3" />
      </div>
      <Skeleton className="h-4 w-24 hidden md:block" />
      <Skeleton className="h-4 w-24 hidden lg:block" />
      <Skeleton className="h-4 w-16 hidden sm:block" />
      <Skeleton className="h-6 w-20 rounded-full hidden sm:block" />
    </div>
  );
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);

  // Filters
  const [status, setStatus] = useState("new");
  const [minScore, setMinScore] = useState("0");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sort, setSort] = useState<SortOption>("match");

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  const fetchJobs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (status !== "all") params.set("status", status);
      if (minScore !== "0") params.set("score_min", minScore);
      if (debouncedSearch) params.set("search", debouncedSearch);
      params.set("sort", sort);
      params.set("page", String(page));

      const data = await api.get<JobsResponse>(`/jobs?${params}`);
      setJobs(data.jobs);
      setTotal(data.total);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to fetch jobs";
      setError(message);
      toast.error("Failed to load jobs");
    } finally {
      setLoading(false);
    }
  }, [status, minScore, debouncedSearch, sort, page]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const clearFilters = () => {
    setStatus("all");
    setMinScore("0");
    setSearch("");
    setPage(1);
  };

  return (
    <div className="flex flex-col h-full">
      <Header title="Jobs" showRefresh onRefreshComplete={fetchJobs} />

      <div className="flex-1 overflow-auto p-6 space-y-4">
        {/* Filter Panel */}
        <FilterPanel
          status={status}
          minScore={minScore}
          search={search}
          sort={sort}
          onStatusChange={(v) => {
            setStatus(v);
            setPage(1);
          }}
          onMinScoreChange={(v) => {
            setMinScore(v);
            setPage(1);
          }}
          onSearchChange={setSearch}
          onSortChange={setSort}
          onClearFilters={clearFilters}
        />

        {/* Job Count */}
        <div className="flex items-center justify-between px-1">
          <span className="text-sm font-medium text-muted-foreground">
            {loading ? "Loading..." : `${total} jobs`}
          </span>
        </div>

        {/* Job List */}
        <div className="bg-card rounded-xl border border-border overflow-hidden">
          {/* Table Header */}
          <div className="hidden sm:flex items-center gap-4 px-4 py-2 border-b border-border bg-surface text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            <div className="w-14 text-center">Match</div>
            <div className="flex-1">Job</div>
            <div className="hidden md:block w-28">Location</div>
            <div className="hidden lg:block w-28">Salary</div>
            <div className="w-16 text-right">Posted</div>
            <div className="w-24">Status</div>
            <div className="w-20"></div>
          </div>

          {/* Job Rows */}
          {error ? (
            <div className="flex flex-col items-center justify-center py-16 px-4">
              <div className="w-12 h-12 rounded-full bg-destructive/10 flex items-center justify-center mb-4">
                <AlertCircle className="h-6 w-6 text-destructive" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-1">
                Failed to load jobs
              </h3>
              <p className="text-sm text-muted-foreground mb-4">{error}</p>
              <Button variant="outline" onClick={fetchJobs}>
                Try again
              </Button>
            </div>
          ) : loading ? (
            <div className="divide-y divide-border">
              {[...Array(8)].map((_, i) => (
                <JobRowSkeleton key={i} />
              ))}
            </div>
          ) : jobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 px-4">
              <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                <Search className="h-6 w-6 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-1">
                No jobs found
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                Try adjusting your filters or search query
              </p>
              <Button variant="outline" onClick={clearFilters}>
                Clear filters
              </Button>
            </div>
          ) : (
            <div className="divide-y divide-border/50">
              {jobs.map((job) => (
                <JobRow key={job.id} job={job} onStatusChange={fetchJobs} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
