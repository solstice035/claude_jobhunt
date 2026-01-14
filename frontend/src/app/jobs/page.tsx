"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { Header } from "@/components/layout/Header";
import { JobRow } from "@/components/jobs/JobRow";
import { FilterPanel, SortOption } from "@/components/jobs/FilterPanel";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { Search, RefreshCw } from "lucide-react";

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
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);

  // Filters
  const [status, setStatus] = useState("new");
  const [minScore, setMinScore] = useState("0");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortOption>("match");

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

  // Sort jobs client-side (note: sorts current page only, not full dataset)
  // TODO: Move sorting to backend API for accurate cross-page sorting
  const sortedJobs = useMemo(() => {
    const sorted = [...jobs];
    switch (sort) {
      case "match":
        return sorted.sort((a, b) => b.match_score - a.match_score);
      case "date":
        return sorted.sort(
          (a, b) =>
            new Date(b.posted_at).getTime() - new Date(a.posted_at).getTime()
        );
      case "salary":
        return sorted.sort(
          (a, b) => (b.salary_max || 0) - (a.salary_max || 0)
        );
      default:
        return sorted;
    }
  }, [jobs, sort]);

  const clearFilters = () => {
    setStatus("all");
    setMinScore("0");
    setSearch("");
    setPage(1);
  };

  return (
    <div className="flex flex-col h-full">
      <Header title="Jobs" showRefresh />

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
          {loading ? (
            <div className="divide-y divide-border">
              {[...Array(8)].map((_, i) => (
                <JobRowSkeleton key={i} />
              ))}
            </div>
          ) : sortedJobs.length === 0 ? (
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
              {sortedJobs.map((job) => (
                <JobRow key={job.id} job={job} onStatusChange={fetchJobs} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
