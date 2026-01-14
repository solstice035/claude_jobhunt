"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/layout/Header";
import { PipelineCard } from "@/components/jobs/PipelineCard";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { cn } from "@/lib/utils";
import { Inbox } from "lucide-react";

const PIPELINE_STAGES = [
  { key: "saved", label: "Saved", color: "bg-status-saved", borderColor: "border-t-status-saved", badgeColor: "bg-status-saved/20", emptyText: "save them" },
  { key: "applied", label: "Applied", color: "bg-status-applied", borderColor: "border-t-status-applied", badgeColor: "bg-status-applied/20", emptyText: "mark them as applied" },
  { key: "interviewing", label: "Interviewing", color: "bg-status-interviewing", borderColor: "border-t-status-interviewing", badgeColor: "bg-status-interviewing/20", emptyText: "get interviews" },
  { key: "offered", label: "Offered", color: "bg-status-offered", borderColor: "border-t-status-offered", badgeColor: "bg-status-offered/20", emptyText: "receive offers" },
  { key: "rejected", label: "Rejected", color: "bg-status-rejected", borderColor: "border-t-status-rejected", badgeColor: "bg-status-rejected/20", emptyText: "get rejections" },
] as const;

function ColumnSkeleton() {
  return (
    <div className="flex flex-col min-w-[260px]">
      <div className="flex items-center gap-2 p-3 bg-surface rounded-t-lg border-t-[3px] border-t-muted">
        <Skeleton className="h-4 w-20" />
        <Skeleton className="h-5 w-8 rounded-full ml-auto" />
      </div>
      <div className="flex-1 p-2 space-y-3 bg-base rounded-b-lg">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-24 w-full rounded-lg" />
        ))}
      </div>
    </div>
  );
}

export default function ApplicationsPage() {
  const [jobsByStatus, setJobsByStatus] = useState<Record<string, Job[]>>({});
  const [loading, setLoading] = useState(true);

  const totalJobs = Object.values(jobsByStatus).reduce(
    (sum, jobs) => sum + jobs.length,
    0
  );

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const results = await Promise.all(
          PIPELINE_STAGES.map((stage) =>
            api.get<{ jobs: Job[] }>(`/jobs?status=${stage.key}&per_page=50`)
          )
        );

        const grouped: Record<string, Job[]> = {};
        PIPELINE_STAGES.forEach((stage, index) => {
          grouped[stage.key] = results[index].jobs;
        });
        setJobsByStatus(grouped);
      } catch (error) {
        console.error("Failed to fetch jobs:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchJobs();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Applications" />
        <div className="flex-1 overflow-auto p-6">
          <div className="flex gap-4 min-h-[calc(100vh-10rem)]">
            {PIPELINE_STAGES.map((stage) => (
              <ColumnSkeleton key={stage.key} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Applications" />
      <div className="flex-1 overflow-auto p-6">
        {/* Summary Bar */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-muted-foreground">
              {totalJobs} jobs in pipeline
            </span>
            <div className="flex items-center gap-3">
              {PIPELINE_STAGES.map((stage) => {
                const count = jobsByStatus[stage.key]?.length || 0;
                if (count === 0) return null;
                return (
                  <div key={stage.key} className="flex items-center gap-1.5">
                    <span className={cn("h-2 w-2 rounded-full", stage.color)} />
                    <span className="text-xs text-muted-foreground">
                      {count} {stage.label.toLowerCase()}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Pipeline Columns */}
        <div className="flex gap-4 min-h-[calc(100vh-12rem)] overflow-x-auto pb-4">
          {PIPELINE_STAGES.map((stage) => {
            const jobs = jobsByStatus[stage.key] || [];
            const isEmpty = jobs.length === 0;

            return (
              <div
                key={stage.key}
                className="flex flex-col min-w-[260px] w-[260px] flex-shrink-0"
              >
                {/* Column Header */}
                <div
                  className={cn(
                    "flex items-center gap-2 px-3 py-2.5 bg-surface rounded-t-lg",
                    "border-t-[3px]",
                    stage.borderColor
                  )}
                >
                  <h3 className="text-sm font-bold uppercase tracking-wide text-foreground">
                    {stage.label}
                  </h3>
                  <span
                    className={cn(
                      "ml-auto px-2 py-0.5 rounded-full text-xs font-semibold",
                      jobs.length > 0
                        ? `${stage.badgeColor} text-foreground`
                        : "bg-muted text-muted-foreground"
                    )}
                  >
                    {jobs.length}
                  </span>
                </div>

                {/* Column Body */}
                <div
                  className={cn(
                    "flex-1 p-2 space-y-3 overflow-y-auto",
                    "bg-base/50 rounded-b-lg border border-t-0 border-border"
                  )}
                >
                  {jobs.map((job) => (
                    <PipelineCard key={job.id} job={job} />
                  ))}

                  {isEmpty && (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center mb-3">
                        <Inbox className="h-5 w-5 text-muted-foreground" />
                      </div>
                      <p className="text-sm text-muted-foreground">
                        No jobs
                      </p>
                      <p className="text-xs text-muted-foreground/70 mt-1">
                        Jobs will appear here when you {stage.emptyText}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
