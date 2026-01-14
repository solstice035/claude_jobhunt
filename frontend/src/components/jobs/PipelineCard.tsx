"use client";

import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Job } from "@/types";

interface PipelineCardProps {
  job: Job;
}

export function PipelineCard({ job }: PipelineCardProps) {
  const getMatchVariant = (score: number) => {
    if (score >= 80) return "bg-match-high/20 text-match-high";
    if (score >= 60) return "bg-match-mid/20 text-match-mid";
    return "bg-match-low/20 text-match-low";
  };

  const formatSalary = () => {
    if (!job.salary_min && !job.salary_max) return null;
    const min = job.salary_min ? `£${(job.salary_min / 1000).toFixed(0)}k` : "";
    const max = job.salary_max ? `£${(job.salary_max / 1000).toFixed(0)}k` : "";
    if (min && max) return `${min}-${max}`;
    return min || max;
  };

  const formatRelativeDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "1d ago";
    if (diffDays < 7) return `${diffDays}d ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
    return `${Math.floor(diffDays / 30)}mo ago`;
  };

  return (
    <Link href={`/jobs/${job.id}`}>
      <Card
        className={cn(
          "cursor-pointer transition-all",
          "hover:shadow-lg hover:border-primary/50",
          "hover:-translate-y-0.5"
        )}
      >
        <CardContent className="p-3">
          {/* Title + Score Row */}
          <div className="flex items-start justify-between gap-2">
            <h4 className="font-bold text-sm text-foreground truncate flex-1">
              {job.title}
            </h4>
            <span
              className={cn(
                "flex-shrink-0 px-2 py-0.5 rounded text-xs font-bold",
                getMatchVariant(job.match_score)
              )}
            >
              {job.match_score}
            </span>
          </div>

          {/* Company + Salary Row */}
          <div className="flex items-center gap-1.5 mt-1.5 text-xs text-muted-foreground">
            <span className="font-medium text-primary/80 truncate">
              {job.company}
            </span>
            {formatSalary() && (
              <>
                <span className="text-border">·</span>
                <span className="truncate">{formatSalary()}</span>
              </>
            )}
          </div>

          {/* Status Date Row */}
          <div className="mt-2 text-xs text-muted-foreground/70">
            Updated {formatRelativeDate(job.updated_at || job.posted_at)}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
