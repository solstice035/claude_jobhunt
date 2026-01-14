"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Job, JobStatus } from "@/types";
import {
  Bookmark,
  Archive,
  ExternalLink,
  MapPin,
  Banknote,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

interface JobRowProps {
  job: Job;
  onStatusChange: () => void;
}

const statusConfig: Record<
  JobStatus,
  { label: string; color: string; bgColor: string }
> = {
  new: {
    label: "New",
    color: "text-foreground",
    bgColor: "bg-muted",
  },
  saved: {
    label: "Saved",
    color: "text-status-saved",
    bgColor: "bg-status-saved/20",
  },
  applied: {
    label: "Applied",
    color: "text-status-applied",
    bgColor: "bg-status-applied/20",
  },
  interviewing: {
    label: "Interview",
    color: "text-status-interviewing",
    bgColor: "bg-status-interviewing/20",
  },
  offered: {
    label: "Offered",
    color: "text-status-offered",
    bgColor: "bg-status-offered/20",
  },
  rejected: {
    label: "Rejected",
    color: "text-status-rejected",
    bgColor: "bg-status-rejected/20",
  },
  archived: {
    label: "Archived",
    color: "text-status-archived",
    bgColor: "bg-status-archived/20",
  },
};

export function JobRow({ job, onStatusChange }: JobRowProps) {
  const formatRelativeDate = (date: string) => {
    const now = new Date();
    const posted = new Date(date);
    const diffDays = Math.floor(
      (now.getTime() - posted.getTime()) / (1000 * 60 * 60 * 24)
    );

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "1d ago";
    if (diffDays < 7) return `${diffDays}d ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
    return `${Math.floor(diffDays / 30)}mo ago`;
  };

  const formatSalary = () => {
    if (!job.salary_min && !job.salary_max) return null;
    const min = job.salary_min
      ? `£${(job.salary_min / 1000).toFixed(0)}k`
      : "";
    const max = job.salary_max
      ? `£${(job.salary_max / 1000).toFixed(0)}k`
      : "";
    if (min && max) return `${min}-${max}`;
    return min || max;
  };

  const getMatchColor = (score: number) => {
    if (score >= 80) return "text-match-high bg-match-high/20";
    if (score >= 60) return "text-match-mid bg-match-mid/20";
    return "text-match-low bg-match-low/20";
  };

  const handleStatusChange = async (
    e: React.MouseEvent,
    status: JobStatus
  ) => {
    e.preventDefault();
    e.stopPropagation();
    await api.patch(`/jobs/${job.id}`, { status });
    onStatusChange();
  };

  const handleExternalClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const status = statusConfig[job.status];
  const salary = formatSalary();

  return (
    <Link href={`/jobs/${job.id}`} className="block group">
      <div
        className={cn(
          "flex items-center gap-4 px-4 py-3 rounded-lg transition-all",
          "border-l-[3px] border-l-transparent",
          "hover:bg-surface-hover hover:border-l-primary",
          "cursor-pointer"
        )}
      >
        {/* Match Score */}
        <div
          className={cn(
            "flex-shrink-0 w-14 h-10 flex items-center justify-center rounded-md font-black text-lg",
            getMatchColor(job.match_score)
          )}
        >
          {job.match_score}
        </div>

        {/* Title & Company */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-bold text-foreground truncate">
              {job.title}
            </span>
            <span className="text-muted-foreground">·</span>
            <span className="font-semibold text-primary truncate">
              {job.company}
            </span>
          </div>
        </div>

        {/* Location */}
        <div className="hidden md:flex items-center gap-1.5 w-28 text-sm font-medium text-muted-foreground">
          <MapPin className="h-3.5 w-3.5 flex-shrink-0" />
          <span className="truncate">{job.location}</span>
        </div>

        {/* Salary */}
        <div className="hidden lg:flex items-center gap-1.5 w-28 text-sm font-medium text-muted-foreground">
          {salary ? (
            <>
              <Banknote className="h-3.5 w-3.5 flex-shrink-0" />
              <span>{salary}</span>
            </>
          ) : (
            <span className="text-text-tertiary">—</span>
          )}
        </div>

        {/* Posted Date */}
        <div className="hidden sm:block w-16 text-sm text-text-tertiary text-right">
          {formatRelativeDate(job.posted_at)}
        </div>

        {/* Status */}
        <div className="hidden sm:flex items-center gap-2 w-24">
          <span
            className={cn(
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold",
              status.color,
              status.bgColor
            )}
          >
            <span
              className={cn("w-1.5 h-1.5 rounded-full", status.color)}
              style={{ backgroundColor: "currentColor" }}
            />
            {status.label}
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {job.status === "new" && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={(e) => handleStatusChange(e, "saved")}
              title="Save"
              className="h-8 w-8"
            >
              <Bookmark className="h-4 w-4" />
            </Button>
          )}
          {job.status !== "archived" && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={(e) => handleStatusChange(e, "archived")}
              title="Archive"
              className="h-8 w-8"
            >
              <Archive className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon-sm"
            asChild
            className="h-8 w-8"
            onClick={handleExternalClick}
          >
            <a href={job.url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4" />
            </a>
          </Button>
        </div>
      </div>
    </Link>
  );
}
