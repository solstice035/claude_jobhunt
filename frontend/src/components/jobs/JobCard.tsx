import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MatchBadge } from "./MatchBadge";
import { Job, JobStatus } from "@/types";
import {
  Bookmark,
  Archive,
  ExternalLink,
  MapPin,
  Building2,
  Calendar,
} from "lucide-react";
import { api } from "@/lib/api";

interface JobCardProps {
  job: Job;
  onStatusChange: () => void;
}

export function JobCard({ job, onStatusChange }: JobCardProps) {
  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
    });
  };

  const formatSalary = () => {
    if (!job.salary_min && !job.salary_max) return null;
    const min = job.salary_min ? `£${(job.salary_min / 1000).toFixed(0)}k` : "";
    const max = job.salary_max ? `£${(job.salary_max / 1000).toFixed(0)}k` : "";
    if (min && max) return `${min} - ${max}`;
    return min || max;
  };

  const handleStatusChange = async (status: JobStatus) => {
    await api.patch(`/jobs/${job.id}`, { status });
    onStatusChange();
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <Link href={`/jobs/${job.id}`}>
              <h3 className="font-semibold text-lg hover:text-primary truncate">
                {job.title}
              </h3>
            </Link>
            <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Building2 className="h-3 w-3" />
                {job.company}
              </span>
              <span className="flex items-center gap-1">
                <MapPin className="h-3 w-3" />
                {job.location}
              </span>
            </div>
          </div>
          <MatchBadge score={job.match_score} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2 mb-3">
          {job.match_reasons.slice(0, 3).map((reason, idx) => (
            <Badge key={idx} variant="secondary" className="text-xs">
              {reason}
            </Badge>
          ))}
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            {formatSalary() && <span>{formatSalary()}</span>}
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {formatDate(job.posted_at)}
            </span>
            <Badge variant="outline" className="text-xs">
              {job.source}
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            {job.status === "new" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleStatusChange("saved")}
                title="Save"
              >
                <Bookmark className="h-4 w-4" />
              </Button>
            )}
            {job.status !== "archived" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleStatusChange("archived")}
                title="Archive"
              >
                <Archive className="h-4 w-4" />
              </Button>
            )}
            <Button variant="ghost" size="sm" asChild>
              <a href={job.url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4" />
              </a>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
