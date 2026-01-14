"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Header } from "@/components/layout/Header";
import { MatchScoreCard } from "@/components/jobs/MatchScoreCard";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Job, JobStatus } from "@/types";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  ExternalLink,
  MapPin,
  Banknote,
  Clock,
  Briefcase,
  Copy,
  Check,
} from "lucide-react";

const STATUS_OPTIONS: { value: JobStatus; label: string; color: string }[] = [
  { value: "new", label: "New", color: "bg-muted-foreground" },
  { value: "saved", label: "Saved", color: "bg-status-saved" },
  { value: "applied", label: "Applied", color: "bg-status-applied" },
  { value: "interviewing", label: "Interviewing", color: "bg-status-interviewing" },
  { value: "offered", label: "Offered", color: "bg-status-offered" },
  { value: "rejected", label: "Rejected", color: "bg-status-rejected" },
  { value: "archived", label: "Archived", color: "bg-status-archived" },
];

export default function JobDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const fetchJob = async () => {
      try {
        const data = await api.get<Job>(`/jobs/${params.id}`);
        setJob(data);
        setNotes(data.notes || "");
      } catch (error) {
        console.error("Failed to fetch job:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchJob();
  }, [params.id]);

  const handleStatusChange = async (status: JobStatus) => {
    if (!job) return;
    setSaving(true);
    try {
      const updated = await api.patch<Job>(`/jobs/${job.id}`, { status });
      setJob(updated);
    } finally {
      setSaving(false);
    }
  };

  const handleSaveNotes = async () => {
    if (!job) return;
    setSaving(true);
    try {
      const updated = await api.patch<Job>(`/jobs/${job.id}`, { notes });
      setJob(updated);
    } finally {
      setSaving(false);
    }
  };

  const handleCopyLink = async () => {
    if (!job?.url) return;
    await navigator.clipboard.writeText(job.url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatSalary = () => {
    if (!job?.salary_min && !job?.salary_max) return null;
    const min = job.salary_min
      ? `£${(job.salary_min / 1000).toFixed(0)}k`
      : "";
    const max = job.salary_max
      ? `£${(job.salary_max / 1000).toFixed(0)}k`
      : "";
    if (min && max) return `${min} - ${max}`;
    return min || max;
  };

  const formatRelativeDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    return date.toLocaleDateString("en-GB");
  };

  const getStatusColor = (status: JobStatus) => {
    return STATUS_OPTIONS.find((opt) => opt.value === status)?.color || "bg-muted-foreground";
  };

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Job Details" />
        <div className="flex-1 overflow-auto p-6">
          <Skeleton className="h-8 w-32 mb-6" />
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <Skeleton className="h-64 w-full rounded-xl" />
              <Skeleton className="h-96 w-full rounded-xl" />
            </div>
            <div className="space-y-6">
              <Skeleton className="h-48 w-full rounded-xl" />
              <Skeleton className="h-32 w-full rounded-xl" />
              <Skeleton className="h-48 w-full rounded-xl" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Job Details" />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-foreground mb-2">Job not found</h3>
            <p className="text-sm text-muted-foreground mb-4">
              This job may have been removed or archived.
            </p>
            <Button variant="outline" onClick={() => router.back()}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Go Back
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Job Details" />
      <div className="flex-1 overflow-auto p-6">
        {/* Back Button */}
        <Button
          variant="ghost"
          size="sm"
          className="mb-6 text-primary hover:text-primary/80"
          onClick={() => router.back()}
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Jobs
        </Button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content - Left 2/3 */}
          <div className="lg:col-span-2 space-y-6">
            {/* Job Header Card */}
            <Card>
              <CardHeader className="pb-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-2">
                    <h1 className="text-3xl font-black tracking-tight text-foreground">
                      {job.title}
                    </h1>
                    <div className="flex items-center gap-2 text-lg">
                      <span className="font-semibold text-primary">
                        {job.company}
                      </span>
                    </div>
                  </div>
                  <Button asChild size="lg" className="shrink-0">
                    <a
                      href={job.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Apply Now
                      <ExternalLink className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </div>

                {/* Meta Row */}
                <div className="flex flex-wrap items-center gap-4 mt-4 text-sm text-muted-foreground">
                  <span className="flex items-center gap-1.5">
                    <MapPin className="h-4 w-4" />
                    {job.location}
                  </span>
                  {formatSalary() && (
                    <span className="flex items-center gap-1.5">
                      <Banknote className="h-4 w-4" />
                      {formatSalary()}
                    </span>
                  )}
                  <span className="flex items-center gap-1.5">
                    <Clock className="h-4 w-4" />
                    Posted {formatRelativeDate(job.posted_at)}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Briefcase className="h-4 w-4" />
                    {job.source}
                  </span>
                </div>
              </CardHeader>
            </Card>

            {/* Description Card */}
            <Card>
              <CardHeader className="border-b border-border pb-4">
                <CardTitle className="text-lg font-bold text-foreground">
                  Job Description
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="prose prose-sm max-w-none text-muted-foreground leading-relaxed whitespace-pre-wrap">
                  {job.description}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar - Right 1/3 */}
          <div className="space-y-6">
            {/* Match Score Card */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-bold text-foreground">
                  Match Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <MatchScoreCard
                  score={job.match_score}
                  reasons={job.match_reasons}
                />
              </CardContent>
            </Card>

            {/* Status Card */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-bold text-foreground">
                  Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Select
                  value={job.status}
                  onValueChange={handleStatusChange}
                  disabled={saving}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue>
                      <div className="flex items-center gap-2">
                        <span
                          className={cn(
                            "h-2.5 w-2.5 rounded-full",
                            getStatusColor(job.status)
                          )}
                        />
                        {STATUS_OPTIONS.find((opt) => opt.value === job.status)?.label}
                      </div>
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {STATUS_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        <div className="flex items-center gap-2">
                          <span
                            className={cn(
                              "h-2.5 w-2.5 rounded-full",
                              opt.color
                            )}
                          />
                          {opt.label}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            {/* Notes Card */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-bold text-foreground">
                  Notes
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add your notes about this opportunity..."
                  rows={5}
                  className="resize-none font-mono text-sm"
                />
                <Button
                  onClick={handleSaveNotes}
                  disabled={saving || notes === (job.notes || "")}
                  className="w-full"
                  variant={notes !== (job.notes || "") ? "default" : "outline"}
                >
                  {saving ? "Saving..." : "Save Notes"}
                </Button>
              </CardContent>
            </Card>

            {/* Quick Actions Card */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg font-bold text-foreground">
                  Quick Actions
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  asChild
                >
                  <a
                    href={job.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Open Original Listing
                  </a>
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  onClick={handleCopyLink}
                >
                  {copied ? (
                    <>
                      <Check className="h-4 w-4 mr-2 text-match-high" />
                      Link Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4 mr-2" />
                      Copy Link
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
