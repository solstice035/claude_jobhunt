"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Header } from "@/components/layout/Header";
import { MatchBreakdown } from "@/components/jobs/MatchBreakdown";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
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
import {
  ArrowLeft,
  ExternalLink,
  Building2,
  MapPin,
  Calendar,
  Banknote,
} from "lucide-react";

const STATUS_OPTIONS: { value: JobStatus; label: string }[] = [
  { value: "new", label: "New" },
  { value: "saved", label: "Saved" },
  { value: "applied", label: "Applied" },
  { value: "interviewing", label: "Interviewing" },
  { value: "offered", label: "Offered" },
  { value: "rejected", label: "Rejected" },
  { value: "archived", label: "Archived" },
];

export default function JobDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);

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

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Job Details" />
        <div className="p-6 space-y-4">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Job Details" />
        <div className="p-6 text-center">Job not found</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Job Details" />
      <div className="flex-1 overflow-auto p-6">
        <Button
          variant="ghost"
          className="mb-4"
          onClick={() => router.back()}
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Jobs
        </Button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main content */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-2xl">{job.title}</CardTitle>
                    <CardDescription className="flex items-center gap-4 mt-2">
                      <span className="flex items-center gap-1">
                        <Building2 className="h-4 w-4" />
                        {job.company}
                      </span>
                      <span className="flex items-center gap-1">
                        <MapPin className="h-4 w-4" />
                        {job.location}
                      </span>
                    </CardDescription>
                  </div>
                  <Button asChild>
                    <a
                      href={job.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Apply
                      <ExternalLink className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </div>

                <div className="flex flex-wrap gap-3 mt-4">
                  {formatSalary() && (
                    <Badge variant="outline" className="text-sm">
                      <Banknote className="h-3 w-3 mr-1" />
                      {formatSalary()}
                    </Badge>
                  )}
                  <Badge variant="outline" className="text-sm">
                    <Calendar className="h-3 w-3 mr-1" />
                    Posted{" "}
                    {new Date(job.posted_at).toLocaleDateString("en-GB")}
                  </Badge>
                  <Badge variant="outline" className="text-sm">
                    {job.source}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <h3 className="font-semibold mb-3">Job Description</h3>
                <div className="prose prose-sm max-w-none whitespace-pre-wrap text-muted-foreground">
                  {job.description}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Match Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <MatchBreakdown
                  score={job.match_score}
                  reasons={job.match_reasons}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Status</CardTitle>
              </CardHeader>
              <CardContent>
                <Select
                  value={job.status}
                  onValueChange={handleStatusChange}
                  disabled={saving}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STATUS_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Notes</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add your notes..."
                  rows={4}
                />
                <Button
                  onClick={handleSaveNotes}
                  disabled={saving || notes === (job.notes || "")}
                  className="w-full"
                >
                  {saving ? "Saving..." : "Save Notes"}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
