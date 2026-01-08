"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Header } from "@/components/layout/Header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { Building2, MapPin } from "lucide-react";

const PIPELINE_STAGES = [
  { key: "saved", label: "Saved", color: "bg-blue-500" },
  { key: "applied", label: "Applied", color: "bg-yellow-500" },
  { key: "interviewing", label: "Interviewing", color: "bg-purple-500" },
  { key: "offered", label: "Offered", color: "bg-green-500" },
  { key: "rejected", label: "Rejected", color: "bg-red-500" },
] as const;

export default function ApplicationsPage() {
  const [jobsByStatus, setJobsByStatus] = useState<Record<string, Job[]>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const grouped: Record<string, Job[]> = {};
        for (const stage of PIPELINE_STAGES) {
          const data = await api.get<{ jobs: Job[] }>(
            `/jobs?status=${stage.key}&per_page=50`
          );
          grouped[stage.key] = data.jobs;
        }
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
        <div className="p-6 grid grid-cols-5 gap-4">
          {PIPELINE_STAGES.map((stage) => (
            <Skeleton key={stage.key} className="h-96 w-full" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Applications" />
      <div className="flex-1 overflow-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 min-h-[calc(100vh-8rem)]">
          {PIPELINE_STAGES.map((stage) => (
            <div key={stage.key} className="flex flex-col">
              <div className="flex items-center gap-2 mb-3">
                <div className={`w-3 h-3 rounded-full ${stage.color}`} />
                <h3 className="font-semibold">{stage.label}</h3>
                <Badge variant="secondary" className="ml-auto">
                  {jobsByStatus[stage.key]?.length || 0}
                </Badge>
              </div>

              <div className="flex-1 space-y-3 overflow-auto">
                {jobsByStatus[stage.key]?.map((job) => (
                  <Link key={job.id} href={`/jobs/${job.id}`}>
                    <Card className="cursor-pointer hover:shadow-md transition-shadow">
                      <CardContent className="p-3">
                        <h4 className="font-medium text-sm truncate">
                          {job.title}
                        </h4>
                        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                          <Building2 className="h-3 w-3" />
                          <span className="truncate">{job.company}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                          <MapPin className="h-3 w-3" />
                          <span className="truncate">{job.location}</span>
                        </div>
                        <div className="flex items-center justify-between mt-2">
                          <Badge
                            variant={
                              job.match_score >= 80
                                ? "default"
                                : job.match_score >= 60
                                ? "secondary"
                                : "outline"
                            }
                            className="text-xs"
                          >
                            {job.match_score}%
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                ))}

                {(!jobsByStatus[stage.key] ||
                  jobsByStatus[stage.key].length === 0) && (
                  <div className="text-center py-8 text-sm text-muted-foreground border-2 border-dashed rounded-lg">
                    No jobs
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
