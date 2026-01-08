"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/layout/Header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Profile } from "@/types";
import { Save } from "lucide-react";

export default function ProfilePage() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Form state
  const [cvText, setCvText] = useState("");
  const [targetRoles, setTargetRoles] = useState("");
  const [targetSectors, setTargetSectors] = useState("");
  const [locations, setLocations] = useState("");
  const [salaryMin, setSalaryMin] = useState("");
  const [salaryTarget, setSalaryTarget] = useState("");
  const [excludeKeywords, setExcludeKeywords] = useState("");

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const data = await api.get<Profile>("/profile");
        setProfile(data);
        setCvText(data.cv_text || "");
        setTargetRoles(data.target_roles?.join(", ") || "");
        setTargetSectors(data.target_sectors?.join(", ") || "");
        setLocations(data.locations?.join(", ") || "");
        setSalaryMin(data.salary_min?.toString() || "");
        setSalaryTarget(data.salary_target?.toString() || "");
        setExcludeKeywords(data.exclude_keywords?.join(", ") || "");
      } catch (error) {
        console.error("Failed to fetch profile:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      const updated = await api.put<Profile>("/profile", {
        cv_text: cvText,
        target_roles: targetRoles
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        target_sectors: targetSectors
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        locations: locations
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        salary_min: salaryMin ? parseInt(salaryMin) : null,
        salary_target: salaryTarget ? parseInt(salaryTarget) : null,
        exclude_keywords: excludeKeywords
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
      });
      setProfile(updated);
    } catch (error) {
      console.error("Failed to save profile:", error);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Profile" />
        <div className="p-6 space-y-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Profile" />
      <div className="flex-1 overflow-auto p-6 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Your CV</CardTitle>
            <CardDescription>
              Paste your CV text for AI matching. This will be used to calculate
              match scores against job descriptions.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              value={cvText}
              onChange={(e) => setCvText(e.target.value)}
              placeholder="Paste your CV content here (markdown or plain text)..."
              rows={12}
              className="font-mono text-sm"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Job Preferences</CardTitle>
            <CardDescription>
              Define what you&apos;re looking for. Separate multiple values with
              commas.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="roles">Target Roles</Label>
                <Input
                  id="roles"
                  value={targetRoles}
                  onChange={(e) => setTargetRoles(e.target.value)}
                  placeholder="Director, Head of Technology, CTO"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="sectors">Target Sectors</Label>
                <Input
                  id="sectors"
                  value={targetSectors}
                  onChange={(e) => setTargetSectors(e.target.value)}
                  placeholder="FinTech, Consulting, SaaS"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="locations">Locations</Label>
                <Input
                  id="locations"
                  value={locations}
                  onChange={(e) => setLocations(e.target.value)}
                  placeholder="London, Remote, UK"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="exclude">Exclude Keywords</Label>
                <Input
                  id="exclude"
                  value={excludeKeywords}
                  onChange={(e) => setExcludeKeywords(e.target.value)}
                  placeholder="junior, intern, contract"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="salaryMin">Minimum Salary (£)</Label>
                <Input
                  id="salaryMin"
                  type="number"
                  value={salaryMin}
                  onChange={(e) => setSalaryMin(e.target.value)}
                  placeholder="80000"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="salaryTarget">Target Salary (£)</Label>
                <Input
                  id="salaryTarget"
                  type="number"
                  value={salaryTarget}
                  onChange={(e) => setSalaryTarget(e.target.value)}
                  placeholder="120000"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end">
          <Button onClick={handleSave} disabled={saving} size="lg">
            <Save className="h-4 w-4 mr-2" />
            {saving ? "Saving..." : "Save Profile"}
          </Button>
        </div>
      </div>
    </div>
  );
}
