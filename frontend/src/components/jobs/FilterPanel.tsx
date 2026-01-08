"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search } from "lucide-react";

interface FilterPanelProps {
  status: string;
  minScore: string;
  search: string;
  onStatusChange: (value: string) => void;
  onMinScoreChange: (value: string) => void;
  onSearchChange: (value: string) => void;
}

export function FilterPanel({
  status,
  minScore,
  search,
  onStatusChange,
  onMinScoreChange,
  onSearchChange,
}: FilterPanelProps) {
  return (
    <div className="flex flex-wrap gap-4 p-4 bg-card border rounded-lg">
      <div className="flex-1 min-w-[200px]">
        <Label htmlFor="search" className="text-xs text-muted-foreground">
          Search
        </Label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="search"
            placeholder="Search jobs..."
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <div className="w-[150px]">
        <Label htmlFor="status" className="text-xs text-muted-foreground">
          Status
        </Label>
        <Select value={status} onValueChange={onStatusChange}>
          <SelectTrigger id="status">
            <SelectValue placeholder="All" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All</SelectItem>
            <SelectItem value="new">New</SelectItem>
            <SelectItem value="saved">Saved</SelectItem>
            <SelectItem value="applied">Applied</SelectItem>
            <SelectItem value="interviewing">Interviewing</SelectItem>
            <SelectItem value="archived">Archived</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="w-[150px]">
        <Label htmlFor="score" className="text-xs text-muted-foreground">
          Min Score
        </Label>
        <Select value={minScore} onValueChange={onMinScoreChange}>
          <SelectTrigger id="score">
            <SelectValue placeholder="Any" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="0">Any</SelectItem>
            <SelectItem value="50">50%+</SelectItem>
            <SelectItem value="60">60%+</SelectItem>
            <SelectItem value="70">70%+</SelectItem>
            <SelectItem value="80">80%+</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
