"use client";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, X } from "lucide-react";
import { cn } from "@/lib/utils";

export type SortOption = "match" | "date" | "salary";

interface FilterPanelProps {
  status: string;
  minScore: string;
  search: string;
  sort: SortOption;
  onStatusChange: (value: string) => void;
  onMinScoreChange: (value: string) => void;
  onSearchChange: (value: string) => void;
  onSortChange: (value: SortOption) => void;
  onClearFilters: () => void;
}

export function FilterPanel({
  status,
  minScore,
  search,
  sort,
  onStatusChange,
  onMinScoreChange,
  onSearchChange,
  onSortChange,
  onClearFilters,
}: FilterPanelProps) {
  const hasActiveFilters =
    status !== "all" || minScore !== "0" || search !== "";

  const activeFilters = [
    status !== "all" && {
      key: "status",
      label: `Status: ${status.charAt(0).toUpperCase() + status.slice(1)}`,
      onClear: () => onStatusChange("all"),
    },
    minScore !== "0" && {
      key: "score",
      label: `Score: ${minScore}%+`,
      onClear: () => onMinScoreChange("0"),
    },
    search !== "" && {
      key: "search",
      label: `"${search}"`,
      onClear: () => onSearchChange(""),
    },
  ].filter(Boolean) as Array<{
    key: string;
    label: string;
    onClear: () => void;
  }>;

  return (
    <div className="space-y-3">
      {/* Main Filter Bar */}
      <div className="flex flex-wrap items-center gap-3 p-4 bg-surface rounded-lg border border-border">
        {/* Search Input */}
        <div className="relative flex-1 min-w-[250px] max-w-[350px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search titles, companies..."
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        {/* Status Filter */}
        <Select value={status} onValueChange={onStatusChange}>
          <SelectTrigger className="w-[130px]">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="new">New</SelectItem>
            <SelectItem value="saved">Saved</SelectItem>
            <SelectItem value="applied">Applied</SelectItem>
            <SelectItem value="interviewing">Interviewing</SelectItem>
            <SelectItem value="offered">Offered</SelectItem>
            <SelectItem value="rejected">Rejected</SelectItem>
            <SelectItem value="archived">Archived</SelectItem>
          </SelectContent>
        </Select>

        {/* Min Score Filter */}
        <Select value={minScore} onValueChange={onMinScoreChange}>
          <SelectTrigger className="w-[120px]">
            <SelectValue placeholder="Score" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="0">Any Score</SelectItem>
            <SelectItem value="50">50%+</SelectItem>
            <SelectItem value="60">60%+</SelectItem>
            <SelectItem value="70">70%+</SelectItem>
            <SelectItem value="80">80%+</SelectItem>
            <SelectItem value="90">90%+</SelectItem>
          </SelectContent>
        </Select>

        {/* Sort */}
        <Select value={sort} onValueChange={(v) => onSortChange(v as SortOption)}>
          <SelectTrigger className="w-[130px]">
            <SelectValue placeholder="Sort" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="match">Sort: Match</SelectItem>
            <SelectItem value="date">Sort: Date</SelectItem>
            <SelectItem value="salary">Sort: Salary</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Active Filter Pills */}
      {hasActiveFilters && (
        <div className="flex flex-wrap items-center gap-2 px-1">
          {activeFilters.map((filter) => (
            <button
              key={filter.key}
              onClick={filter.onClear}
              className={cn(
                "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium",
                "bg-primary/10 text-primary hover:bg-primary/20 transition-colors",
                "group cursor-pointer"
              )}
            >
              {filter.label}
              <X className="h-3 w-3 opacity-60 group-hover:opacity-100" />
            </button>
          ))}
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearFilters}
            className="text-xs text-muted-foreground hover:text-foreground h-7 px-2"
          >
            Clear all
          </Button>
        </div>
      )}
    </div>
  );
}
