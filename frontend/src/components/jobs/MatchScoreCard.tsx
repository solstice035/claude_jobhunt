"use client";

import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";

interface MatchScoreCardProps {
  score: number;
  reasons: string[];
}

interface BreakdownItem {
  label: string;
  value: number;
  color: string;
}

// Parse match reasons to extract breakdown values
function parseReasons(reasons: string[]): BreakdownItem[] {
  const items: BreakdownItem[] = [];

  // Default breakdown structure
  const defaultBreakdown = [
    { label: "Skills Match", value: 0, color: "bg-chart-1" },
    { label: "Seniority", value: 0, color: "bg-chart-2" },
    { label: "Location", value: 0, color: "bg-chart-3" },
    { label: "Semantic", value: 0, color: "bg-chart-4" },
  ];

  // Try to parse percentages from reasons
  reasons.forEach((reason) => {
    const match = reason.match(/(\d+)%/);
    if (match) {
      const value = parseInt(match[1], 10);
      const lowerReason = reason.toLowerCase();

      if (lowerReason.includes("skill")) {
        defaultBreakdown[0].value = value;
      } else if (
        lowerReason.includes("seniority") ||
        lowerReason.includes("level")
      ) {
        defaultBreakdown[1].value = value;
      } else if (lowerReason.includes("location")) {
        defaultBreakdown[2].value = value;
      } else if (
        lowerReason.includes("semantic") ||
        lowerReason.includes("cv")
      ) {
        defaultBreakdown[3].value = value;
      }
    }
  });

  // Only include items with values
  return defaultBreakdown.filter((item) => item.value > 0);
}

export function MatchScoreCard({ score, reasons }: MatchScoreCardProps) {
  const getScoreColor = (s: number) => {
    if (s >= 80) return "text-match-high";
    if (s >= 60) return "text-match-mid";
    return "text-match-low";
  };

  const getScoreBg = (s: number) => {
    if (s >= 80) return "bg-match-high/10";
    if (s >= 60) return "bg-match-mid/10";
    return "bg-match-low/10";
  };

  const breakdown = parseReasons(reasons);

  return (
    <div className="space-y-6">
      {/* Overall Score */}
      <div
        className={cn(
          "flex flex-col items-center justify-center py-6 rounded-xl",
          getScoreBg(score)
        )}
      >
        <span
          className={cn("text-5xl font-black tracking-tight", getScoreColor(score))}
        >
          {score}
        </span>
        <span className="text-sm font-medium text-muted-foreground mt-1">
          Overall Match
        </span>
      </div>

      {/* Breakdown Bars */}
      {breakdown.length > 0 && (
        <div className="space-y-4">
          {breakdown.map((item) => (
            <div key={item.label} className="space-y-1.5">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-foreground">{item.label}</span>
                <span className="font-semibold text-muted-foreground">
                  {item.value}%
                </span>
              </div>
              <Progress
                value={item.value}
                className={cn("h-2", item.color)}
              />
            </div>
          ))}
        </div>
      )}

      {/* Match Reasons */}
      {reasons.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-foreground">Match Factors</h4>
          <ul className="space-y-1.5">
            {reasons.slice(0, 5).map((reason, idx) => (
              <li
                key={idx}
                className="text-sm text-muted-foreground flex items-start gap-2"
              >
                <span className="text-primary mt-0.5">•</span>
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
