import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface MatchBreakdownProps {
  score: number;
  reasons: string[];
}

export function MatchBreakdown({ score, reasons }: MatchBreakdownProps) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">Match Score</span>
          <span className="text-2xl font-bold">{score}%</span>
        </div>
        <Progress value={score} className="h-3" />
      </div>

      {reasons.length > 0 && (
        <div>
          <span className="text-sm font-medium">Match Reasons</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {reasons.map((reason, idx) => (
              <Badge key={idx} variant="secondary">
                {reason}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
