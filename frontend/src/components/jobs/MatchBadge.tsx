import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface MatchBadgeProps {
  score: number;
  size?: "sm" | "md" | "lg";
}

export function MatchBadge({ score, size = "md" }: MatchBadgeProps) {
  const getColor = () => {
    if (score >= 80) return "bg-green-500 hover:bg-green-600";
    if (score >= 60) return "bg-amber-500 hover:bg-amber-600";
    return "bg-red-500 hover:bg-red-600";
  };

  const getSize = () => {
    switch (size) {
      case "sm":
        return "text-xs px-2 py-0.5";
      case "lg":
        return "text-lg px-4 py-1";
      default:
        return "text-sm px-3 py-1";
    }
  };

  return (
    <Badge className={cn(getColor(), getSize(), "text-white font-semibold")}>
      {score}%
    </Badge>
  );
}
