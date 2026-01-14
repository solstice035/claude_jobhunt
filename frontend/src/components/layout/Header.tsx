"use client";

import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  title: string;
  showRefresh?: boolean;
  subtitle?: string;
}

export function Header({ title, showRefresh = false, subtitle }: HeaderProps) {
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.post("/jobs/refresh", {});
      setLastRefresh(new Date());
    } catch {
      // Ignore errors
    } finally {
      setTimeout(() => setRefreshing(false), 2000);
    }
  };

  const formatLastRefresh = () => {
    if (!lastRefresh) return null;
    return `Updated ${lastRefresh.toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
    })}`;
  };

  return (
    <header className="h-16 border-b border-border bg-background flex items-center justify-between px-6">
      <div>
        <h2 className="text-2xl font-black tracking-tight text-foreground">
          {title}
        </h2>
        {subtitle && (
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        )}
      </div>

      {showRefresh && (
        <div className="flex items-center gap-3">
          {lastRefresh && (
            <span className="text-xs text-text-tertiary">
              {formatLastRefresh()}
            </span>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
            className="font-semibold"
          >
            <RefreshCw
              className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")}
            />
            {refreshing ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      )}
    </header>
  );
}
