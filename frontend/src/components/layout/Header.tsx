"use client";

import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  title: string;
  showRefresh?: boolean;
}

export function Header({ title, showRefresh = false }: HeaderProps) {
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.post("/jobs/refresh", {});
    } catch {
      // Ignore errors
    } finally {
      setTimeout(() => setRefreshing(false), 2000);
    }
  };

  return (
    <header className="h-16 border-b flex items-center justify-between px-6">
      <h2 className="text-lg font-semibold">{title}</h2>
      {showRefresh && (
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <RefreshCw
            className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")}
          />
          {refreshing ? "Refreshing..." : "Refresh Jobs"}
        </Button>
      )}
    </header>
  );
}
