"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Briefcase, User, BarChart3, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const navItems = [
  { href: "/jobs", label: "Jobs", icon: Briefcase },
  { href: "/applications", label: "Applications", icon: BarChart3 },
  { href: "/profile", label: "Profile", icon: User },
];

export function Sidebar() {
  const pathname = usePathname();

  const handleLogout = async () => {
    try {
      await api.post("/auth/logout", {});
    } catch {
      // Ignore errors on logout
    }
    window.location.href = "/login";
  };

  return (
    <aside className="w-64 border-r bg-card h-screen flex flex-col">
      <div className="p-6">
        <h1 className="text-xl font-bold">Job Search</h1>
        <p className="text-sm text-muted-foreground">AI-Powered Agent</p>
      </div>

      <nav className="flex-1 px-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname.startsWith(item.href);

          return (
            <Link key={item.href} href={item.href}>
              <div
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-accent"
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t">
        <Button
          variant="ghost"
          className="w-full justify-start gap-3"
          onClick={handleLogout}
        >
          <LogOut className="h-4 w-4" />
          Logout
        </Button>
      </div>
    </aside>
  );
}
