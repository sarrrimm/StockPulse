"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, Upload, Database, Clock } from "lucide-react";
import { ThemeToggle } from "../components/theme-toggle";
import { Stats } from "@/types";

export function Header({
  activeTab,
  setActiveTab,
  stats,
}: {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  stats: Stats | null;
}) {
  const lastUpdateLabel = stats?.last_update
    ? new Date(stats.last_update).toLocaleString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        day: "numeric",
        month: "short",
      })
    : "N/A";

  return (
    <div className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-sm">
      <div className="container mx-auto px-6 py-4">
        <Card className="bg-gradient-to-r from-cyan-500/10 via-sky-500/10 to-blue-500/10 border-cyan-200/20 dark:border-cyan-800/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
                  StockPulse
                </h1>
                <p className="text-muted-foreground mt-1">
                  Advanced Stock Anomaly Detection Platform
                </p>
              </div>
              <ThemeToggle />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex gap-2">
                <Button
                  variant={activeTab === "dashboard" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveTab("dashboard")}
                  className="rounded-full"
                >
                  <Activity className="w-4 h-4 mr-2" />
                  Dashboard
                </Button>
                <Button
                  variant={activeTab === "upload" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveTab("upload")}
                  className="rounded-full"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload
                </Button>
                <Button
                  variant={activeTab === "reports" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveTab("reports")}
                  className="rounded-full"
                >
                  <Database className="w-4 h-4 mr-2" />
                  Reports
                </Button>
              </div>

              <div className="flex gap-4">
                <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg">
                  <Database className="w-4 h-4 text-cyan-600" />
                  <div className="text-sm">
                    <div className="font-medium">
                      {stats ? stats.total_records.toLocaleString() : "--"}
                    </div>
                    <div className="text-xs text-muted-foreground">Records</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg">
                  <Activity className="w-4 h-4 text-sky-600" />
                  <div className="text-sm">
                    <div className="font-medium">
                      {stats ? stats.total_tickers : "--"}
                    </div>
                    <div className="text-xs text-muted-foreground">Tickers</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg">
                  <Clock className="w-4 h-4 text-blue-600" />
                  <div className="text-sm">
                    <div className="font-medium">{lastUpdateLabel}</div>
                    <div className="text-xs text-muted-foreground">
                      Last Update
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
