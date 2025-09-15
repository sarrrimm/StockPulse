"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import { ReportSummary } from "@/types";

export function FilterSidebar() {
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchReports() {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/reports`);
        const reportsData: ReportSummary[] = await res.json();

        const completed = reportsData.filter((r) => r.status === "completed");
        setReports(completed);
      } catch (err) {
        console.error("Error fetching reports:", err);
      } finally {
        setLoading(false);
      }
    }

    fetchReports();
  }, []);

  const totalReports = reports.length;
  const totalAnomalies = reports.reduce((sum, d) => sum + d.anomaly_count, 0);
  const totalRecords = reports.reduce((sum, d) => sum + d.total_records, 0);
  const avgPercentage =
    reports.length > 0
      ? (
          reports.reduce((sum, d) => sum + d.anomaly_percentage, 0) /
          reports.length
        ).toFixed(2)
      : 0;

  return (
    <div className="space-y-4 sticky top-64 h-[50vh]">
      <Card className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-cyan-200/20 dark:border-cyan-800/20 h-full">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Quick Stats
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-3">
          {loading ? (
            <div className="text-sm text-muted-foreground">Loading...</div>
          ) : (
            <>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Total Reports
                </span>
                <span className="font-medium text-blue-600">
                  {totalReports}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Total Records
                </span>
                <span className="font-medium text-cyan-600">
                  {totalRecords.toLocaleString()}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Total Anomalies
                </span>
                <span className="font-medium text-red-600">
                  {totalAnomalies}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">
                  Detection Rate
                </span>
                <span className="font-medium text-sky-600">
                  {avgPercentage}%
                </span>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
