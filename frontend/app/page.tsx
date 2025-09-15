"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, Activity, Database, Clock } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { FileUpload } from "@/components/file-upload";
import { AnomalyTable } from "@/components/anomaly-table";
import { FilterSidebar } from "@/components/filter-sidebar";
import { ReportManagement } from "@/components/report-management";
import { ChartData, ReportSummary, Stats } from "@/types";
import DashboardAnomaliesChart from "@/components/overall-chart";
import { Header } from "@/components/header";
import FullScreenLoader from "@/components/loading";

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [stats, setStats] = useState<Stats | null>(null);
  const [data, setData] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);
  const [firstLoad, setFirstLoad] = useState(true);
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);

      try {
        const [statsResponse, reportsResponse] = await Promise.all([
          fetch(`${process.env.NEXT_PUBLIC_API_URL}/stats`),
          fetch(`${process.env.NEXT_PUBLIC_API_URL}/reports`),
        ]);

        const statsData: Stats = await statsResponse.json();
        setStats(statsData);

        const reports: ReportSummary[] = await reportsResponse.json();

        const formatted = reports
          .filter((r) => r.status === "completed")
          .sort(
            (a, b) =>
              new Date(a.created_at).getTime() -
              new Date(b.created_at).getTime()
          )
          .map((r, index) => {
            const date = new Date(r.created_at);
            const timeStr = date.toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit",
              hour12: true,
            });
            const dateStr = date.toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
            });

            return {
              sequence: index + 1,
              label: `Report #${index + 1}`,
              anomalyCount: r.anomaly_count,
              anomalyPercentage: r.anomaly_percentage,
              totalRecords: r.total_records,
              filename: r.filename,
              time: timeStr,
              date: dateStr,
            };
          });

        setData(formatted);
      } catch (err) {
        console.error("Error fetching data:", err);
      } finally {
        setLoading(false);
        setFirstLoad(false);
      }
    };

    fetchData();
  }, []);
  if (firstLoad) {
    return <FullScreenLoader />;
  } else
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
        <Header
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          stats={stats}
        />
        <div className="container mx-auto px-6 py-6">
          {activeTab === "upload" ? (
            <FileUpload />
          ) : activeTab === "reports" ? (
            <ReportManagement />
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="lg:col-span-1">
                <FilterSidebar />
              </div>

              <div className="lg:col-span-3 space-y-6">
                <DashboardAnomaliesChart data={data} loading={loading} />
                <AnomalyTable />
              </div>
            </div>
          )}
        </div>
      </div>
    );
}
