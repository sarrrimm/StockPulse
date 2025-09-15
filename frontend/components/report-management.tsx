"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  FileText,
  Search,
  Download,
  Trash2,
  Eye,
  Calendar,
  AlertCircle,
  CheckCircle,
  Clock,
  Loader2,
} from "lucide-react";
import ExportButton from "./export-button";
import { MonthlyDistributionChart } from "./monthly-chart";

interface Report {
  id: string;
  filename: string;
  created_at: string;
  status: "processing" | "completed" | "failed";
  total_records: number;
  anomaly_count: number;
  anomaly_percentage: number;
  threshold_percentile?: number;
  error_message?: string;
}

interface ReportStatistics {
  report_id: string;
  total_records: number;
  anomaly_count: number;
  anomaly_percentage: number;
  threshold_percentile: number;
  date_range: {
    start: string;
    end: string;
  };
  score_statistics: {
    min_score: number;
    max_score: number;
    avg_score: number;
    avg_anomaly_score: number;
    avg_normal_score: number;
  };
  monthly_distribution: Array<{
    month: string;
    total: number;
    anomalies: number;
    anomaly_rate: number;
  }>;
}

export function ReportManagement() {
  const [reports, setReports] = useState<Report[]>([]);
  const [filteredReports, setFilteredReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [reportStats, setReportStats] = useState<ReportStatistics | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);

  const fetchReports = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/reports`
      );
      const data = await response.json();

      setReports(data);
      setFilteredReports(data);
    } catch (error) {
      console.error("Failed to fetch reports:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchReportStatistics = async (reportId: string) => {
    try {
      setStatsLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/reports/${reportId}/statistics`
      );
      const data = await response.json();

      setReportStats(data);
    } catch (error) {
      console.error("Failed to fetch report statistics:", error);
    } finally {
      setStatsLoading(false);
    }
  };

  const deleteReport = async (reportId: string) => {
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/reports/${reportId}`, {
        method: "DELETE",
      });

      setReports((prev) => prev.filter((report) => report.id !== reportId));
      setFilteredReports((prev) =>
        prev.filter((report) => report.id !== reportId)
      );
    } catch (error) {
      console.error("Failed to delete report:", error);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  useEffect(() => {
    let filtered = reports;

    if (searchTerm) {
      filtered = filtered.filter(
        (report) =>
          report.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
          report.id.includes(searchTerm)
      );
    }

    if (statusFilter !== "all") {
      filtered = filtered.filter((report) => report.status === statusFilter);
    }

    setFilteredReports(filtered);
  }, [reports, searchTerm, statusFilter]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case "processing":
        return <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />;
      case "failed":
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      default:
        return <Clock className="w-4 h-4 text-gray-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "processing":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
      case "failed":
        return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400";
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Report Management
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center">
            <div className="text-center space-y-2">
              <div className="w-8 h-8 border-2 border-cyan-600 border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm text-muted-foreground">
                Loading reports...
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Report Management
            </CardTitle>
            <Badge variant="outline" className="text-xs">
              {filteredReports.length} of {reports.length} reports
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search by filename or ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-full sm:w-40">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="processing">Processing</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            {filteredReports.map((report) => (
              <div
                key={report.id}
                className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-4 flex-1">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(report.status)}
                    <FileText className="w-4 h-4 text-muted-foreground" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-medium truncate">
                        {report.filename}
                      </h3>
                      <Badge
                        variant="outline"
                        className={`text-xs ${getStatusColor(report.status)}`}
                      >
                        {report.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {new Date(report.created_at).toLocaleDateString()}
                      </span>
                      {report.status === "completed" && (
                        <>
                          <span>{report.total_records} records</span>
                          <span className="text-red-600">
                            {report.anomaly_count} anomalies
                          </span>
                          <span>
                            {report.anomaly_percentage.toFixed(2)}% detection
                            rate
                          </span>
                        </>
                      )}
                      {report.threshold_percentile && (
                        <span>Threshold: {report.threshold_percentile}%</span>
                      )}
                    </div>
                    {report.error_message && (
                      <Alert variant="destructive" className="mt-2">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="text-xs">
                          {report.error_message}
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {report.status === "completed" && (
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setSelectedReport(report);
                            fetchReportStatistics(report.id);
                          }}
                        >
                          <Eye className="w-4 h-4 mr-1" />
                          View
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                        <DialogHeader>
                          <DialogTitle>
                            Report Details: {selectedReport?.filename}
                          </DialogTitle>
                        </DialogHeader>
                        {statsLoading ? (
                          <div className="flex items-center justify-center py-8">
                            <Loader2 className="w-6 h-6 animate-spin" />
                          </div>
                        ) : (
                          reportStats && (
                            <ReportStatisticsView stats={reportStats} />
                          )
                        )}
                      </DialogContent>
                    </Dialog>
                  )}
                  <ExportButton
                    reportId={report?.id}
                    disabled={report.status !== "completed"}
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => deleteReport(report.id)}
                    className="text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>

          {filteredReports.length === 0 && (
            <div className="text-center py-8">
              <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No reports found</h3>
              <p className="text-muted-foreground">
                Try adjusting your search criteria or upload a new file.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-cyan-600" />
              <div>
                <div className="text-2xl font-bold">{reports.length}</div>
                <div className="text-xs text-muted-foreground">
                  Total Reports
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <div>
                <div className="text-2xl font-bold">
                  {reports.filter((r) => r.status === "completed").length}
                </div>
                <div className="text-xs text-muted-foreground">Completed</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 text-blue-600" />
              <div>
                <div className="text-2xl font-bold">
                  {reports.filter((r) => r.status === "processing").length}
                </div>
                <div className="text-xs text-muted-foreground">Processing</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-600" />
              <div>
                <div className="text-2xl font-bold">
                  {reports.filter((r) => r.status === "failed").length}
                </div>
                <div className="text-xs text-muted-foreground">Failed</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function ReportStatisticsView({ stats }: { stats: ReportStatistics }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-4 bg-muted/50 rounded-lg">
          <div className="text-2xl font-bold text-cyan-600">
            {stats.total_records}
          </div>
          <div className="text-xs text-muted-foreground">Total Records</div>
        </div>
        <div className="text-center p-4 bg-muted/50 rounded-lg">
          <div className="text-2xl font-bold text-red-600">
            {stats.anomaly_count}
          </div>
          <div className="text-xs text-muted-foreground">Anomalies</div>
        </div>
        <div className="text-center p-4 bg-muted/50 rounded-lg">
          <div className="text-2xl font-bold text-green-600">
            {stats.anomaly_percentage.toFixed(2)}%
          </div>
          <div className="text-xs text-muted-foreground">Detection Rate</div>
        </div>
        <div className="text-center p-4 bg-muted/50 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">
            {stats.threshold_percentile}%
          </div>
          <div className="text-xs text-muted-foreground">Threshold</div>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Date Range</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">From:</span>
            <span className="font-medium">{stats.date_range.start}</span>
          </div>
          <div className="flex items-center justify-between mt-2">
            <span className="text-sm text-muted-foreground">To:</span>
            <span className="font-medium">{stats.date_range.end}</span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Score Statistics</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">Min Score:</span>
            <span className="font-mono text-sm">
              {stats.score_statistics.min_score.toFixed(4)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">Max Score:</span>
            <span className="font-mono text-sm">
              {stats.score_statistics.max_score.toFixed(4)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">Avg Score:</span>
            <span className="font-mono text-sm">
              {stats.score_statistics.avg_score.toFixed(4)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">
              Avg Anomaly Score:
            </span>
            <span className="font-mono text-sm text-red-600">
              {stats.score_statistics.avg_anomaly_score.toFixed(4)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">
              Avg Normal Score:
            </span>
            <span className="font-mono text-sm text-green-600">
              {stats.score_statistics.avg_normal_score.toFixed(4)}
            </span>
          </div>
        </CardContent>
      </Card>
      <MonthlyDistributionChart stats={stats} />
    </div>
  );
}
