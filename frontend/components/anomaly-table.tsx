"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertTriangle,
  Search,
  MoreHorizontal,
  TrendingUp,
  TrendingDown,
  X,
  Filter,
} from "lucide-react";
import { Pagination } from "@/components/pagination";
import { debounce } from "lodash";
import ExportButton from "./export-button";
import { FilterState } from "@/types";

interface AnomalyData {
  date: string;
  close: number;
  volume: number;
  anomaly_score: number;
  is_anomaly: boolean;
  severity: "low" | "medium" | "high";
  type: "price" | "volume" | "combined";
  change_percent: number;
}

interface ApiResponse {
  anomalies: AnomalyData[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export function AnomalyTable() {
  const [data, setData] = useState<AnomalyData[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [sortBy, setSortBy] = useState<keyof AnomalyData>("date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [totalItems, setTotalItems] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  const buildQueryParams = useCallback(() => {
    const params = new URLSearchParams();

    params.append("page", currentPage.toString());
    params.append("page_size", pageSize.toString());
    params.append("sort_by", sortBy);
    params.append("sort_order", sortOrder);

    if (searchTerm.trim()) params.append("search", searchTerm.trim());
    if (severityFilter !== "all") params.append("severity", severityFilter);
    if (typeFilter !== "all") params.append("type", typeFilter);

    return params.toString();
  }, [
    currentPage,
    pageSize,
    sortBy,
    sortOrder,
    searchTerm,
    severityFilter,
    typeFilter,
  ]);

  const debouncedSearch = useMemo(
    () =>
      debounce((value: string) => {
        setSearchTerm(value);
        setCurrentPage(1);
      }, 400),
    []
  );

  useEffect(() => {
    return () => debouncedSearch.cancel();
  }, [debouncedSearch]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    debouncedSearch(e.target.value);
  };

  useEffect(() => {
    const fetchAnomalies = async () => {
      try {
        setLoading(true);
        const queryParams = buildQueryParams();
        const url = `${process.env.NEXT_PUBLIC_API_URL}/anomalies?${queryParams}`;

        const res = await fetch(url, {
          headers: { "Content-Type": "application/json" },
        });
        if (!res.ok)
          throw new Error(`Failed to fetch anomalies: ${res.status}`);

        const json: ApiResponse = await res.json();
        setData(json.anomalies || []);
        setTotalItems(json.total || 0);
        setTotalPages(json.total_pages || 0);
      } catch (err) {
        console.error("Error fetching anomalies:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchAnomalies();
  }, [buildQueryParams]);

  const handleSort = (column: keyof AnomalyData) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(column);
      setSortOrder("desc");
    }
    setCurrentPage(1);
  };

  const resetAllFilters = () => {
    setSeverityFilter("all");
    setTypeFilter("all");
    setSearchTerm("");
    setCurrentPage(1);
  };

  const hasActiveFilters = useMemo(
    () =>
      searchTerm.trim() !== "" ||
      severityFilter !== "all" ||
      typeFilter !== "all",
    [searchTerm, severityFilter, typeFilter]
  );

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" /> Anomaly Detection Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center">
            <div className="text-center space-y-2">
              <div className="w-8 h-8 border-2 border-cyan-600 border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm text-muted-foreground">
                Loading anomaly data...
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" /> Anomaly Detection Results
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {totalItems} total anomalies
              </Badge>
              {hasActiveFilters && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetAllFilters}
                  className="text-xs"
                >
                  <X className="w-3 h-3 mr-1" /> Clear Filters
                </Button>
              )}
              <ExportButton />
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search by date or type..."
                onChange={handleSearchChange}
                className="pl-10"
                value={searchTerm}
              />
            </div>
            <Select
              value={severityFilter}
              onValueChange={(value) => {
                setSeverityFilter(value);
                setCurrentPage(1);
              }}
            >
              <SelectTrigger className="w-full sm:w-40">
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={typeFilter}
              onValueChange={(value) => {
                setTypeFilter(value);
                setCurrentPage(1);
              }}
            >
              <SelectTrigger className="w-full sm:w-40">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="price">Price</SelectItem>
                <SelectItem value="volume">Volume</SelectItem>
                <SelectItem value="combined">Combined</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {hasActiveFilters && (
            <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-blue-600" />
                <span className="text-sm text-blue-700 dark:text-blue-300">
                  Filters active - showing {totalItems} results
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={resetAllFilters}
                className="h-6 px-2 text-blue-600 hover:text-blue-700"
              >
                Clear all
              </Button>
            </div>
          )}

          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => handleSort("date")}
                  >
                    Date{" "}
                    {sortBy === "date" && (sortOrder === "asc" ? "↑" : "↓")}
                  </TableHead>
                  <TableHead
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => handleSort("close")}
                  >
                    Price{" "}
                    {sortBy === "close" && (sortOrder === "asc" ? "↑" : "↓")}
                  </TableHead>
                  <TableHead
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => handleSort("anomaly_score")}
                  >
                    Score{" "}
                    {sortBy === "anomaly_score" &&
                      (sortOrder === "asc" ? "↑" : "↓")}
                  </TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead>Change</TableHead>
                  <TableHead className="w-12"></TableHead>
                </TableRow>
              </TableHeader>

              <TableBody>
                {data.length > 0 ? (
                  data.map((anomaly, index) => (
                    <TableRow
                      key={`${anomaly.date}-${index}`}
                      className="hover:bg-muted/50"
                    >
                      <TableCell className="font-medium">
                        {new Date(anomaly.date).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })}
                      </TableCell>
                      <TableCell>${anomaly.close.toFixed(2)}</TableCell>
                      <TableCell>
                        <span
                          className={`font-mono text-sm ${
                            anomaly.anomaly_score < -3
                              ? "text-red-600 dark:text-red-400"
                              : anomaly.anomaly_score < -2
                              ? "text-yellow-600 dark:text-yellow-400"
                              : "text-blue-600 dark:text-blue-400"
                          }`}
                        >
                          {anomaly.anomaly_score.toFixed(3)}
                        </span>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={`text-xs ${
                            anomaly.severity === "high"
                              ? "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400"
                              : anomaly.severity === "medium"
                              ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400"
                              : "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
                          }`}
                        >
                          {anomaly.severity}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          {anomaly.change_percent > 0 ? (
                            <TrendingUp className="w-3 h-3 text-green-600" />
                          ) : (
                            <TrendingDown className="w-3 h-3 text-red-600" />
                          )}
                          <span
                            className={`text-sm font-medium ${
                              anomaly.change_percent > 0
                                ? "text-green-600"
                                : "text-red-600"
                            }`}
                          >
                            {anomaly.change_percent > 0 ? "+" : ""}
                            {anomaly.change_percent.toFixed(2)}%
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="w-8 h-8 p-0"
                            >
                              <MoreHorizontal className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem>View Details</DropdownMenuItem>
                            <DropdownMenuItem>
                              Add to Watchlist
                            </DropdownMenuItem>
                            <DropdownMenuItem>Export Data</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center py-8">
                      <AlertTriangle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="text-lg font-medium mb-2">
                        No anomalies found
                      </h3>
                      <p className="text-muted-foreground mb-4">
                        {hasActiveFilters
                          ? "Try adjusting your filters or search criteria."
                          : "No data available at the moment."}
                      </p>
                      {hasActiveFilters && (
                        <Button variant="outline" onClick={resetAllFilters}>
                          Clear all filters
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>

          {totalItems > 0 && (
            <div className="mt-6">
              <Pagination
                currentPage={currentPage}
                totalPages={totalPages}
                pageSize={pageSize}
                totalItems={totalItems}
                onPageChange={setCurrentPage}
                onPageSizeChange={setPageSize}
                loading={loading}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
