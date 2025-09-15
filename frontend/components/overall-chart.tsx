"use client";
import { ChartData, ReportSummary } from "@/types";
import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";

export default function DashboardAnomaliesChart({
  loading,
  data,
}: {
  loading: boolean;
  data: ChartData[];
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px] bg-card rounded-2xl shadow">
        <div className="text-gray-600">Loading anomaly overview...</div>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg max-w-xs">
          <p className="font-semibold text-gray-800 mb-2">{dataPoint.label}</p>
          <div className="space-y-1 text-sm">
            <p className="text-gray-600">{`${dataPoint.date} at ${dataPoint.time}`}</p>
            <p className="text-gray-600">{`File: ${dataPoint.filename.substring(
              0,
              20
            )}...`}</p>
            <div className="border-t pt-2 mt-2">
              <p className="font-medium" style={{ color: "#06b6d4" }}>
                {`Anomalies: ${dataPoint.anomalyCount} / ${dataPoint.totalRecords}`}
              </p>
              <p className="font-medium" style={{ color: "#ef4444" }}>
                {`Percentage: ${dataPoint.anomalyPercentage.toFixed(2)}%`}
              </p>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };
  return (
    <div className="w-full bg-card rounded-2xl shadow p-6">
      <div className="">
        <h2 className="text-xl font-semibold mb-3">
          Anomaly Detection Reports
        </h2>
      </div>

      <div className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="sequence"
              tick={{ fontSize: 12, fill: "#64748b" }}
              tickLine={{ stroke: "#cbd5e1" }}
              axisLine={{ stroke: "#cbd5e1" }}
              label={{
                value: "Report Sequence",
                position: "insideBottom",
                offset: -10,
                style: {
                  textAnchor: "middle",
                  fontSize: "12px",
                  fill: "#64748b",
                },
              }}
            />
            <YAxis
              yAxisId="left"
              tick={{ fontSize: 12, fill: "#64748b" }}
              tickLine={{ stroke: "#cbd5e1" }}
              axisLine={{ stroke: "#cbd5e1" }}
              label={{
                value: "Anomaly Count",
                angle: -90,
                position: "insideLeft",
                style: {
                  textAnchor: "middle",
                  fontSize: "12px",
                  fill: "#64748b",
                },
              }}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{
                stroke: "#06b6d4",
                strokeWidth: 2,
                strokeDasharray: "5 5",
                opacity: 0.7,
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "20px" }} />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="anomalyCount"
              stroke="#06b6d4"
              name="Anomaly Count"
              strokeWidth={3}
              dot={{
                r: 5,
                fill: "#06b6d4",
                stroke: "#ffffff",
                strokeWidth: 2,
              }}
              activeDot={{
                r: 7,
                fill: "#0891b2",
                stroke: "#ffffff",
                strokeWidth: 2,
              }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
