export type FilterState = {
  search: string;
  severity: string;
  type: string;
  dateRange: { start: string; end: string };
  anomalyScore: { min: number; max: number };
  priceRange: { min: number; max: number };
  volumeRange: { min: number; max: number };
  anomalyTypes: string[];
  severityLevels: string[];
  anomaliesOnly: boolean;
};

export type Stats = {
  total_records: number;
  total_tickers: number;
  last_update: string | null;
};

export type ReportSummary = {
  id: string;
  filename: string;
  created_at: string;
  status: string;
  total_records: number;
  anomaly_count: number;
  anomaly_percentage: number;
};
export type ChartDataPoint = {
  date: string;
  close: number;
  volume: number;
  is_anomaly: boolean;
  anomaly_score: number;
  open?: number;
  high?: number;
  low?: number;
};

export type ChartData = {
  sequence: number;
  label: string;
  anomalyCount: number;
  anomalyPercentage: number;
  totalRecords: number;
  filename: string;
  time: string;
  date: string;
};
export type MonthlyDistributionProps = {
  stats: {
    monthly_distribution: {
      month: string;
      total: number;
      anomalies: number;
      anomaly_rate: number;
    }[];
  };
};
