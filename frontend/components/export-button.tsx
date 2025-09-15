"use client";

import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";

export default function ExportButton({
  reportId,
  disabled,
}: {
  reportId?: string;
  disabled?: boolean;
}) {
  const handleExport = () => {
    console.log(reportId);
    const id = reportId || "0";
    window.open(
      `${process.env.NEXT_PUBLIC_API_URL}/anomalies/export?report_id=${id}`,
      "_blank"
    );
  };

  return (
    <Button
      onClick={handleExport}
      variant="outline"
      size="sm"
      className="cursor-pointer"
    >
      <Download className="w-4 h-4 mr-1" />
      Export
    </Button>
  );
}
