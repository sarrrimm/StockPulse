"use client";

import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Upload,
  FileText,
  AlertCircle,
  CheckCircle,
  Loader2,
  Settings,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

interface UploadResponse {
  report_id: string;
  status: string;
  message: string;
  threshold_percentile?: number;
}

export function FileUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useCustomThreshold, setUseCustomThreshold] = useState(false);
  const [threshold, setThreshold] = useState([5.0]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const csvFile = acceptedFiles.find(
      (file) => file.type === "text/csv" || file.name.endsWith(".csv")
    );
    if (csvFile) {
      setFile(csvFile);
      setError(null);
      setUploadResult(null);
    } else {
      setError("Please upload a CSV file");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.ms-excel": [".csv"],
    },
    multiple: false,
  });

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const endpoint = useCustomThreshold
        ? `/upload-and-analyze-custom?threshold_percentile=${threshold[0]}`
        : "/upload-and-analyze";

      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}${endpoint}`,
        {
          method: "POST",
          body: formData,
        }
      );

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result: UploadResponse = await response.json();
      setUploadResult(result);

      setTimeout(() => {
        setFile(null);
        setUploadProgress(0);
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (
      Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
    );
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Analysis Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm font-medium">Custom Threshold</Label>
              <p className="text-xs text-muted-foreground">
                Use custom sensitivity threshold instead of default 5%
              </p>
            </div>
            <Switch
              checked={useCustomThreshold}
              onCheckedChange={setUseCustomThreshold}
            />
          </div>

          {useCustomThreshold && (
            <div className="space-y-2">
              <Label className="text-sm">
                Threshold Percentile: {threshold[0]}%
              </Label>
              <Slider
                value={threshold}
                onValueChange={setThreshold}
                max={25}
                min={1}
                step={0.5}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Lower values = more sensitive detection (1.0% - 25.0%)
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Stock Data
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? "border-cyan-500 bg-cyan-50 dark:bg-cyan-950/20"
                : "border-muted-foreground/25 hover:border-cyan-400"
            }`}
          >
            <input {...getInputProps()} />
            <div className="space-y-4">
              <div className="mx-auto w-12 h-12 bg-cyan-100 dark:bg-cyan-900/20 rounded-full flex items-center justify-center">
                <Upload className="w-6 h-6 text-cyan-600" />
              </div>

              {file ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-center gap-2">
                    <FileText className="w-4 h-4 text-green-600" />
                    <span className="font-medium">{file.name}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(file.size)} • Ready to analyze
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-lg font-medium">
                    {isDragActive
                      ? "Drop your CSV file here"
                      : "Drag & drop your CSV file here"}
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    or click to browse files
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 p-4 bg-muted/50 rounded-lg">
            <h4 className="text-sm font-medium mb-2">
              CSV Format Requirements:
            </h4>
            <div className="text-xs text-muted-foreground space-y-1">
              <p>• Required columns: Date, Open, High, Low, Close, Volume</p>
              <p>• Date format: YYYY-MM-DD or similar standard formats</p>
              <p>• Numeric values for price and volume data</p>
              <p>• Maximum file size: 50MB</p>
            </div>
          </div>

          {uploading && (
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Uploading and analyzing...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="w-full" />
            </div>
          )}

          <div className="flex gap-3 mt-6">
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="flex-1"
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Start Analysis
                </>
              )}
            </Button>

            {file && !uploading && (
              <Button
                variant="outline"
                onClick={() => {
                  setFile(null);
                  setError(null);
                  setUploadResult(null);
                }}
              >
                Clear
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {uploadResult && (
        <Card className="border-green-200 dark:border-green-800">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center flex-shrink-0">
                <CheckCircle className="w-4 h-4 text-green-600" />
              </div>
              <div className="space-y-2 flex-1">
                <h3 className="font-medium text-green-900 dark:text-green-100">
                  Analysis Started Successfully
                </h3>
                <p className="text-sm text-green-700 dark:text-green-300">
                  {uploadResult.message}
                </p>
                <div className="flex items-center gap-4 mt-3">
                  <Badge variant="outline" className="text-xs">
                    Report ID: {uploadResult.report_id.slice(0, 8)}...
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    Status: {uploadResult.status}
                  </Badge>
                  {uploadResult.threshold_percentile && (
                    <Badge variant="outline" className="text-xs">
                      Threshold: {uploadResult.threshold_percentile}%
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
