"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Filter, X, ChevronDown, ChevronUp } from "lucide-react";
import { FilterState } from "@/types";

interface AdvancedFiltersProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  onReset: () => void;
}

export function AdvancedFilters({
  filters,
  onFiltersChange,
  onReset,
}: AdvancedFiltersProps) {
  const [isOpen, setIsOpen] = useState(false);

  const updateFilter = (key: keyof FilterState, value: any) => {
    onFiltersChange({
      ...filters,
      [key]: value,
    });
  };

  const updateNestedFilter = (
    parentKey: keyof FilterState,
    childKey: string,
    value: any
  ) => {
    onFiltersChange({
      ...filters,
      [parentKey]: {
        ...(filters[parentKey] as any),
        [childKey]: value,
      },
    });
  };

  const toggleArrayFilter = (
    key: "anomalyTypes" | "severityLevels",
    value: string
  ) => {
    const currentArray = filters[key];
    const newArray = currentArray.includes(value)
      ? currentArray.filter((item) => item !== value)
      : [...currentArray, value];

    updateFilter(key, newArray);
  };

  const getActiveFiltersCount = () => {
    let count = 0;

    if (filters.dateRange.start || filters.dateRange.end) count++;
    if (filters.anomalyScore.min !== -5 || filters.anomalyScore.max !== 5)
      count++;
    if (filters.priceRange.min !== 0 || filters.priceRange.max !== 1000)
      count++;
    if (filters.volumeRange.min !== 0 || filters.volumeRange.max !== 10)
      count++;
    if (filters.anomalyTypes.length > 0) count++;
    if (filters.severityLevels.length > 0) count++;
    if (filters.anomaliesOnly) count++;

    return count;
  };

  const activeFiltersCount = getActiveFiltersCount();

  return (
    <Card>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base">
                <Filter className="w-4 h-4" />
                Advanced Filters
              </CardTitle>
              <div className="flex items-center gap-2">
                {activeFiltersCount > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onReset();
                    }}
                    className="text-xs"
                  >
                    <X className="w-3 h-3 mr-1" />
                    Clear
                  </Button>
                )}
                {isOpen ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <Label className="text-sm font-medium">Date Range</Label>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label
                    htmlFor="start-date"
                    className="text-xs text-muted-foreground"
                  >
                    From
                  </Label>
                  <Input
                    id="start-date"
                    type="date"
                    value={filters.dateRange.start}
                    onChange={(e) =>
                      updateNestedFilter("dateRange", "start", e.target.value)
                    }
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label
                    htmlFor="end-date"
                    className="text-xs text-muted-foreground"
                  >
                    To
                  </Label>
                  <Input
                    id="end-date"
                    type="date"
                    value={filters.dateRange.end}
                    onChange={(e) =>
                      updateNestedFilter("dateRange", "end", e.target.value)
                    }
                    className="mt-1"
                  />
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <Label className="text-sm font-medium">
                Anomaly Score Range: {filters.anomalyScore.min} to{" "}
                {filters.anomalyScore.max}
              </Label>
              <div className="px-2">
                <Slider
                  value={[filters.anomalyScore.min, filters.anomalyScore.max]}
                  onValueChange={([min, max]) =>
                    updateFilter("anomalyScore", { min, max })
                  }
                  min={-5}
                  max={5}
                  step={0.1}
                  className="w-full"
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>-5.0 (Strong Anomaly)</span>
                <span>5.0 (Normal)</span>
              </div>
            </div>

            <div className="space-y-3">
              <Label className="text-sm font-medium">
                Price Range: ${filters.priceRange.min} - $
                {filters.priceRange.max}
              </Label>
              <div className="px-2">
                <Slider
                  value={[filters.priceRange.min, filters.priceRange.max]}
                  onValueChange={([min, max]) =>
                    updateFilter("priceRange", { min, max })
                  }
                  min={0}
                  max={1000}
                  step={5}
                  className="w-full"
                />
              </div>
            </div>

            <div className="space-y-3">
              <Label className="text-sm font-medium">
                Volume Range: {filters.volumeRange.min}M -{" "}
                {filters.volumeRange.max}M
              </Label>
              <div className="px-2">
                <Slider
                  value={[filters.volumeRange.min, filters.volumeRange.max]}
                  onValueChange={([min, max]) =>
                    updateFilter("volumeRange", { min, max })
                  }
                  min={0}
                  max={10}
                  step={0.1}
                  className="w-full"
                />
              </div>
            </div>

            <div className="space-y-3">
              <Label className="text-sm font-medium">Anomaly Types</Label>
              <div className="flex flex-wrap gap-2">
                {["price", "volume", "combined"].map((type) => (
                  <Button
                    key={type}
                    variant={
                      filters.anomalyTypes.includes(type)
                        ? "default"
                        : "outline"
                    }
                    size="sm"
                    onClick={() => toggleArrayFilter("anomalyTypes", type)}
                    className="text-xs capitalize"
                  >
                    {type}
                  </Button>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <Label className="text-sm font-medium">Severity Levels</Label>
              <div className="flex flex-wrap gap-2">
                {["low", "medium", "high"].map((severity) => (
                  <Button
                    key={severity}
                    variant={
                      filters.severityLevels.includes(severity)
                        ? "default"
                        : "outline"
                    }
                    size="sm"
                    onClick={() =>
                      toggleArrayFilter("severityLevels", severity)
                    }
                    className="text-xs capitalize"
                  >
                    {severity}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-sm font-medium">
                  Show Anomalies Only
                </Label>
                <p className="text-xs text-muted-foreground">
                  Hide normal data points
                </p>
              </div>
              <Switch
                checked={filters.anomaliesOnly}
                onCheckedChange={(checked) =>
                  updateFilter("anomaliesOnly", checked)
                }
              />
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}
