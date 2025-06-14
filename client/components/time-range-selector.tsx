"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface TimeRangeSelectorProps {
  selectedRange: string
  onRangeChange: (range: string) => void
}

export function TimeRangeSelector({ selectedRange, onRangeChange }: TimeRangeSelectorProps) {
  const timeRanges = [
    { label: "7D", value: "7d" },
    { label: "1M", value: "1m" },
    { label: "3M", value: "3m" },
    { label: "6M", value: "6m" },
    { label: "1Y", value: "1y" },
    { label: "3Y", value: "3y" },
  ]

  return (
    <div className="flex flex-wrap gap-2">
      {timeRanges.map((range) => (
        <Button
          key={range.value}
          variant={selectedRange === range.value ? "default" : "outline"}
          size="sm"
          className={cn("px-3 py-1 h-8", selectedRange === range.value ? "bg-primary text-primary-foreground" : "")}
          onClick={() => onRangeChange(range.value)}
        >
          {range.label}
        </Button>
      ))}
    </div>
  )
}
