import { ArrowDown, ArrowUp } from "lucide-react"
import { formatNumber, formatPercentage } from "@/lib/utils"

interface DayDetailsProps {
  data: any | null
  ticker: string
}

export function DayDetails({ data, ticker }: DayDetailsProps) {
  if (!data) {
    return (
      <div className="h-24 border-t flex items-center justify-center text-muted-foreground text-sm">
        Hover over the chart to see daily details
      </div>
    )
  }

  // Calculate price change and percentage
  const priceChange = data.close && data.open ? data.close - data.open : null
  const priceChangePercent = data.close && data.open ? (priceChange! / data.open) * 100 : null
  const isPredicted = data.predicted !== undefined

  // Format date
  const date = new Date(data.date)
  const formattedDate = date.toLocaleDateString("en-US", {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  })

  return (
    <div className="h-auto py-3 border-t">
      <div className="flex flex-col space-y-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <span className="font-medium text-sm">{ticker}</span>
            <span className="mx-2 text-muted-foreground">•</span>
            <span className="text-sm text-muted-foreground">{formattedDate}</span>
            {data.isEarningsDate && (
              <span className="ml-2 bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-500 text-xs px-2 py-0.5 rounded-full">
                Earnings
              </span>
            )}
          </div>
          {isPredicted && (
            <span className="text-xs bg-rose-100 dark:bg-rose-900/30 text-rose-800 dark:text-rose-500 px-2 py-0.5 rounded-full">
              AI Prediction
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {!isPredicted ? (
            <>
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">Open</span>
                <span className="font-medium">${formatNumber(data.open)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">Close</span>
                <span className="font-medium">${formatNumber(data.close)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">High</span>
                <span className="font-medium">${formatNumber(data.high)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">Low</span>
                <span className="font-medium">${formatNumber(data.low)}</span>
              </div>
              <div className="flex flex-col col-span-2">
                <span className="text-xs text-muted-foreground">Volume</span>
                <span className="font-medium">{formatNumber(data.volume, 0)}</span>
              </div>
              <div className="flex flex-col col-span-2">
                <span className="text-xs text-muted-foreground">Change</span>
                <div className="flex items-center">
                  {priceChange !== null && priceChange > 0 ? (
                    <ArrowUp className="h-4 w-4 text-green-500 mr-1" />
                  ) : priceChange !== null && priceChange < 0 ? (
                    <ArrowDown className="h-4 w-4 text-red-500 mr-1" />
                  ) : null}
                  <span
                    className={`font-medium ${
                      priceChange !== null
                        ? priceChange > 0
                          ? "text-green-600 dark:text-green-500"
                          : priceChange < 0
                            ? "text-red-600 dark:text-red-500"
                            : ""
                        : ""
                    }`}
                  >
                    {priceChange !== null
                      ? `$${formatNumber(Math.abs(priceChange))} (${formatPercentage(Math.abs(priceChangePercent!))}%)`
                      : "-"}
                  </span>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">Predicted Price</span>
                <span className="font-medium">${formatNumber(data.predicted)}</span>
              </div>
              {data.isEarningsDate && (
                <div className="flex flex-col">
                  <span className="text-xs text-muted-foreground">Expected Move</span>
                  <span className="font-medium">±{data.expectedMove}%</span>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
