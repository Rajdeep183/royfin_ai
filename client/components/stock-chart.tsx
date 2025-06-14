"use client"

import { useMemo, useState, useEffect, useCallback } from "react"
import { Area, AreaChart, Brush, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { DayDetails } from "./day-details"
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts/highstock";
import { useTheme } from "next-themes";
import { createRealTimeConnection } from "@/lib/stock-api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, Loader2 } from "lucide-react";
import { ErrorBoundary } from "./error-boundary";

// Update the StockChartProps interface
interface StockChartProps {
  stockData: {
    historical: any[]
    earnings: {
      date: string
      expectedMove: number
    }[]
    realTime?: boolean
  }
  predictedData: any[] | null
  ticker: string
  isLoading?: boolean
  error?: string | null
}

// Custom cursor component that shows price next to cursor
const CustomCursor = ({ points, width, height, stroke, payload, price }: any) => {
  try {
    if (!points || points.length === 0) return null

    const { x, y } = points[0]
    const value = payload && payload[0] ? payload[0].value : null

    return (
      <g>
        {/* Vertical line */}
        <line x1={x} y1={0} x2={x} y2={height} stroke="#9ca3af" strokeWidth={1} strokeDasharray="3 3" />

        {/* Price display */}
        {value !== null && (
          <g>
            <rect x={x + 8} y={y - 15} width={60} height={30} rx={4} fill="rgba(0, 0, 0, 0.7)" fillOpacity={0.8} />
            <text x={x + 38} y={y + 5} textAnchor="middle" fill="#ffffff" fontSize={12}>
              ${typeof value === "number" ? value.toFixed(2) : value}
            </text>
          </g>
        )}
      </g>
    )
  } catch (error) {
    console.error('CustomCursor error:', error)
    return null
  }
}

// Loading component
const ChartLoading = () => (
  <div className="flex items-center justify-center h-64">
    <div className="flex flex-col items-center gap-2">
      <Loader2 className="h-8 w-8 animate-spin" />
      <span className="text-sm text-muted-foreground">Loading chart data...</span>
    </div>
  </div>
)

// Error component
const ChartError = ({ error, onRetry }: { error: string; onRetry?: () => void }) => (
  <Alert variant="destructive" className="m-4">
    <AlertTriangle className="h-4 w-4" />
    <AlertDescription>
      {error}
      {onRetry && (
        <button 
          onClick={onRetry}
          className="ml-2 underline hover:no-underline"
        >
          Try again
        </button>
      )}
    </AlertDescription>
  </Alert>
)

// Update the StockChart component
export function StockChart({ stockData, predictedData, ticker, isLoading = false, error = null }: StockChartProps) {
  const [activePayload, setActivePayload] = useState<any>(null)
  const [hoveredData, setHoveredData] = useState<any>(null)
  const [isCandlestickView, setIsCandlestickView] = useState(false)
  const [realTimePrice, setRealTimePrice] = useState<number | null>(null)
  const [priceChange, setPriceChange] = useState<{ value: number; percentage: number } | null>(null)
  const [wsError, setWsError] = useState<string | null>(null)
  const [chartError, setChartError] = useState<string | null>(null)
  const { theme } = useTheme()

  // Real-time WebSocket connection with error handling
  useEffect(() => {
    if (!ticker || !stockData?.realTime) return

    let socket: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout | null = null

    const connectWebSocket = () => {
      try {
        setWsError(null)
        socket = createRealTimeConnection(ticker, (tradeData) => {
          try {
            if (tradeData && typeof tradeData.p === 'number') {
              setRealTimePrice(tradeData.p)
              
              // Calculate price change
              if (stockData?.historical?.length > 0) {
                const lastClose = stockData.historical[stockData.historical.length - 1].close
                const change = tradeData.p - lastClose
                const changePercent = (change / lastClose) * 100
                setPriceChange({ value: change, percentage: changePercent })
              }
            }
          } catch (error) {
            console.error('WebSocket data processing error:', error)
          }
        })

        if (socket) {
          socket.onerror = () => {
            setWsError('Real-time connection failed')
            // Attempt to reconnect after 5 seconds
            reconnectTimeout = setTimeout(connectWebSocket, 5000)
          }

          socket.onclose = () => {
            setRealTimePrice(null)
            setPriceChange(null)
          }
        }
      } catch (error) {
        console.error('WebSocket connection error:', error)
        setWsError('Failed to establish real-time connection')
      }
    }

    connectWebSocket()

    // Cleanup on unmount
    return () => {
      if (socket) {
        socket.close()
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
    }
  }, [ticker, stockData])

  // Define the new darker, more saturated green color
  const richGreenColor = "#5e6c3e"

  const chartData = useMemo(() => {
    try {
      if (!stockData || !stockData.historical || !Array.isArray(stockData.historical)) {
        return []
      }

      // Validate and clean historical data
      const validHistoricalData = stockData.historical
        .filter(item => item && item.date && typeof item.close === 'number')
        .map(item => ({
          ...item,
          close: Number(item.close) || 0,
          open: Number(item.open) || 0,
          high: Number(item.high) || 0,
          low: Number(item.low) || 0,
          volume: Number(item.volume) || 0
        }))

      if (validHistoricalData.length === 0) {
        throw new Error('No valid historical data available')
      }

      const combinedData = [...validHistoricalData]

      // Add earnings data to historical points
      if (stockData.earnings && Array.isArray(stockData.earnings)) {
        stockData.earnings.forEach((earning) => {
          if (earning && earning.date) {
            const matchingDataPoint = combinedData.find((item) => item.date === earning.date)
            if (matchingDataPoint) {
              matchingDataPoint.isEarningsDate = true
              matchingDataPoint.expectedMove = Number(earning.expectedMove) || 0
            }
          }
        })
      }

      // Add predicted data
      if (predictedData && Array.isArray(predictedData)) {
        const validPredictedData = predictedData
          .filter(item => item && item.date && typeof item.predicted === 'number')
          .map(item => ({
            ...item,
            predicted: Number(item.predicted) || 0,
            actual: null
          }))

        combinedData.push(...validPredictedData)
      }

      return combinedData
    } catch (error) {
      console.error('Chart data processing error:', error)
      setChartError('Failed to process chart data')
      return []
    }
  }, [stockData, predictedData])

  const formatDate = useCallback((dateStr: string) => {
    try {
      const date = new Date(dateStr)
      if (isNaN(date.getTime())) {
        return dateStr
      }
      return `${date.getMonth() + 1}/${date.getDate()}`
    } catch (error) {
      return dateStr
    }
  }, [])

  const formatPrice = useCallback((value: number) => {
    try {
      return `${Number(value).toFixed(2)}`
    } catch (error) {
      return '0.00'
    }
  }, [])

  // Custom tooltip to show earnings information
  const CustomTooltip = ({ active, payload, label }: any) => {
    try {
      if (active && payload && payload.length) {
        const data = payload[0].payload
        return (
          <div className="bg-background border border-border p-3 rounded-md shadow-md">
            <p className="font-medium">{new Date(label).toLocaleDateString()}</p>
            {payload.map((entry: any, index: number) => (
              <p key={`item-${index}`} style={{ color: entry.color }}>
                {entry.name}: ${Number(entry.value).toFixed(2)}
              </p>
            ))}
            {data.isEarningsDate && (
              <div className="mt-2 pt-2 border-t border-border">
                <p className="text-amber-500 font-medium">Earnings Date</p>
                {data.expectedMove && <p className="text-sm">Expected Move: ±{Number(data.expectedMove).toFixed(1)}%</p>}
              </div>
            )}
          </div>
        )
      }
      return null
    } catch (error) {
      console.error('Tooltip error:', error)
      return null
    }
  }

  // Custom dot to highlight earnings dates
  const renderDot = useCallback((props: any) => {
    try {
      const { cx, cy, payload, index } = props

      if (payload && payload.isEarningsDate) {
        return (
          <svg key={`dot-${index}`} x={cx - 10} y={cy - 10} width={20} height={20} fill="none" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" fill="#fbbf24" opacity="0.7" />
            <circle cx="10" cy="10" r="4" fill="#f59e0b" />
          </svg>
        )
      }
      // Return an empty svg element instead of null to satisfy TypeScript
      return <svg key={`dot-${index}`} />
    } catch (error) {
      console.error('Dot render error:', error)
      // Return an empty svg element instead of null
      return <svg />
    }
  }, [])

  // Handle mouse move to update active payload and hovered data
  const handleMouseMove = useCallback((props: any) => {
    try {
      if (props && props.activePayload && props.activePayload.length) {
        setActivePayload(props.activePayload)
        setHoveredData(props.activePayload[0].payload)
      }
    } catch (error) {
      console.error('Mouse move error:', error)
    }
  }, [])

  // Handle mouse leave to clear active payload
  const handleMouseLeave = useCallback(() => {
    try {
      setActivePayload(null)
      setHoveredData(null)
    } catch (error) {
      console.error('Mouse leave error:', error)
    }
  }, [])

  const candlestickData = useMemo(() => {
    try {
      if (!stockData || !stockData.historical || !Array.isArray(stockData.historical)) {
        return []
      }

      const validData = stockData.historical.filter(item => 
        item && item.date && 
        typeof item.open === 'number' && 
        typeof item.high === 'number' && 
        typeof item.low === 'number' && 
        typeof item.close === 'number'
      )

      const combinedData = [...validData]

      if (predictedData && Array.isArray(predictedData)) {
        const validPredicted = predictedData.filter(item => 
          item && item.date && typeof item.predicted === 'number'
        )

        validPredicted.forEach((item) => {
          const predicted = Number(item.predicted)
          combinedData.push({
            date: item.date,
            open: predicted,
            high: predicted * 1.02,
            low: predicted * 0.98,
            close: predicted
          })
        })
      }

      return combinedData.map((item) => {
        try {
          return [
            new Date(item.date).getTime(),
            Number(item.open),
            Number(item.high),
            Number(item.low),
            Number(item.close)
          ]
        } catch (error) {
          console.warn('Invalid candlestick data point:', item)
          return null
        }
      }).filter(Boolean)
    } catch (error) {
      console.error('Candlestick data error:', error)
      return []
    }
  }, [stockData, predictedData])

  const candlestickOptions = useMemo(() => {
    try {
      const isDark = theme === "dark"
      
      return {
        chart: {
          backgroundColor: isDark ? "#0a0a0a" : "#ffffff",
          style: {
            fontFamily: "'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
          },
          plotBorderColor: isDark ? "#374151" : "#e5e7eb"
        },
        
        rangeSelector: {
          selected: 1,
          buttonTheme: {
            fill: isDark ? "#374151" : "#f3f4f6",
            stroke: isDark ? "#6b7280" : "#d1d5db",
            style: {
              color: isDark ? "#f9fafb" : "#111827"
            },
            states: {
              hover: {
                fill: isDark ? "#4b5563" : "#e5e7eb",
                style: {
                  color: isDark ? "#ffffff" : "#000000"
                }
              },
              select: {
                fill: isDark ? "#3b82f6" : "#2563eb",
                style: {
                  color: "#ffffff"
                }
              }
            }
          },
          inputBoxBorderColor: isDark ? "#6b7280" : "#d1d5db",
          inputStyle: {
            backgroundColor: isDark ? "#374151" : "#ffffff",
            color: isDark ? "#f9fafb" : "#111827",
            border: `1px solid ${isDark ? "#6b7280" : "#d1d5db"}`
          },
          labelStyle: {
            color: isDark ? "#d1d5db" : "#6b7280"
          }
        },
        
        title: {
          text: `${ticker} Stock Price`,
          style: {
            color: isDark ? "#f9fafb" : "#111827",
            fontSize: "18px",
            fontWeight: "600"
          }
        },
        
        xAxis: {
          type: 'datetime',
          labels: {
            style: {
              color: isDark ? "#d1d5db" : "#6b7280",
              fontSize: "12px"
            }
          },
          lineColor: isDark ? "#4b5563" : "#d1d5db",
          tickColor: isDark ? "#4b5563" : "#d1d5db",
          gridLineColor: isDark ? "#374151" : "#f3f4f6"
        },
        
        yAxis: {
          labels: {
            style: {
              color: isDark ? "#d1d5db" : "#6b7280",
              fontSize: "12px"
            }
          },
          gridLineColor: isDark ? "#374151" : "#f3f4f6",
          lineColor: isDark ? "#4b5563" : "#d1d5db",
          tickColor: isDark ? "#4b5563" : "#d1d5db"
        },
        
        navigator: {
          maskFill: isDark ? "rgba(55, 65, 81, 0.3)" : "rgba(229, 231, 235, 0.3)",
          series: {
            color: isDark ? "#60a5fa" : "#3b82f6",
            lineColor: isDark ? "#60a5fa" : "#3b82f6"
          },
          xAxis: {
            gridLineColor: isDark ? "#374151" : "#f3f4f6"
          }
        },
        
        scrollbar: {
          barBackgroundColor: isDark ? "#4b5563" : "#e5e7eb",
          barBorderColor: isDark ? "#6b7280" : "#d1d5db",
          buttonBackgroundColor: isDark ? "#374151" : "#f3f4f6",
          buttonBorderColor: isDark ? "#6b7280" : "#d1d5db",
          buttonArrowColor: isDark ? "#d1d5db" : "#6b7280",
          trackBackgroundColor: isDark ? "#1f2937" : "#f9fafb",
          trackBorderColor: isDark ? "#374151" : "#e5e7eb"
        },
        
        legend: {
          itemStyle: {
            color: isDark ? "#d1d5db" : "#374151"
          },
          itemHoverStyle: {
            color: isDark ? "#f9fafb" : "#111827"
          }
        },
        
        tooltip: {
          backgroundColor: isDark ? "#1f2937" : "#ffffff",
          borderColor: isDark ? "#4b5563" : "#d1d5db",
          style: {
            color: isDark ? "#f9fafb" : "#111827"
          }
        },
        
        series: [
          {
            type: "candlestick",
            name: ticker,
            data: candlestickData,
            tooltip: {
              valueDecimals: 2
            },
            // Bullish (up) candles - green
            upColor: isDark ? "#10b981" : "#059669", // Green for gains
            upLineColor: isDark ? "#059669" : "#047857",
            
            // Bearish (down) candles - red  
            color: isDark ? "#ef4444" : "#dc2626", // Red for losses
            lineColor: isDark ? "#dc2626" : "#b91c1c",
            
            dataGrouping: {
              enabled: false
            }
          }
        ],
        
        // Additional dark mode styling
        plotOptions: {
          candlestick: {
            lineWidth: 1,
            upLineColor: isDark ? "#059669" : "#047857",
            upColor: isDark ? "#10b981" : "#059669",
            color: isDark ? "#ef4444" : "#dc2626",
            lineColor: isDark ? "#dc2626" : "#b91c1c"
          }
        },
        
        // Credits styling
        credits: {
          style: {
            color: isDark ? "#6b7280" : "#9ca3af"
          }
        }
      }
    } catch (error) {
      console.error('Candlestick options error:', error)
      return {}
    }
  }, [theme, ticker, candlestickData])

  // Show loading state
  if (isLoading) {
    return <ChartLoading />
  }

  // Show error state
  if (error || chartError) {
    return <ChartError error={error || chartError || 'Unknown error'} />
  }

  // Show no data state
  if (!stockData || !chartData || chartData.length === 0) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          No chart data available for {ticker}
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <ErrorBoundary fallback={<ChartError error="Chart component crashed" />}>
      <div className="flex flex-col h-full">
        {/* Real-time price display */}
        {realTimePrice && (
          <div className="flex items-center justify-between mb-4 p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-slate-600 dark:text-slate-400">LIVE</span>
              </div>
              <div>
                <span className="text-2xl font-bold text-slate-900 dark:text-white">
                  ${realTimePrice.toFixed(2)}
                </span>
                {priceChange && (
                  <span className={`ml-2 text-sm font-medium ${
                    priceChange.value >= 0 
                      ? 'text-green-600 dark:text-green-400' 
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    {priceChange.value >= 0 ? '+' : ''}
                    {priceChange.value.toFixed(2)} ({priceChange.percentage.toFixed(2)}%)
                  </span>
                )}
              </div>
            </div>
            <div className="text-xs text-slate-500">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        )}

        {/* WebSocket error display */}
        {wsError && (
          <Alert variant="destructive" className="mb-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{wsError}</AlertDescription>
          </Alert>
        )}

        {/* Data source and prediction indicator */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            {/* Data source indicator */}
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${stockData?.realTime ? 'bg-green-500' : 'bg-orange-500'}`}></div>
              <span className="text-xs text-slate-600 dark:text-slate-400">
                {stockData?.realTime ? 'Real-time data' : 'Demo data'}
              </span>
            </div>
            
            {/* AI Prediction indicator */}
            {predictedData && predictedData.length > 0 && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-pink-500 animate-pulse"></div>
                <span className="text-xs text-pink-600 dark:text-pink-400 font-medium">
                  AI Prediction Active ({predictedData.length} days forecasted)
                </span>
              </div>
            )}
          </div>
          
          <button
            className="px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600"
            onClick={() => setIsCandlestickView(!isCandlestickView)}
          >
            {isCandlestickView ? "Switch to Area Chart" : "Switch to Candlestick Chart"}
          </button>
        </div>

        <div className="flex-grow">
          <ErrorBoundary fallback={<ChartError error="Chart rendering failed" />}>
            {isCandlestickView ? (
              <HighchartsReact 
                highcharts={Highcharts} 
                constructorType={"stockChart"} 
                options={candlestickOptions} 
              />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={chartData}
                  margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
                  onMouseMove={handleMouseMove}
                  onMouseLeave={handleMouseLeave}
                >
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={richGreenColor} stopOpacity={0.8} />
                      <stop offset="95%" stopColor={richGreenColor} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" tickFormatter={formatDate} minTickGap={30} tick={{ fontSize: 12 }} />
                  <YAxis domain={['auto', 'auto']} tickFormatter={formatPrice} />
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <Tooltip
                    content={<CustomTooltip />}
                    cursor={<CustomCursor payload={activePayload} />}
                    isAnimationActive={false}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="close"
                    name="Historical"
                    stroke={richGreenColor}
                    fillOpacity={1}
                    fill="url(#colorActual)"
                    strokeWidth={2}
                    dot={renderDot}
                    activeDot={{ r: 6, stroke: richGreenColor, strokeWidth: 2, fill: "#fff" }}
                  />
                  <Area
                    type="monotone"
                    dataKey="predicted"
                    name="AI Prediction"
                    stroke="#f43f5e"
                    fillOpacity={0.3}
                    fill="url(#colorPredicted)"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={renderDot}
                    activeDot={{ r: 6, stroke: "#f43f5e", strokeWidth: 2, fill: "#fff" }}
                  />
                  <Brush dataKey="date" height={30} stroke="#8884d8" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </ErrorBoundary>
        </div>

        {/* Day details panel */}
        <ErrorBoundary fallback={<div className="text-sm text-muted-foreground p-4">Day details unavailable</div>}>
          <DayDetails data={hoveredData} ticker={ticker} />
        </ErrorBoundary>
      </div>
    </ErrorBoundary>
  )
}
