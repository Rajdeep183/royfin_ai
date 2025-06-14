"use client"

import { useState, useEffect, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Combobox } from "@/components/combobox"
import { StockChart } from "@/components/stock-chart"
import { TimeRangeSelector } from "@/components/time-range-selector"
import { getStockData, predictStockData, searchTickers } from "@/lib/stock-api"
import { Loader2, TrendingUp, Settings, Calendar, Target } from "lucide-react"

export default function PredictPage() {
  const [selectedTicker, setSelectedTicker] = useState("")
  const [stockData, setStockData] = useState<any>(null)
  const [predictedData, setPredictedData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [aggressiveness, setAggressiveness] = useState(50)
  const [predictionDays, setPredictionDays] = useState(30)
  const [timeRange, setTimeRange] = useState("6m") // Default to 6 months
  const [error, setError] = useState("")
  const [tickerOptions, setTickerOptions] = useState<{ value: string; label: string }[]>([])

  // Load default ticker options on component mount
  useEffect(() => {
    const loadDefaultTickers = async () => {
      try {
        const defaultOptions = await searchTickers("")
        setTickerOptions(defaultOptions)
      } catch (error) {
        console.error("Failed to load default tickers:", error)
        // Fallback to hardcoded popular stocks
        setTickerOptions([
          { value: "AAPL", label: "AAPL - Apple Inc." },
          { value: "MSFT", label: "MSFT - Microsoft Corporation" },
          { value: "GOOGL", label: "GOOGL - Alphabet Inc." },
          { value: "AMZN", label: "AMZN - Amazon.com Inc." },
          { value: "META", label: "META - Meta Platforms Inc." },
          { value: "TSLA", label: "TSLA - Tesla Inc." },
          { value: "NVDA", label: "NVDA - NVIDIA Corporation" },
          { value: "NFLX", label: "NFLX - Netflix Inc." },
        ])
      }
    }
    loadDefaultTickers()
  }, [])

  useEffect(() => {
    if (selectedTicker) {
      fetchStockData()
    }
  }, [selectedTicker, timeRange]) // Re-fetch when ticker or time range changes

  // Enhanced error handling and retry logic
  const fetchStockData = async () => {
    setLoading(true)
    setError("")
    setPredictedData(null) // Clear previous predictions
    try {
      console.log(`Fetching stock data for ${selectedTicker}...`)
      
      // Fetch stock data first
      const response = await fetch(`/api/stock?ticker=${selectedTicker}&timeRange=${timeRange}`)
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`)
      }
      
      const result = await response.json()
      
      if (!result.success || !result.data) {
        throw new Error(result.error || 'Failed to fetch stock data')
      }
      
      console.log('Stock data fetched successfully:', result.data)
      setStockData(result.data)

      // Automatically generate prediction after stock data is loaded
      if (result.data && result.data.historical && result.data.historical.length > 0) {
        await generatePrediction(result.data)
      }
    } catch (err: any) {
      console.error("Error fetching stock data:", err)
      setStockData(null)
      setPredictedData(null)
      
      let errorMessage = "Failed to fetch stock data. Please try again later."
      
      if (err.message.includes("network") || err.message.includes("fetch")) {
        errorMessage = "Network error: Please check your internet connection and try again."
      } else if (err.message.includes("timeout")) {
        errorMessage = "Request timeout: The server took too long to respond. Please try again."
      } else if (err.message.includes("404")) {
        errorMessage = `Stock symbol "${selectedTicker}" not found. Please check the symbol and try again.`
      } else if (err.message.includes("429")) {
        errorMessage = "Too many requests. Please wait a moment and try again."
      } else if (err.message) {
        errorMessage = err.message
      }
      
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  // Separate function for generating predictions
  const generatePrediction = async (stockDataToUse = stockData) => {
    if (!stockDataToUse || !stockDataToUse.historical || stockDataToUse.historical.length === 0) {
      console.warn("No stock data available for prediction")
      return
    }

    try {
      console.log(`Generating prediction for ${predictionDays} days with ${aggressiveness}% aggressiveness...`)
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stockData: stockDataToUse,
          days: predictionDays,
          aggressiveness: aggressiveness
        })
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        console.warn('Prediction API error:', errorData.error)
        return // Don't throw error, just log warning
      }
      
      const result = await response.json()
      
      if (result.success && result.predictions) {
        console.log('Prediction generated successfully:', result.predictions)
        setPredictedData(result.predictions)
      } else {
        console.warn('Prediction failed:', result.error)
      }
    } catch (err: any) {
      console.warn("Error generating prediction:", err)
      // Don't set error state for prediction failures - just log them
    }
  }

  const handleAggresiveChange = (value: number[]) => {
    setAggressiveness(value[0])
  }

  const handlePredictionDaysChange = (value: number[]) => {
    setPredictionDays(value[0])
  }

  const handleTimeRangeChange = (range: string) => {
    setTimeRange(range)
  }

  // Update the handleGeneratePrediction function
  const handleGeneratePrediction = async () => {
    if (!stockData || !stockData.historical || stockData.historical.length === 0) {
      setError("No stock data available for prediction. Please select a different stock.")
      return
    }

    setLoading(true)
    setError("")
    try {
      await generatePrediction()
    } catch (err: any) {
      console.error("Error generating prediction:", err)
      
      let errorMessage = "Failed to generate prediction. Please try again later."
      
      if (err.message.includes("timeout")) {
        errorMessage = "Prediction timeout: The server took too long to respond. Please try again."
      } else if (err.message.includes("INSUFFICIENT_DATA")) {
        errorMessage = "Insufficient historical data for prediction. Please try a different stock or time range."
      } else if (err.message) {
        errorMessage = err.message
      }
      
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  // Auto-regenerate predictions when parameters change
  useEffect(() => {
    if (stockData && stockData.historical && stockData.historical.length > 0) {
      const timeoutId = setTimeout(() => {
        generatePrediction()
      }, 500) // Debounce prediction generation
      
      return () => clearTimeout(timeoutId)
    }
  }, [aggressiveness, predictionDays]) // Regenerate when these change

  // Updated chartData to filter based on selected time frame
  const chartData = useMemo(() => {
    if (!stockData || !stockData.historical) return []

    // Filter data for the selected time frame
    const endDate = new Date()
    let startDate = new Date()
    switch (timeRange) {
      case "7d":
        startDate.setDate(endDate.getDate() - 7)
        break
      case "1m":
        startDate.setMonth(endDate.getMonth() - 1)
        break
      case "3m":
        startDate.setMonth(endDate.getMonth() - 3)
        break
      case "6m":
        startDate.setMonth(endDate.getMonth() - 6)
        break
      case "1y":
        startDate.setFullYear(endDate.getFullYear() - 1)
        break
      case "3y":
        startDate.setFullYear(endDate.getFullYear() - 3)
        break
      default:
        startDate = new Date(stockData.historical[0].date)
    }

    return stockData.historical.filter((item: any) => {
      const itemDate = new Date(item.date)
      return itemDate >= startDate && itemDate <= endDate
    })
  }, [stockData, timeRange])

  console.log("Stock Data:", stockData);
  console.log("Predicted Data:", predictedData);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-emerald-50 to-slate-50 dark:from-slate-900 dark:via-emerald-800 dark:to-slate-900">
      <div className="container mx-auto py-8 px-4">
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-100/80 dark:bg-emerald-500/20 backdrop-blur-sm rounded-full border border-emerald-200 dark:border-emerald-300/30 mb-6">
            <TrendingUp className="h-4 w-4 text-emerald-600 dark:text-emerald-300" />
            <span className="text-emerald-700 dark:text-emerald-200 text-sm font-medium">AI Market Intelligence Dashboard</span>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-slate-800 via-emerald-700 to-teal-700 dark:from-white dark:via-emerald-200 dark:to-teal-200 bg-clip-text text-transparent mb-4">
            Smart Stock Forecasting
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Leverage neural networks to predict market movements and optimize your investment strategy
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          {/* Control Panel */}
          <div className="xl:col-span-1 space-y-6">
            <Card className="bg-white/70 dark:bg-slate-800/50 backdrop-blur-sm border-slate-200 dark:border-slate-700/50">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                  <CardTitle className="text-slate-800 dark:text-white">Market Selection</CardTitle>
                </div>
                <CardDescription className="text-slate-600 dark:text-slate-400">
                  Choose your target asset for analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Asset Symbol</label>
                  <Combobox 
                    options={tickerOptions} 
                    onSelect={setSelectedTicker} 
                    placeholder="Search for a stock ticker..."
                    isLoading={loading}
                    noOptionsMessage="No tickers found. Try a different search."
                  />
                </div>

                <div className="space-y-3">
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Analysis Timeframe</label>
                  <TimeRangeSelector selectedRange={timeRange} onRangeChange={handleTimeRangeChange} />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/70 dark:bg-slate-800/50 backdrop-blur-sm border-slate-200 dark:border-slate-700/50">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-2">
                  <Settings className="h-5 w-5 text-teal-600 dark:text-teal-400" />
                  <CardTitle className="text-slate-800 dark:text-white">Model Configuration</CardTitle>
                </div>
                <CardDescription className="text-slate-600 dark:text-slate-400">
                  Fine-tune prediction parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Risk Tolerance</label>
                    <span className="text-sm text-emerald-600 dark:text-emerald-400 font-medium">{aggressiveness}%</span>
                  </div>
                  <Slider
                    defaultValue={[50]}
                    max={100}
                    step={1}
                    value={[aggressiveness]}
                    onValueChange={handleAggresiveChange}
                    className="w-full"
                  />
                  <p className="text-xs text-slate-500">
                    Higher values increase prediction sensitivity and potential volatility
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Forecast Horizon</label>
                    <span className="text-sm text-teal-600 dark:text-teal-400 font-medium">{predictionDays} days</span>
                  </div>
                  <Slider
                    defaultValue={[30]}
                    min={7}
                    max={90}
                    step={1}
                    value={[predictionDays]}
                    onValueChange={handlePredictionDaysChange}
                    className="w-full"
                  />
                </div>

                <Button 
                  className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold py-3 rounded-xl shadow-lg" 
                  onClick={handleGeneratePrediction} 
                  disabled={!stockData || loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="mr-2 h-4 w-4" />
                      Generate Forecast
                    </>
                  )}
                </Button>

                {error && (
                  <div className="bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/20 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400 mb-3">{error}</p>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={fetchStockData} 
                      disabled={loading}
                      className="border-red-300 text-red-600 hover:bg-red-50 dark:border-red-500/30 dark:text-red-400 dark:hover:bg-red-500/10"
                    >
                      Retry Analysis
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="bg-white/70 dark:bg-slate-800/50 backdrop-blur-sm border-slate-200 dark:border-slate-700/50">
              <CardContent className="pt-6">
                <div className="text-center space-y-2">
                  <Calendar className="h-8 w-8 text-amber-500 dark:text-amber-400 mx-auto" />
                  <p className="text-xs text-slate-600 dark:text-slate-500 font-medium">Risk Disclaimer</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                    AI forecasts are for informational purposes only. Market predictions involve inherent risks. 
                    Always conduct your own research and consider consulting financial advisors.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Chart Section */}
          <Card className="xl:col-span-3 bg-white/50 dark:bg-slate-800/30 backdrop-blur-sm border-slate-200 dark:border-slate-700/50">
            <CardHeader className="pb-4">
              <CardTitle className="text-slate-800 dark:text-white text-2xl">
                {selectedTicker ? `${selectedTicker} Market Intelligence` : "Market Intelligence Dashboard"}
              </CardTitle>
              <CardDescription className="text-slate-600 dark:text-slate-400">
                {selectedTicker 
                  ? `Advanced neural network analysis for ${selectedTicker} with predictive modeling`
                  : "Select an asset to begin AI-powered market analysis"
                }
              </CardDescription>
            </CardHeader>
            <CardContent className="h-[600px] flex flex-col p-6">
              <div className="flex-grow bg-slate-50 dark:bg-slate-900/50 rounded-xl border border-slate-200 dark:border-slate-700/30 p-4">
                {loading ? (
                  <div className="flex flex-col items-center justify-center h-full">
                    <div className="w-16 h-16 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full flex items-center justify-center mb-4">
                      <Loader2 className="h-8 w-8 animate-spin text-white" />
                    </div>
                    <p className="text-lg font-medium text-slate-800 dark:text-white mb-2">Processing Market Data</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">Neural networks are analyzing patterns...</p>
                  </div>
                ) : !selectedTicker ? (
                  <div className="flex flex-col items-center justify-center h-full">
                    <div className="w-16 h-16 bg-gradient-to-r from-emerald-200 to-teal-200 dark:from-emerald-500/20 dark:to-teal-500/20 rounded-full flex items-center justify-center mb-4">
                      <TrendingUp className="h-8 w-8 text-emerald-600 dark:text-emerald-400" />
                    </div>
                    <p className="text-lg font-medium text-slate-800 dark:text-white mb-2">Ready for Analysis</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">Choose an asset symbol to start forecasting</p>
                  </div>
                ) : !stockData ? (
                  <div className="flex flex-col items-center justify-center h-full">
                    <div className="w-16 h-16 bg-gradient-to-r from-red-200 to-orange-200 dark:from-red-500/20 dark:to-orange-500/20 rounded-full flex items-center justify-center mb-4">
                      <Target className="h-8 w-8 text-red-600 dark:text-red-400" />
                    </div>
                    <p className="text-lg font-medium text-slate-800 dark:text-white mb-2">Data Unavailable</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">No market data found for {selectedTicker}</p>
                  </div>
                ) : (
                  <StockChart stockData={stockData} predictedData={predictedData} ticker={selectedTicker} />
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

// Helper function to get a human-readable label for the time range
function getTimeRangeLabel(range: string): string {
  switch (range) {
    case "7d":
      return "7 Days"
    case "1m":
      return "1 Month"
    case "3m":
      return "3 Months"
    case "6m":
      return "6 Months"
    case "1y":
      return "1 Year"
    case "3y":
      return "3 Years"
    default:
      return "Custom"
  }
}
