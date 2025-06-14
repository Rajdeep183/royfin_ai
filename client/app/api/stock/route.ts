import { NextRequest, NextResponse } from 'next/server'
import { getStockData, StockAPIError } from '@/lib/stock-api'

// Prevent FUNCTION_INVOCATION_TIMEOUT by setting max duration
export const maxDuration = 25 // 25 seconds (under Vercel's limit)

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const ticker = searchParams.get('ticker')
    const timeRange = searchParams.get('timeRange') || '6m'

    // Input validation
    if (!ticker) {
      return NextResponse.json(
        { 
          error: 'Missing ticker parameter',
          code: 'MISSING_TICKER',
          success: false 
        },
        { status: 400 }
      )
    }

    // Sanitize inputs
    const sanitizedTicker = ticker.trim().toUpperCase().slice(0, 10)
    const validTimeRanges = ['7d', '1m', '3m', '6m', '1y', '3y']
    const sanitizedTimeRange = validTimeRanges.includes(timeRange) ? timeRange : '6m'

    console.log(`Fetching stock data for ${sanitizedTicker} with range ${sanitizedTimeRange}`)

    // Add timeout wrapper
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => {
        reject(new StockAPIError(
          'Request timeout - operation took too long',
          'REQUEST_TIMEOUT',
          408
        ))
      }, 23000) // 23 seconds to leave buffer
    })

    const dataPromise = getStockData(sanitizedTicker, sanitizedTimeRange)
    
    const stockData = await Promise.race([dataPromise, timeoutPromise])

    // Ensure response isn't too large (prevent FUNCTION_RESPONSE_PAYLOAD_TOO_LARGE)
    const responseSize = JSON.stringify(stockData).length
    if (responseSize > 4.5 * 1024 * 1024) { // 4.5MB limit (Vercel's is 5MB)
      console.warn(`Response too large: ${responseSize} bytes, truncating data`)
      
      // Truncate historical data if needed
      if (stockData.historical && stockData.historical.length > 100) {
        stockData.historical = stockData.historical.slice(-100)
      }
    }

    return NextResponse.json({
      success: true,
      data: stockData,
      ticker: sanitizedTicker,
      timeRange: sanitizedTimeRange,
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error('Stock data API error:', error)

    if (error instanceof StockAPIError) {
      return NextResponse.json(
        {
          error: error.message,
          code: error.code,
          success: false,
          timestamp: new Date().toISOString()
        },
        { status: error.statusCode || 500 }
      )
    }

    // Handle unexpected errors
    return NextResponse.json(
      {
        error: 'Internal server error',
        code: 'INTERNAL_ERROR',
        success: false,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    )
  }
}

// Handle unsupported methods
export async function POST() {
  return NextResponse.json(
    { error: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' },
    { status: 405 }
  )
}

export async function PUT() {
  return NextResponse.json(
    { error: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' },
    { status: 405 }
  )
}

export async function DELETE() {
  return NextResponse.json(
    { error: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' },
    { status: 405 }
  )
}