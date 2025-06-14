import { NextRequest, NextResponse } from 'next/server'
import { predictStockData, StockAPIError } from '@/lib/stock-api'

export const maxDuration = 25

export async function POST(request: NextRequest) {
  try {
    // Parse request body with size limit
    const body = await request.json()
    
    const { stockData, days = 30, aggressiveness = 50 } = body

    // Input validation
    if (!stockData) {
      return NextResponse.json(
        { 
          error: 'Missing stock data',
          code: 'MISSING_STOCK_DATA',
          success: false 
        },
        { status: 400 }
      )
    }

    // Validate parameters
    const validatedDays = Math.min(Math.max(1, Math.floor(days)), 90)
    const validatedAggressiveness = Math.min(Math.max(0, Math.floor(aggressiveness)), 100)

    console.log(`Generating ${validatedDays} day prediction with ${validatedAggressiveness}% aggressiveness`)

    // Add timeout wrapper
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => {
        reject(new StockAPIError(
          'Prediction timeout - operation took too long',
          'PREDICTION_TIMEOUT',
          408
        ))
      }, 23000)
    })

    const predictionPromise = predictStockData(stockData, validatedDays, validatedAggressiveness)
    
    const predictions = await Promise.race([predictionPromise, timeoutPromise])

    return NextResponse.json({
      success: true,
      predictions,
      parameters: {
        days: validatedDays,
        aggressiveness: validatedAggressiveness
      },
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error('Prediction API error:', error)

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

export async function GET() {
  return NextResponse.json(
    { error: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' },
    { status: 405 }
  )
}