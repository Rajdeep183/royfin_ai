# Real-Time Stock Data Integration Guide

## Quick Setup (Choose One Option)

### Option 1: Alpha Vantage (Free - 25 requests/day)
1. Sign up at: https://www.alphavantage.co/support/#api-key
2. Get your free API key
3. Add to `.env.local`: `NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY=your_key_here`

### Option 2: Finnhub (Free - 60 calls/minute)
1. Sign up at: https://finnhub.io/register
2. Get your free API key
3. Add to `.env.local`: `NEXT_PUBLIC_FINNHUB_API_KEY=your_key_here`

### Option 3: Yahoo Finance (Unofficial - Unlimited but may break)
1. No API key needed
2. Uses yfinance library
3. Less reliable but free

## Features Implemented

✅ **Real-time price updates** - Live WebSocket connections
✅ **Historical data** - Up to 3 years of data
✅ **Earnings integration** - Quarterly earnings data
✅ **Fallback system** - Automatically uses demo data if API fails
✅ **Rate limiting** - Handles API limits gracefully
✅ **Multi-market support** - US, Indian, and international stocks

## Usage

The system automatically:
- Tries to fetch real data first
- Falls back to realistic demo data if APIs fail
- Shows live price updates when available
- Displays data source indicators

## API Limits

- **Alpha Vantage**: 25 requests/day (free), 500/day (paid)
- **Finnhub**: 60 calls/minute (free), 600/minute (paid)
- **Yahoo Finance**: Unlimited but unofficial

## Next Steps

1. Get API keys from your preferred provider
2. Add them to `.env.local`
3. Restart your development server
4. Test with real stock symbols like AAPL, MSFT, GOOGL