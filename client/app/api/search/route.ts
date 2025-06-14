import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const query = searchParams.get('q')

  if (!query || query.trim().length === 0) {
    return NextResponse.json({ 
      success: false, 
      error: 'Query parameter is required' 
    }, { status: 400 })
  }

  // Comprehensive stock database for reliable search
  const stockDatabase = [
    // US Tech Giants
    { value: "AAPL", label: "AAPL - Apple Inc.", sector: "Technology" },
    { value: "MSFT", label: "MSFT - Microsoft Corporation", sector: "Technology" },
    { value: "GOOGL", label: "GOOGL - Alphabet Inc. Class A", sector: "Technology" },
    { value: "GOOG", label: "GOOG - Alphabet Inc. Class C", sector: "Technology" },
    { value: "AMZN", label: "AMZN - Amazon.com Inc.", sector: "Consumer Discretionary" },
    { value: "META", label: "META - Meta Platforms Inc.", sector: "Technology" },
    { value: "TSLA", label: "TSLA - Tesla Inc.", sector: "Consumer Discretionary" },
    { value: "NVDA", label: "NVDA - NVIDIA Corporation", sector: "Technology" },
    { value: "NFLX", label: "NFLX - Netflix Inc.", sector: "Communication Services" },
    { value: "ADBE", label: "ADBE - Adobe Inc.", sector: "Technology" },
    { value: "CRM", label: "CRM - Salesforce Inc.", sector: "Technology" },
    { value: "ORCL", label: "ORCL - Oracle Corporation", sector: "Technology" },
    { value: "INTC", label: "INTC - Intel Corporation", sector: "Technology" },
    { value: "CSCO", label: "CSCO - Cisco Systems Inc.", sector: "Technology" },
    { value: "AMD", label: "AMD - Advanced Micro Devices Inc.", sector: "Technology" },
    { value: "IBM", label: "IBM - International Business Machines Corp.", sector: "Technology" },
    
    // US Financial
    { value: "JPM", label: "JPM - JPMorgan Chase & Co.", sector: "Financial Services" },
    { value: "BAC", label: "BAC - Bank of America Corp.", sector: "Financial Services" },
    { value: "WFC", label: "WFC - Wells Fargo & Co.", sector: "Financial Services" },
    { value: "GS", label: "GS - Goldman Sachs Group Inc.", sector: "Financial Services" },
    { value: "MS", label: "MS - Morgan Stanley", sector: "Financial Services" },
    { value: "V", label: "V - Visa Inc.", sector: "Financial Services" },
    { value: "MA", label: "MA - Mastercard Inc.", sector: "Financial Services" },
    { value: "PYPL", label: "PYPL - PayPal Holdings Inc.", sector: "Financial Services" },
    { value: "AXP", label: "AXP - American Express Co.", sector: "Financial Services" },
    { value: "BLK", label: "BLK - BlackRock Inc.", sector: "Financial Services" },
    
    // US Healthcare & Pharma
    { value: "JNJ", label: "JNJ - Johnson & Johnson", sector: "Healthcare" },
    { value: "PFE", label: "PFE - Pfizer Inc.", sector: "Healthcare" },
    { value: "ABBV", label: "ABBV - AbbVie Inc.", sector: "Healthcare" },
    { value: "MRK", label: "MRK - Merck & Co. Inc.", sector: "Healthcare" },
    { value: "TMO", label: "TMO - Thermo Fisher Scientific Inc.", sector: "Healthcare" },
    { value: "ABT", label: "ABT - Abbott Laboratories", sector: "Healthcare" },
    { value: "MDT", label: "MDT - Medtronic PLC", sector: "Healthcare" },
    { value: "UNH", label: "UNH - UnitedHealth Group Inc.", sector: "Healthcare" },
    { value: "LLY", label: "LLY - Eli Lilly and Co.", sector: "Healthcare" },
    { value: "BMY", label: "BMY - Bristol-Myers Squibb Co.", sector: "Healthcare" },
    
    // US Consumer
    { value: "DIS", label: "DIS - The Walt Disney Company", sector: "Communication Services" },
    { value: "NKE", label: "NKE - Nike Inc.", sector: "Consumer Discretionary" },
    { value: "SBUX", label: "SBUX - Starbucks Corporation", sector: "Consumer Discretionary" },
    { value: "MCD", label: "MCD - McDonald's Corporation", sector: "Consumer Discretionary" },
    { value: "KO", label: "KO - The Coca-Cola Company", sector: "Consumer Staples" },
    { value: "PEP", label: "PEP - PepsiCo Inc.", sector: "Consumer Staples" },
    { value: "PG", label: "PG - Procter & Gamble Co.", sector: "Consumer Staples" },
    { value: "WMT", label: "WMT - Walmart Inc.", sector: "Consumer Staples" },
    { value: "HD", label: "HD - Home Depot Inc.", sector: "Consumer Discretionary" },
    { value: "LOW", label: "LOW - Lowe's Companies Inc.", sector: "Consumer Discretionary" },
    
    // US Energy & Industrials
    { value: "XOM", label: "XOM - Exxon Mobil Corporation", sector: "Energy" },
    { value: "CVX", label: "CVX - Chevron Corporation", sector: "Energy" },
    { value: "COP", label: "COP - ConocoPhillips", sector: "Energy" },
    { value: "SLB", label: "SLB - Schlumberger NV", sector: "Energy" },
    { value: "BA", label: "BA - Boeing Co.", sector: "Industrials" },
    { value: "CAT", label: "CAT - Caterpillar Inc.", sector: "Industrials" },
    { value: "GE", label: "GE - General Electric Co.", sector: "Industrials" },
    { value: "MMM", label: "MMM - 3M Co.", sector: "Industrials" },
    
    // International Stocks
    { value: "BABA", label: "BABA - Alibaba Group Holding Limited", sector: "Consumer Discretionary" },
    { value: "TSM", label: "TSM - Taiwan Semiconductor Manufacturing Co.", sector: "Technology" },
    { value: "ASML", label: "ASML - ASML Holding NV", sector: "Technology" },
    { value: "NVO", label: "NVO - Novo Nordisk A/S", sector: "Healthcare" },
    { value: "SAP", label: "SAP - SAP SE", sector: "Technology" },
    { value: "TM", label: "TM - Toyota Motor Corporation", sector: "Consumer Discretionary" },
    { value: "NVS", label: "NVS - Novartis AG", sector: "Healthcare" },
    { value: "UL", label: "UL - Unilever PLC", sector: "Consumer Staples" },
    
    // Popular ETFs
    { value: "SPY", label: "SPY - SPDR S&P 500 ETF Trust", sector: "ETF" },
    { value: "QQQ", label: "QQQ - Invesco QQQ Trust ETF", sector: "ETF" },
    { value: "IWM", label: "IWM - iShares Russell 2000 ETF", sector: "ETF" },
    { value: "VTI", label: "VTI - Vanguard Total Stock Market ETF", sector: "ETF" },
    { value: "VOO", label: "VOO - Vanguard S&P 500 ETF", sector: "ETF" },
    { value: "DIA", label: "DIA - SPDR Dow Jones Industrial Average ETF", sector: "ETF" },
    { value: "GLD", label: "GLD - SPDR Gold Shares", sector: "ETF" },
    { value: "SLV", label: "SLV - iShares Silver Trust", sector: "ETF" },
    
    // Crypto-related stocks
    { value: "COIN", label: "COIN - Coinbase Global Inc.", sector: "Financial Services" },
    { value: "MSTR", label: "MSTR - MicroStrategy Inc.", sector: "Technology" },
    { value: "SQ", label: "SQ - Block Inc.", sector: "Financial Services" },
    { value: "RIOT", label: "RIOT - Riot Platforms Inc.", sector: "Financial Services" },
    { value: "MARA", label: "MARA - Marathon Digital Holdings Inc.", sector: "Financial Services" },
    
    // Growth/Emerging Stocks
    { value: "PLTR", label: "PLTR - Palantir Technologies Inc.", sector: "Technology" },
    { value: "SNOW", label: "SNOW - Snowflake Inc.", sector: "Technology" },
    { value: "RBLX", label: "RBLX - Roblox Corporation", sector: "Communication Services" },
    { value: "UBER", label: "UBER - Uber Technologies Inc.", sector: "Technology" },
    { value: "LYFT", label: "LYFT - Lyft Inc.", sector: "Technology" },
    { value: "ZM", label: "ZM - Zoom Video Communications Inc.", sector: "Technology" },
    { value: "DOCU", label: "DOCU - DocuSign Inc.", sector: "Technology" },
    { value: "TWLO", label: "TWLO - Twilio Inc.", sector: "Technology" },
    { value: "OKTA", label: "OKTA - Okta Inc.", sector: "Technology" },
    { value: "SHOP", label: "SHOP - Shopify Inc.", sector: "Technology" },
    { value: "SQ", label: "SQ - Block Inc.", sector: "Technology" },
    { value: "SPOT", label: "SPOT - Spotify Technology SA", sector: "Communication Services" },
    
    // Additional Popular Stocks
    { value: "BRK.B", label: "BRK.B - Berkshire Hathaway Inc. Class B", sector: "Financial Services" },
    { value: "T", label: "T - AT&T Inc.", sector: "Communication Services" },
    { value: "VZ", label: "VZ - Verizon Communications Inc.", sector: "Communication Services" },
    { value: "CMCSA", label: "CMCSA - Comcast Corporation", sector: "Communication Services" },
    { value: "F", label: "F - Ford Motor Co.", sector: "Consumer Discretionary" },
    { value: "GM", label: "GM - General Motors Co.", sector: "Consumer Discretionary" },
    { value: "AAL", label: "AAL - American Airlines Group Inc.", sector: "Industrials" },
    { value: "DAL", label: "DAL - Delta Air Lines Inc.", sector: "Industrials" },
    { value: "UAL", label: "UAL - United Airlines Holdings Inc.", sector: "Industrials" }
  ]

  try {
    const sanitizedQuery = query.trim().toLowerCase()
    
    // Advanced search logic
    const searchResults = stockDatabase.filter(stock => {
      const symbol = stock.value.toLowerCase()
      const name = stock.label.toLowerCase()
      const sector = stock.sector.toLowerCase()
      
      // Exact symbol match gets highest priority
      if (symbol === sanitizedQuery) return true
      
      // Symbol starts with query
      if (symbol.startsWith(sanitizedQuery)) return true
      
      // Symbol contains query
      if (symbol.includes(sanitizedQuery)) return true
      
      // Company name contains query words
      const queryWords = sanitizedQuery.split(' ').filter(word => word.length > 2)
      if (queryWords.some(word => name.includes(word))) return true
      
      // Sector match
      if (sector.includes(sanitizedQuery)) return true
      
      return false
    })
    
    // Sort by relevance
    const sortedResults = searchResults.sort((a, b) => {
      const aSymbol = a.value.toLowerCase()
      const bSymbol = b.value.toLowerCase()
      
      // Exact matches first
      if (aSymbol === sanitizedQuery && bSymbol !== sanitizedQuery) return -1
      if (bSymbol === sanitizedQuery && aSymbol !== sanitizedQuery) return 1
      
      // Starts with query second
      const aStartsWith = aSymbol.startsWith(sanitizedQuery)
      const bStartsWith = bSymbol.startsWith(sanitizedQuery)
      if (aStartsWith && !bStartsWith) return -1
      if (bStartsWith && !aStartsWith) return 1
      
      // Alphabetical order for remaining
      return aSymbol.localeCompare(bSymbol)
    })

    // Return top 20 results
    const results = sortedResults.slice(0, 20).map(stock => ({
      value: stock.value,
      label: stock.label
    }))

    if (results.length > 0) {
      return NextResponse.json({
        success: true,
        results,
        source: 'Local Database',
        total: results.length
      })
    }

    // If no results found, return popular stocks as suggestions
    const popularSuggestions = stockDatabase.slice(0, 10).map(stock => ({
      value: stock.value,
      label: stock.label
    }))

    return NextResponse.json({
      success: true,
      results: popularSuggestions,
      source: 'Suggestions',
      message: `No results found for "${query}". Here are some popular stocks:`
    })

  } catch (error) {
    console.error('Search API error:', error)
    
    // Fallback to basic popular stocks
    const fallbackStocks = [
      { value: "AAPL", label: "AAPL - Apple Inc." },
      { value: "MSFT", label: "MSFT - Microsoft Corporation" },
      { value: "GOOGL", label: "GOOGL - Alphabet Inc." },
      { value: "AMZN", label: "AMZN - Amazon.com Inc." },
      { value: "META", label: "META - Meta Platforms Inc." },
      { value: "TSLA", label: "TSLA - Tesla Inc." },
      { value: "NVDA", label: "NVDA - NVIDIA Corporation" },
      { value: "NFLX", label: "NFLX - Netflix Inc." },
      { value: "JPM", label: "JPM - JPMorgan Chase & Co." },
      { value: "V", label: "V - Visa Inc." }
    ]
    
    return NextResponse.json({
      success: true,
      results: fallbackStocks,
      source: 'Fallback'
    })
  }
}