#!/usr/bin/env python3
"""
Enhanced Data Ingestion Script for RoyFin AI
Fetches market data from multiple sources using yfinance
Supports individual stocks, indices, and volatility data
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestionEngine:
    """Enhanced data ingestion engine with multi-source support"""
    
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = output_dir
        self.create_directories()
        
        # Define data sources
        self.indices = {
            'SPY': 'SPDR S&P 500 ETF',
            '^GSPC': 'S&P 500 Index',
            '^DJI': 'Dow Jones Industrial Average',
            '^IXIC': 'NASDAQ Composite',
            '^RUT': 'Russell 2000',
            'QQQ': 'Invesco QQQ Trust'
        }
        
        self.volatility_indices = {
            '^VIX': 'CBOE Volatility Index',
            '^VXN': 'NASDAQ 100 Volatility Index',
            '^RVX': 'Russell 2000 Volatility Index'
        }
        
        self.indian_stocks = {
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services',
            'HDFCBANK.NS': 'HDFC Bank',
            'INFY.NS': 'Infosys',
            'HINDUNILVR.NS': 'Hindustan Unilever'
        }
        
        # Popular US stocks
        self.us_stocks = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'AMD': 'Advanced Micro Devices'
        }

    def create_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, 'stocks'),
            os.path.join(self.output_dir, 'indices'),
            os.path.join(self.output_dir, 'volatility'),
            os.path.join(self.output_dir, 'metadata')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def fetch_single_ticker(self, ticker: str, start_date: str, end_date: str, 
                           retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker with retry logic"""
        for attempt in range(retries):
            try:
                logger.info(f"Fetching data for {ticker} (attempt {attempt + 1})")
                
                # Fetch data using yfinance
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True,
                    prepost=True,
                    repair=True
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None
                
                # Add metadata
                df['Ticker'] = ticker
                df['Fetch_Date'] = datetime.now().isoformat()
                
                # Calculate basic features
                df = self.add_basic_features(df)
                
                logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {ticker} on attempt {attempt + 1}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None

    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical features to the data"""
        try:
            # Price-based features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            
            # Volatility measures
            df['Volatility_5d'] = df['Returns'].rolling(window=5).std() * np.sqrt(252)
            df['Volatility_20d'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Price moving averages
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # Momentum indicators
            df['RSI_14'] = self.calculate_rsi(df['Close'], window=14)
            df['Price_Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
            
        except Exception as e:
            logger.warning(f"Error adding basic features: {str(e)}")
        
        return df

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(index=prices.index, dtype=float)

    def fetch_multiple_tickers(self, tickers: List[str], start_date: str, 
                              end_date: str, max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.fetch_single_ticker, ticker, start_date, end_date): ticker
                for ticker in tickers
            }
            
            # Collect results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        results[ticker] = data
                        logger.info(f"✓ Completed fetching {ticker}")
                    else:
                        logger.error(f"✗ Failed to fetch {ticker}")
                except Exception as e:
                    logger.error(f"✗ Exception fetching {ticker}: {str(e)}")
        
        return results

    def save_data(self, data_dict: Dict[str, pd.DataFrame], data_type: str = "stocks"):
        """Save data to CSV files with proper organization"""
        saved_files = []
        
        for ticker, df in data_dict.items():
            try:
                # Determine subdirectory based on data type
                if data_type == "indices":
                    subdir = "indices"
                elif data_type == "volatility":
                    subdir = "volatility"
                else:
                    subdir = "stocks"
                
                # Create filename
                safe_ticker = ticker.replace("^", "").replace(".", "_")
                filename = f"{safe_ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = os.path.join(self.output_dir, subdir, filename)
                
                # Save to CSV
                df.to_csv(filepath, index=True)
                saved_files.append(filepath)
                logger.info(f"Saved {ticker} data to {filepath}")
                
                # Save metadata
                metadata = {
                    'ticker': ticker,
                    'data_type': data_type,
                    'rows': len(df),
                    'start_date': df.index.min().isoformat() if len(df) > 0 else None,
                    'end_date': df.index.max().isoformat() if len(df) > 0 else None,
                    'columns': list(df.columns),
                    'file_path': filepath,
                    'created_at': datetime.now().isoformat()
                }
                
                metadata_file = os.path.join(
                    self.output_dir, 
                    "metadata", 
                    f"{safe_ticker}_metadata.json"
                )
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error saving {ticker}: {str(e)}")
        
        return saved_files

    def create_ingestion_summary(self, all_results: Dict[str, Dict[str, pd.DataFrame]]):
        """Create a summary of the data ingestion process"""
        summary = {
            'ingestion_timestamp': datetime.now().isoformat(),
            'total_tickers': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'data_types': {},
            'date_range': {},
            'file_count': 0
        }
        
        for data_type, results in all_results.items():
            summary['data_types'][data_type] = {
                'requested': len(results) if results else 0,
                'successful': len([r for r in results.values() if r is not None]) if results else 0
            }
            
            if results:
                # Calculate date range
                all_dates = []
                for df in results.values():
                    if df is not None and len(df) > 0:
                        all_dates.extend([df.index.min(), df.index.max()])
                
                if all_dates:
                    summary['date_range'][data_type] = {
                        'start': min(all_dates).isoformat(),
                        'end': max(all_dates).isoformat()
                    }
        
        # Calculate totals
        for data_type_stats in summary['data_types'].values():
            summary['total_tickers'] += data_type_stats['requested']
            summary['successful_fetches'] += data_type_stats['successful']
        
        summary['failed_fetches'] = summary['total_tickers'] - summary['successful_fetches']
        
        # Count files
        try:
            for root, dirs, files in os.walk(self.output_dir):
                summary['file_count'] += len([f for f in files if f.endswith('.csv')])
        except Exception:
            summary['file_count'] = 0
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "ingestion_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Ingestion summary saved to {summary_file}")
        logger.info(f"Summary: {summary['successful_fetches']}/{summary['total_tickers']} successful")
        
        return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Data Ingestion for RoyFin AI")
    
    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated list of tickers to fetch'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for data files'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d'),  # 3 years
        help='Start date for data fetching (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for data fetching (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--include-indices',
        action='store_true',
        help='Include market indices data'
    )
    
    parser.add_argument(
        '--include-volatility',
        action='store_true',
        help='Include volatility indices data'
    )
    
    parser.add_argument(
        '--include-indian-stocks',
        action='store_true',
        help='Include Indian stock market data'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of concurrent workers'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Starting enhanced data ingestion process")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Initialize ingestion engine
    engine = DataIngestionEngine(args.output_dir)
    
    # Parse tickers
    requested_tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    logger.info(f"Requested tickers: {requested_tickers}")
    
    all_results = {}
    
    # Fetch requested tickers
    if requested_tickers:
        logger.info("Fetching requested tickers...")
        results = engine.fetch_multiple_tickers(
            requested_tickers, 
            args.start_date, 
            args.end_date,
            args.max_workers
        )
        engine.save_data(results, "stocks")
        all_results['stocks'] = results
    
    # Fetch indices if requested
    if args.include_indices:
        logger.info("Fetching market indices...")
        indices_results = engine.fetch_multiple_tickers(
            list(engine.indices.keys()), 
            args.start_date, 
            args.end_date,
            args.max_workers
        )
        engine.save_data(indices_results, "indices")
        all_results['indices'] = indices_results
    
    # Fetch volatility indices if requested
    if args.include_volatility:
        logger.info("Fetching volatility indices...")
        vol_results = engine.fetch_multiple_tickers(
            list(engine.volatility_indices.keys()), 
            args.start_date, 
            args.end_date,
            args.max_workers
        )
        engine.save_data(vol_results, "volatility")
        all_results['volatility'] = vol_results
    
    # Fetch Indian stocks if requested
    if args.include_indian_stocks:
        logger.info("Fetching Indian stocks...")
        indian_results = engine.fetch_multiple_tickers(
            list(engine.indian_stocks.keys()), 
            args.start_date, 
            args.end_date,
            args.max_workers
        )
        engine.save_data(indian_results, "indian_stocks")
        all_results['indian_stocks'] = indian_results
    
    # Create summary
    summary = engine.create_ingestion_summary(all_results)
    
    # Exit with appropriate code
    if summary['failed_fetches'] > 0:
        logger.warning(f"Completed with {summary['failed_fetches']} failures")
        sys.exit(1 if summary['failed_fetches'] > summary['successful_fetches'] else 0)
    else:
        logger.info("Data ingestion completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()