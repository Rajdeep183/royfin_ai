#!/usr/bin/env python3
"""
Data Validation Script for RoyFin AI
Validates data quality, completeness, and freshness
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for market data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'tickers_validated': [],
            'passed_validations': [],
            'failed_validations': [],
            'warnings': [],
            'summary': {}
        }
    
    def validate_data_completeness(self, df: pd.DataFrame, ticker: str, 
                                 min_days: int = 200) -> Tuple[bool, List[str]]:
        """Validate data completeness and minimum requirements"""
        issues = []
        
        # Check minimum days
        if len(df) < min_days:
            issues.append(f"Insufficient data: {len(df)} days < {min_days} required")
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for completely empty columns
        for col in required_columns:
            if col in df.columns and df[col].isna().all():
                issues.append(f"Column '{col}' is completely empty")
        
        # Check date index
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index is not a DatetimeIndex")
        
        return len(issues) == 0, issues
    
    def validate_data_quality(self, df: pd.DataFrame, ticker: str, 
                            max_missing_pct: float = 5.0) -> Tuple[bool, List[str]]:
        """Validate data quality and detect anomalies"""
        issues = []
        
        # Check missing data percentage
        missing_pct = (df.isnull().sum() / len(df) * 100)
        for col, pct in missing_pct.items():
            if pct > max_missing_pct:
                issues.append(f"Column '{col}' has {pct:.1f}% missing data (> {max_missing_pct}%)")
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(f"Found {negative_count} non-positive values in '{col}'")
        
        # Check for impossible price relationships
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            # High should be >= Low
            invalid_high_low = (df['High'] < df['Low']).sum()
            if invalid_high_low > 0:
                issues.append(f"Found {invalid_high_low} days where High < Low")
            
            # High and Low should contain Open and Close
            invalid_open = ((df['Open'] > df['High']) | (df['Open'] < df['Low'])).sum()
            if invalid_open > 0:
                issues.append(f"Found {invalid_open} days where Open is outside High-Low range")
            
            invalid_close = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
            if invalid_close > 0:
                issues.append(f"Found {invalid_close} days where Close is outside High-Low range")
        
        # Check for extreme outliers (price changes > 50% in a day)
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()
            extreme_moves = (abs(returns) > 0.5).sum()
            if extreme_moves > len(df) * 0.01:  # More than 1% of days
                issues.append(f"Found {extreme_moves} days with extreme price moves (>50%)")
        
        # Check volume consistency
        if 'Volume' in df.columns:
            zero_volume_days = (df['Volume'] == 0).sum()
            if zero_volume_days > len(df) * 0.1:  # More than 10% of days
                issues.append(f"Found {zero_volume_days} days with zero volume")
        
        return len(issues) == 0, issues
    
    def validate_data_freshness(self, df: pd.DataFrame, ticker: str, 
                              max_age_days: int = 7) -> Tuple[bool, List[str]]:
        """Validate data freshness"""
        issues = []
        
        if len(df) == 0:
            issues.append("No data available for freshness check")
            return False, issues
        
        # Get the latest data date
        latest_date = df.index.max()
        days_old = (datetime.now().date() - latest_date.date()).days
        
        # Check if data is too old
        if days_old > max_age_days:
            issues.append(f"Data is {days_old} days old (> {max_age_days} days)")
        
        # Check for weekend/holiday considerations
        today = datetime.now().date()
        if today.weekday() < 5:  # Monday to Friday
            # If it's a weekday, we expect recent data
            if days_old > 1:
                issues.append(f"Market data is {days_old} days old on a weekday")
        
        return len(issues) == 0, issues
    
    def validate_data_consistency(self, df: pd.DataFrame, ticker: str) -> Tuple[bool, List[str]]:
        """Validate data consistency and detect gaps"""
        issues = []
        
        if len(df) < 2:
            issues.append("Insufficient data for consistency check")
            return False, issues
        
        # Check for large data gaps (more than 7 consecutive missing days)
        date_diff = df.index.to_series().diff()
        large_gaps = (date_diff > timedelta(days=7)).sum()
        if large_gaps > 0:
            issues.append(f"Found {large_gaps} large gaps (>7 days) in data")
        
        # Check for duplicate dates
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            issues.append(f"Found {duplicate_dates} duplicate dates")
        
        # Check chronological order
        if not df.index.is_monotonic_increasing:
            issues.append("Data is not in chronological order")
        
        return len(issues) == 0, issues
    
    def validate_technical_indicators(self, df: pd.DataFrame, ticker: str) -> Tuple[bool, List[str]]:
        """Validate technical indicators if present"""
        issues = []
        warnings = []
        
        # Check RSI bounds
        if 'RSI_14' in df.columns:
            invalid_rsi = ((df['RSI_14'] < 0) | (df['RSI_14'] > 100)).sum()
            if invalid_rsi > 0:
                issues.append(f"Found {invalid_rsi} invalid RSI values (outside 0-100 range)")
        
        # Check moving averages
        ma_columns = [col for col in df.columns if col.startswith('MA_')]
        for ma_col in ma_columns:
            if ma_col in df.columns:
                # Moving averages should not have extreme values
                if df[ma_col].min() <= 0:
                    warnings.append(f"Moving average {ma_col} has non-positive values")
        
        # Check volatility measures
        vol_columns = [col for col in df.columns if 'Volatility' in col or 'Vol' in col]
        for vol_col in vol_columns:
            if vol_col in df.columns:
                # Volatility should be non-negative
                negative_vol = (df[vol_col] < 0).sum()
                if negative_vol > 0:
                    issues.append(f"Found {negative_vol} negative volatility values in {vol_col}")
                
                # Check for extreme volatility (>200%)
                extreme_vol = (df[vol_col] > 2.0).sum()
                if extreme_vol > len(df) * 0.05:  # More than 5% of days
                    warnings.append(f"Found {extreme_vol} days with extreme volatility in {vol_col}")
        
        # Add warnings to global warnings list
        self.validation_results['warnings'].extend(warnings)
        
        return len(issues) == 0, issues
    
    def validate_single_ticker(self, ticker: str, min_days: int = 200, 
                             max_missing_days: int = 5) -> Dict:
        """Validate data for a single ticker"""
        logger.info(f"Validating data for {ticker}")
        
        result = {
            'ticker': ticker,
            'status': 'UNKNOWN',
            'file_found': False,
            'data_loaded': False,
            'validations': {},
            'issues': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Find data file
        data_file = None
        safe_ticker = ticker.replace("^", "").replace(".", "_")
        
        # Search in different subdirectories
        search_paths = [
            self.data_dir / "stocks",
            self.data_dir / "indices", 
            self.data_dir / "volatility",
            self.data_dir
        ]
        
        for search_path in search_paths:
            pattern = f"{safe_ticker}_*.csv"
            files = list(search_path.glob(pattern))
            if files:
                data_file = files[0]  # Take the most recent
                break
        
        if not data_file or not data_file.exists():
            result['issues'].append(f"Data file not found for {ticker}")
            result['status'] = 'FAILED'
            return result
        
        result['file_found'] = True
        result['metadata']['file_path'] = str(data_file)
        result['metadata']['file_size'] = data_file.stat().st_size
        
        # Load data
        try:
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            result['data_loaded'] = True
            result['metadata']['rows'] = len(df)
            result['metadata']['columns'] = list(df.columns)
            result['metadata']['date_range'] = {
                'start': df.index.min().isoformat() if len(df) > 0 else None,
                'end': df.index.max().isoformat() if len(df) > 0 else None
            }
        except Exception as e:
            result['issues'].append(f"Failed to load data: {str(e)}")
            result['status'] = 'FAILED'
            return result
        
        # Run validations
        validations = [
            ('completeness', self.validate_data_completeness, (df, ticker, min_days)),
            ('quality', self.validate_data_quality, (df, ticker, max_missing_days)),
            ('freshness', self.validate_data_freshness, (df, ticker, 7)),
            ('consistency', self.validate_data_consistency, (df, ticker)),
            ('technical_indicators', self.validate_technical_indicators, (df, ticker))
        ]
        
        all_passed = True
        for validation_name, validation_func, args in validations:
            try:
                passed, issues = validation_func(*args)
                result['validations'][validation_name] = {
                    'passed': passed,
                    'issues': issues
                }
                
                if not passed:
                    all_passed = False
                    result['issues'].extend(issues)
                    
            except Exception as e:
                logger.error(f"Validation {validation_name} failed for {ticker}: {str(e)}")
                result['validations'][validation_name] = {
                    'passed': False,
                    'issues': [f"Validation error: {str(e)}"]
                }
                all_passed = False
                result['issues'].append(f"Validation {validation_name} failed: {str(e)}")
        
        result['status'] = 'PASSED' if all_passed else 'FAILED'
        logger.info(f"Validation for {ticker}: {result['status']}")
        
        return result
    
    def validate_all_tickers(self, tickers: List[str], min_days: int = 200, 
                           max_missing_days: int = 5) -> Dict:
        """Validate data for all specified tickers"""
        logger.info(f"Starting validation for {len(tickers)} tickers")
        
        results = []
        passed_count = 0
        failed_count = 0
        
        for ticker in tickers:
            try:
                result = self.validate_single_ticker(ticker, min_days, max_missing_days)
                results.append(result)
                
                if result['status'] == 'PASSED':
                    passed_count += 1
                    self.validation_results['passed_validations'].append(ticker)
                else:
                    failed_count += 1
                    self.validation_results['failed_validations'].append(ticker)
                
                self.validation_results['tickers_validated'].append(ticker)
                
            except Exception as e:
                logger.error(f"Unexpected error validating {ticker}: {str(e)}")
                failed_count += 1
                self.validation_results['failed_validations'].append(ticker)
        
        # Update overall status
        if failed_count == 0:
            self.validation_results['overall_status'] = 'PASSED'
        elif passed_count > failed_count:
            self.validation_results['overall_status'] = 'PARTIAL'
        else:
            self.validation_results['overall_status'] = 'FAILED'
        
        # Create summary
        self.validation_results['summary'] = {
            'total_tickers': len(tickers),
            'passed': passed_count,
            'failed': failed_count,
            'success_rate': passed_count / len(tickers) * 100 if tickers else 0
        }
        
        self.validation_results['detailed_results'] = results
        
        return self.validation_results
    
    def save_validation_report(self, output_file: Optional[str] = None):
        """Save validation report to JSON file"""
        if output_file is None:
            output_file = self.data_dir / "validation_report.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_file}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Data Validation for RoyFin AI")
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing data files'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated list of tickers to validate'
    )
    
    parser.add_argument(
        '--min-days',
        type=int,
        default=200,
        help='Minimum number of days required'
    )
    
    parser.add_argument(
        '--max-missing-days',
        type=int,
        default=5,
        help='Maximum percentage of missing data allowed'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        help='Output file for validation report'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail on any validation warning'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Starting data validation process")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Minimum days required: {args.min_days}")
    
    # Parse tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    logger.info(f"Validating tickers: {tickers}")
    
    # Initialize validator
    validator = DataValidator(args.data_dir)
    
    # Run validation
    results = validator.validate_all_tickers(tickers, args.min_days, args.max_missing_days)
    
    # Save report
    validator.save_validation_report(args.output_report)
    
    # Print summary
    summary = results['summary']
    logger.info("="*50)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total tickers: {summary['total_tickers']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Success rate: {summary['success_rate']:.1f}%")
    logger.info(f"Overall status: {results['overall_status']}")
    
    if results['failed_validations']:
        logger.warning(f"Failed validations: {', '.join(results['failed_validations'])}")
    
    if results['warnings']:
        logger.warning(f"Warnings encountered: {len(results['warnings'])}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'FAILED':
        logger.error("Data validation failed")
        sys.exit(1)
    elif results['overall_status'] == 'PARTIAL' and args.strict:
        logger.error("Partial validation in strict mode")
        sys.exit(1)
    else:
        logger.info("Data validation completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()