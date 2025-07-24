#!/usr/bin/env python3
"""
Market Hours and Trading Days Check for RoyFin AI
Validates if it's appropriate to run data ingestion and training
"""

import argparse
import logging
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketHoursChecker:
    """Check market hours and trading days for different markets"""
    
    def __init__(self):
        # Define market timezones and trading hours
        self.markets = {
            'US': {
                'timezone': 'America/New_York',
                'trading_hours': {
                    'open': '09:30',
                    'close': '16:00'
                },
                'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday
                'holidays': self._get_us_holidays()
            },
            'INDIA': {
                'timezone': 'Asia/Kolkata',
                'trading_hours': {
                    'open': '09:15',
                    'close': '15:30'
                },
                'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday
                'holidays': self._get_indian_holidays()
            }
        }
    
    def _get_us_holidays(self) -> List[str]:
        """Get US market holidays for current year"""
        current_year = datetime.now().year
        # Major US market holidays (simplified list)
        holidays = [
            f"{current_year}-01-01",  # New Year's Day
            f"{current_year}-01-15",  # MLK Day (3rd Monday in January - approximate)
            f"{current_year}-02-19",  # Presidents Day (3rd Monday in February - approximate)
            f"{current_year}-05-27",  # Memorial Day (last Monday in May - approximate)
            f"{current_year}-07-04",  # Independence Day
            f"{current_year}-09-02",  # Labor Day (1st Monday in September - approximate)
            f"{current_year}-11-28",  # Thanksgiving (4th Thursday in November - approximate)
            f"{current_year}-12-25",  # Christmas Day
        ]
        return holidays
    
    def _get_indian_holidays(self) -> List[str]:
        """Get Indian market holidays for current year"""
        current_year = datetime.now().year
        # Major Indian market holidays (simplified list)
        holidays = [
            f"{current_year}-01-26",  # Republic Day
            f"{current_year}-08-15",  # Independence Day
            f"{current_year}-10-02",  # Gandhi Jayanti
        ]
        # Note: Religious holidays vary by year and would need a proper calendar
        return holidays
    
    def is_trading_day(self, date: datetime, market: str = 'US') -> bool:
        """Check if given date is a trading day"""
        market_info = self.markets.get(market, self.markets['US'])
        
        # Check if it's a weekend
        if date.weekday() not in market_info['trading_days']:
            return False
        
        # Check if it's a holiday
        date_str = date.strftime('%Y-%m-%d')
        if date_str in market_info['holidays']:
            return False
        
        return True
    
    def is_market_hours(self, dt: datetime, market: str = 'US') -> bool:
        """Check if given datetime is during market hours"""
        market_info = self.markets.get(market, self.markets['US'])
        
        # Convert to market timezone
        market_tz = pytz.timezone(market_info['timezone'])
        market_time = dt.astimezone(market_tz)
        
        # Check if it's a trading day
        if not self.is_trading_day(market_time, market):
            return False
        
        # Check trading hours
        open_time = datetime.strptime(market_info['trading_hours']['open'], '%H:%M').time()
        close_time = datetime.strptime(market_info['trading_hours']['close'], '%H:%M').time()
        
        current_time = market_time.time()
        return open_time <= current_time <= close_time
    
    def get_last_trading_day(self, market: str = 'US') -> datetime:
        """Get the last trading day"""
        today = datetime.now()
        
        # Go back up to 7 days to find the last trading day
        for i in range(7):
            check_date = today - timedelta(days=i)
            if self.is_trading_day(check_date, market):
                return check_date
        
        # If no trading day found in last 7 days, return today (fallback)
        return today
    
    def get_next_trading_day(self, market: str = 'US') -> datetime:
        """Get the next trading day"""
        today = datetime.now()
        
        # Look ahead up to 7 days to find the next trading day
        for i in range(1, 8):
            check_date = today + timedelta(days=i)
            if self.is_trading_day(check_date, market):
                return check_date
        
        # If no trading day found in next 7 days, return tomorrow (fallback)
        return today + timedelta(days=1)
    
    def should_run_data_ingestion(self) -> Tuple[bool, Dict]:
        """Determine if data ingestion should run now"""
        now = datetime.now()
        
        result = {
            'should_run': False,
            'reason': '',
            'market_status': {},
            'recommendations': []
        }
        
        # Check US market
        us_last_trading = self.get_last_trading_day('US')
        us_is_trading_day = self.is_trading_day(now, 'US')
        us_in_hours = self.is_market_hours(now, 'US')
        
        result['market_status']['US'] = {
            'is_trading_day': us_is_trading_day,
            'in_trading_hours': us_in_hours,
            'last_trading_day': us_last_trading.strftime('%Y-%m-%d'),
            'market_open': us_is_trading_day and not us_in_hours and now.hour >= 16
        }
        
        # Check Indian market
        india_last_trading = self.get_last_trading_day('INDIA')
        india_is_trading_day = self.is_trading_day(now, 'INDIA')
        india_in_hours = self.is_market_hours(now, 'INDIA')
        
        result['market_status']['INDIA'] = {
            'is_trading_day': india_is_trading_day,
            'in_trading_hours': india_in_hours,
            'last_trading_day': india_last_trading.strftime('%Y-%m-%d'),
            'market_open': india_is_trading_day and not india_in_hours and now.hour >= 16
        }
        
        # Decision logic
        current_hour_utc = now.hour
        
        # Scheduled time is 18:30 UTC (00:00 IST, ~13:30 EST depending on DST)
        # This is after both markets close
        if 18 <= current_hour_utc <= 23:  # Evening UTC hours
            if us_is_trading_day or india_is_trading_day:
                result['should_run'] = True
                result['reason'] = 'Scheduled nightly run after market close'
            else:
                # Weekend or holiday
                days_since_us_trading = (now.date() - us_last_trading.date()).days
                days_since_india_trading = (now.date() - india_last_trading.date()).days
                
                if days_since_us_trading <= 3 or days_since_india_trading <= 3:
                    result['should_run'] = True
                    result['reason'] = 'Weekend/holiday run with recent trading data'
                else:
                    result['should_run'] = False
                    result['reason'] = 'Too many days since last trading day'
        
        elif 0 <= current_hour_utc <= 6:  # Early morning UTC
            # This is late night in US, morning in India
            if india_is_trading_day:
                result['should_run'] = True
                result['reason'] = 'Early run before Indian market opens'
            else:
                result['should_run'] = False
                result['reason'] = 'No Indian trading day'
        
        else:
            # During market hours or inappropriate times
            result['should_run'] = False
            if us_in_hours or india_in_hours:
                result['reason'] = 'Markets are currently open'
            else:
                result['reason'] = 'Not scheduled time for data ingestion'
        
        # Add recommendations
        if not result['should_run']:
            if us_in_hours or india_in_hours:
                result['recommendations'].append('Wait for markets to close')
            else:
                next_us_trading = self.get_next_trading_day('US')
                result['recommendations'].append(f'Next recommended run: {next_us_trading.strftime("%Y-%m-%d")} after market close')
        
        return result['should_run'], result
    
    def check_data_freshness_requirements(self) -> Dict:
        """Check what data freshness is required"""
        now = datetime.now()
        
        us_last_trading = self.get_last_trading_day('US')
        india_last_trading = self.get_last_trading_day('INDIA')
        
        return {
            'required_data_dates': {
                'US_stocks': us_last_trading.strftime('%Y-%m-%d'),
                'Indian_stocks': india_last_trading.strftime('%Y-%m-%d'),
                'indices': us_last_trading.strftime('%Y-%m-%d'),
                'volatility': us_last_trading.strftime('%Y-%m-%d')
            },
            'max_acceptable_age_days': {
                'weekday': 1,
                'weekend': 3,
                'holiday_period': 5
            }
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Market Hours and Trading Days Check")
    parser.add_argument(
        '--market',
        choices=['US', 'INDIA', 'ALL'],
        default='ALL',
        help='Market to check (default: ALL)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file for market status JSON'
    )
    
    args = parser.parse_args()
    
    logger.info("Checking market hours and trading status")
    
    checker = MarketHoursChecker()
    
    # Check if data ingestion should run
    should_run, status = checker.should_run_data_ingestion()
    
    # Get data freshness requirements
    freshness_req = checker.check_data_freshness_requirements()
    
    # Combine results
    result = {
        'timestamp': datetime.now().isoformat(),
        'should_run_ingestion': should_run,
        'market_status': status,
        'data_freshness_requirements': freshness_req
    }
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Market status saved to {args.output_file}")
    
    # Print summary
    logger.info("="*50)
    logger.info("MARKET STATUS SUMMARY")
    logger.info("="*50)
    logger.info(f"Should run data ingestion: {should_run}")
    logger.info(f"Reason: {status['reason']}")
    
    for market, status_info in status['market_status'].items():
        logger.info(f"{market} Market:")
        logger.info(f"  - Trading day: {status_info['is_trading_day']}")
        logger.info(f"  - In trading hours: {status_info['in_trading_hours']}")
        logger.info(f"  - Last trading day: {status_info['last_trading_day']}")
    
    if status['recommendations']:
        logger.info("Recommendations:")
        for rec in status['recommendations']:
            logger.info(f"  - {rec}")
    
    # Set GitHub Actions output
    print(f"::set-output name=open::{should_run}")
    print(f"::set-output name=reason::{status['reason']}")
    
    # Exit with appropriate code
    sys.exit(0 if should_run else 1)

if __name__ == "__main__":
    try:
        # Install pytz if not available
        import pytz
    except ImportError:
        logger.warning("pytz not available, using simplified timezone handling")
        # Create a minimal pytz-like interface
        class SimpleTZ:
            @staticmethod
            def timezone(tz_name):
                return None
        pytz = SimpleTZ()
    
    main()