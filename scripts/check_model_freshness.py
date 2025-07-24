#!/usr/bin/env python3
"""
Model Freshness Check Script for RoyFin AI
Checks if model needs retraining based on data changes
"""

import argparse
import logging
import os
import sys
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import boto3
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelFreshnessChecker:
    """Check if models need retraining based on data freshness"""
    
    def __init__(self, s3_bucket: Optional[str] = None):
        self.s3_bucket = s3_bucket
        self.s3_client = None
        
        if s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                logger.info(f"Connected to S3 bucket: {s3_bucket}")
            except Exception as e:
                logger.warning(f"Could not connect to S3: {e}")
    
    def get_local_model_info(self, ticker: str, models_dir: str = "./models") -> Optional[Dict]:
        """Get information about local model"""
        models_path = Path(models_dir)
        safe_ticker = ticker.replace("^", "").replace(".", "_")
        
        # Look for metadata file
        metadata_files = list(models_path.glob(f"{safe_ticker}_metadata.json"))
        if not metadata_files:
            logger.info(f"No local metadata found for {ticker}")
            return None
        
        metadata_file = metadata_files[0]
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check for model file
            model_files = list(models_path.glob(f"{safe_ticker}_model.pkl"))
            if not model_files:
                logger.warning(f"Metadata found but no model file for {ticker}")
                return None
            
            model_file = model_files[0]
            model_age_hours = (datetime.now() - datetime.fromtimestamp(model_file.stat().st_mtime)).total_seconds() / 3600
            
            return {
                'metadata': metadata,
                'model_file': str(model_file),
                'model_age_hours': model_age_hours,
                'data_hash': metadata.get('data_hash'),
                'created_at': metadata.get('created_at')
            }
            
        except Exception as e:
            logger.error(f"Error reading local model info for {ticker}: {e}")
            return None
    
    def get_s3_model_info(self, ticker: str) -> Optional[Dict]:
        """Get information about S3 model"""
        if not self.s3_client or not self.s3_bucket:
            return None
        
        safe_ticker = ticker.replace("^", "").replace(".", "_")
        
        try:
            # Check for latest.json file
            latest_key = f"models/{safe_ticker}/latest.json"
            
            try:
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=latest_key)
                latest_info = json.loads(response['Body'].read())
                
                return {
                    'latest_version': latest_info.get('latest_version'),
                    'timestamp': latest_info.get('timestamp'),
                    'data_hash': latest_info.get('data_hash'),
                    'files': latest_info.get('files', {})
                }
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.info(f"No S3 model found for {ticker}")
                    return None
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Error checking S3 model for {ticker}: {e}")
            return None
    
    def calculate_data_freshness_score(self, data_hash: str, current_data_hash: str) -> float:
        """Calculate a freshness score based on data hash differences"""
        if data_hash == current_data_hash:
            return 1.0  # Perfect match
        
        # Compare hashes to estimate data similarity
        # This is a simplified approach - in practice, you might want more sophisticated comparison
        if data_hash and current_data_hash:
            # Compare first 16 characters of hashes
            hash1_prefix = data_hash[:16]
            hash2_prefix = current_data_hash[:16]
            
            matching_chars = sum(1 for a, b in zip(hash1_prefix, hash2_prefix) if a == b)
            similarity = matching_chars / 16
            
            return similarity
        
        return 0.0  # No hash available
    
    def needs_retraining(self, ticker: str, current_data_hash: str, 
                        max_age_hours: int = 72, min_freshness_score: float = 0.8) -> Dict:
        """Determine if model needs retraining"""
        
        result = {
            'ticker': ticker,
            'needs_training': True,
            'reason': '',
            'local_model': None,
            's3_model': None,
            'recommendations': []
        }
        
        # Check local model
        local_info = self.get_local_model_info(ticker)
        result['local_model'] = local_info
        
        # Check S3 model
        s3_info = self.get_s3_model_info(ticker)
        result['s3_model'] = s3_info
        
        # Decision logic
        if not local_info and not s3_info:
            result['reason'] = 'No existing model found'
            result['recommendations'].append('Train new model')
            return result
        
        # Use the most recent model info
        model_info = None
        model_source = None
        
        if local_info and s3_info:
            # Compare timestamps
            local_time = datetime.fromisoformat(local_info['created_at'].replace('Z', '+00:00')) if local_info.get('created_at') else datetime.min
            s3_time = datetime.fromisoformat(s3_info['timestamp'].replace('Z', '+00:00')) if s3_info.get('timestamp') else datetime.min
            
            if local_time > s3_time:
                model_info = local_info
                model_source = 'local'
            else:
                model_info = s3_info
                model_source = 's3'
        elif local_info:
            model_info = local_info
            model_source = 'local'
        else:
            model_info = s3_info
            model_source = 's3'
        
        logger.info(f"Using {model_source} model info for {ticker}")
        
        # Check data hash match
        model_data_hash = model_info.get('data_hash')
        if model_data_hash == current_data_hash:
            result['needs_training'] = False
            result['reason'] = 'Model data hash matches current data'
            result['recommendations'].append('No retraining needed')
            return result
        
        # Check age
        if model_source == 'local' and 'model_age_hours' in model_info:
            age_hours = model_info['model_age_hours']
        else:
            # Calculate age from timestamp
            if 'timestamp' in model_info:
                model_time = datetime.fromisoformat(model_info['timestamp'].replace('Z', '+00:00'))
                age_hours = (datetime.now() - model_time.replace(tzinfo=None)).total_seconds() / 3600
            elif 'created_at' in model_info:
                model_time = datetime.fromisoformat(model_info['created_at'].replace('Z', '+00:00'))
                age_hours = (datetime.now() - model_time.replace(tzinfo=None)).total_seconds() / 3600
            else:
                age_hours = max_age_hours + 1  # Force retraining if no timestamp
        
        # Check if model is too old
        if age_hours > max_age_hours:
            result['reason'] = f'Model is too old ({age_hours:.1f} hours > {max_age_hours} hours)'
            result['recommendations'].append('Retrain due to age')
            return result
        
        # Check data freshness
        freshness_score = self.calculate_data_freshness_score(model_data_hash, current_data_hash)
        
        if freshness_score < min_freshness_score:
            result['reason'] = f'Data changed significantly (freshness score: {freshness_score:.3f} < {min_freshness_score})'
            result['recommendations'].append('Retrain due to data changes')
            return result
        
        # Model is fresh enough
        result['needs_training'] = False
        result['reason'] = f'Model is fresh (age: {age_hours:.1f}h, freshness: {freshness_score:.3f})'
        result['recommendations'].append('Model is sufficiently fresh')
        
        return result
    
    def batch_check_freshness(self, tickers: list, current_data_hash: str, 
                            max_age_hours: int = 72) -> Dict:
        """Check freshness for multiple tickers"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_tickers': len(tickers),
            'needs_training': [],
            'no_training_needed': [],
            'details': {}
        }
        
        for ticker in tickers:
            logger.info(f"Checking freshness for {ticker}")
            
            try:
                check_result = self.needs_retraining(ticker, current_data_hash, max_age_hours)
                results['details'][ticker] = check_result
                
                if check_result['needs_training']:
                    results['needs_training'].append(ticker)
                    logger.info(f"✓ {ticker} needs retraining: {check_result['reason']}")
                else:
                    results['no_training_needed'].append(ticker)
                    logger.info(f"✓ {ticker} is fresh: {check_result['reason']}")
                    
            except Exception as e:
                logger.error(f"Error checking {ticker}: {e}")
                # Default to needs training on error
                results['needs_training'].append(ticker)
                results['details'][ticker] = {
                    'ticker': ticker,
                    'needs_training': True,
                    'reason': f'Error during check: {str(e)}',
                    'error': True
                }
        
        results['training_needed_count'] = len(results['needs_training'])
        results['fresh_count'] = len(results['no_training_needed'])
        
        return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check model freshness")
    
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Ticker to check'
    )
    
    parser.add_argument(
        '--data-hash',
        type=str,
        required=True,
        help='Current data hash'
    )
    
    parser.add_argument(
        '--s3-bucket',
        type=str,
        help='S3 bucket name'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help='Local models directory'
    )
    
    parser.add_argument(
        '--max-age-hours',
        type=int,
        default=72,
        help='Maximum model age in hours'
    )
    
    parser.add_argument(
        '--min-freshness-score',
        type=float,
        default=0.8,
        help='Minimum data freshness score'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Checking model freshness")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Data hash: {args.data_hash[:8]}...")
    
    # Initialize checker
    checker = ModelFreshnessChecker(args.s3_bucket)
    
    # Check if training is needed
    result = checker.needs_retraining(
        ticker=args.ticker,
        current_data_hash=args.data_hash,
        max_age_hours=args.max_age_hours,
        min_freshness_score=args.min_freshness_score
    )
    
    # Print results
    logger.info("="*50)
    logger.info("MODEL FRESHNESS CHECK RESULTS")
    logger.info("="*50)
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Needs training: {result['needs_training']}")
    logger.info(f"Reason: {result['reason']}")
    
    if result['recommendations']:
        logger.info("Recommendations:")
        for rec in result['recommendations']:
            logger.info(f"  - {rec}")
    
    # Set GitHub Actions output
    print(f"::set-output name=needs-training::{str(result['needs_training']).lower()}")
    print(f"::set-output name=reason::{result['reason']}")
    
    # Exit with code indicating if training is needed
    sys.exit(0 if result['needs_training'] else 1)

if __name__ == "__main__":
    main()