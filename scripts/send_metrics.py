#!/usr/bin/env python3
"""
CloudWatch Metrics Script for RoyFin AI
Sends workflow metrics to AWS CloudWatch
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from typing import Dict, List
import boto3
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CloudWatchMetricsSender:
    """Send workflow metrics to CloudWatch"""
    
    def __init__(self, region: str = 'us-east-1'):
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name=region)
            logger.info(f"Connected to CloudWatch in region: {region}")
        except Exception as e:
            logger.error(f"Failed to connect to CloudWatch: {e}")
            raise
    
    def send_workflow_metrics(self, workflow_run_id: str, tickers: str, 
                            training_results: str) -> bool:
        """Send comprehensive workflow metrics"""
        try:
            ticker_list = [t.strip() for t in tickers.split(',')]
            timestamp = datetime.now()
            
            metrics = []
            
            # Basic workflow metrics
            metrics.append({
                'MetricName': 'WorkflowRuns',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': timestamp,
                'Dimensions': [
                    {'Name': 'WorkflowType', 'Value': 'ModelTraining'},
                    {'Name': 'Status', 'Value': training_results}
                ]
            })
            
            # Tickers processed
            metrics.append({
                'MetricName': 'TickersProcessed',
                'Value': len(ticker_list),
                'Unit': 'Count',
                'Timestamp': timestamp,
                'Dimensions': [
                    {'Name': 'WorkflowRunId', 'Value': workflow_run_id}
                ]
            })
            
            # Success/failure metrics
            if training_results == 'success':
                metrics.append({
                    'MetricName': 'SuccessfulRuns',
                    'Value': 1,
                    'Unit': 'Count',
                    'Timestamp': timestamp
                })
            else:
                metrics.append({
                    'MetricName': 'FailedRuns',
                    'Value': 1,
                    'Unit': 'Count',
                    'Timestamp': timestamp
                })
            
            # Per-ticker metrics
            for ticker in ticker_list:
                metrics.append({
                    'MetricName': 'TickerTrainingAttempt',
                    'Value': 1,
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [
                        {'Name': 'Ticker', 'Value': ticker},
                        {'Name': 'Status', 'Value': training_results}
                    ]
                })
            
            # Send metrics in batches (CloudWatch limit is 20 per request)
            batch_size = 20
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i:i + batch_size]
                
                self.cloudwatch.put_metric_data(
                    Namespace='RoyFinAI/ModelTraining',
                    MetricData=batch
                )
                
                logger.info(f"Sent batch of {len(batch)} metrics to CloudWatch")
            
            logger.info(f"✓ Successfully sent {len(metrics)} metrics to CloudWatch")
            return True
            
        except ClientError as e:
            logger.error(f"✗ CloudWatch API error: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error sending metrics: {e}")
            return False
    
    def send_performance_metrics(self, ticker: str, performance_data: Dict) -> bool:
        """Send model performance metrics"""
        try:
            timestamp = datetime.now()
            metrics = []
            
            # Extract performance metrics
            if 'accuracy' in performance_data:
                metrics.append({
                    'MetricName': 'ModelAccuracy',
                    'Value': float(performance_data['accuracy']),
                    'Unit': 'Percent',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'Ticker', 'Value': ticker}]
                })
            
            if 'mse' in performance_data:
                metrics.append({
                    'MetricName': 'ModelMSE',
                    'Value': float(performance_data['mse']),
                    'Unit': 'None',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'Ticker', 'Value': ticker}]
                })
            
            if 'mae' in performance_data:
                metrics.append({
                    'MetricName': 'ModelMAE',
                    'Value': float(performance_data['mae']),
                    'Unit': 'None',
                    'Timestamp': timestamp,
                    'Dimensions': [{'Name': 'Ticker', 'Value': ticker}]
                })
            
            # Send metrics
            if metrics:
                self.cloudwatch.put_metric_data(
                    Namespace='RoyFinAI/ModelPerformance',
                    MetricData=metrics
                )
                
                logger.info(f"✓ Sent {len(metrics)} performance metrics for {ticker}")
                return True
            else:
                logger.warning(f"No performance metrics found for {ticker}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error sending performance metrics for {ticker}: {e}")
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Send CloudWatch metrics")
    
    parser.add_argument(
        '--workflow-run-id',
        type=str,
        required=True,
        help='GitHub Actions workflow run ID'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated list of tickers'
    )
    
    parser.add_argument(
        '--training-results',
        type=str,
        choices=['success', 'failure', 'skipped'],
        required=True,
        help='Training results status'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Sending CloudWatch metrics")
    logger.info(f"Workflow Run ID: {args.workflow_run_id}")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Training Results: {args.training_results}")
    
    try:
        # Initialize CloudWatch sender
        sender = CloudWatchMetricsSender(args.region)
        
        # Send metrics
        success = sender.send_workflow_metrics(
            workflow_run_id=args.workflow_run_id,
            tickers=args.tickers,
            training_results=args.training_results
        )
        
        if success:
            logger.info("✓ CloudWatch metrics sent successfully")
            sys.exit(0)
        else:
            logger.error("✗ Failed to send CloudWatch metrics")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()