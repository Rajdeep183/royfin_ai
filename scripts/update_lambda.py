#!/usr/bin/env python3
"""
Lambda Update Script for RoyFin AI
Updates AWS Lambda function with new model information
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

class LambdaUpdater:
    """Update Lambda function with new model information"""
    
    def __init__(self, region: str = 'us-east-1'):
        try:
            self.lambda_client = boto3.client('lambda', region_name=region)
            logger.info(f"Connected to Lambda service in region: {region}")
        except Exception as e:
            logger.error(f"Failed to connect to Lambda: {e}")
            raise
    
    def get_function_config(self, function_name: str) -> Dict:
        """Get current Lambda function configuration"""
        try:
            response = self.lambda_client.get_function_configuration(
                FunctionName=function_name
            )
            return response
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.error(f"Lambda function {function_name} not found")
            else:
                logger.error(f"Error getting function config: {e}")
            raise
    
    def update_environment_variables(self, function_name: str, tickers: str, 
                                   data_hash: str) -> bool:
        """Update Lambda function environment variables"""
        try:
            # Get current configuration
            current_config = self.get_function_config(function_name)
            current_env = current_config.get('Environment', {}).get('Variables', {})
            
            # Prepare new environment variables
            new_env = current_env.copy()
            new_env.update({
                'TRAINED_TICKERS': tickers,
                'DATA_HASH': data_hash,
                'LAST_TRAINING_DATE': datetime.now().isoformat(),
                'MODEL_VERSION': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            })
            
            # Update function configuration
            response = self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Environment={'Variables': new_env}
            )
            
            logger.info(f"✓ Updated Lambda function {function_name} environment variables")
            logger.info(f"  - Trained tickers: {tickers}")
            logger.info(f"  - Data hash: {data_hash[:8]}...")
            logger.info(f"  - Model version: {new_env['MODEL_VERSION']}")
            
            return True
            
        except ClientError as e:
            logger.error(f"✗ Error updating Lambda function: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    
    def create_function_alias(self, function_name: str, alias_name: str = "LATEST") -> bool:
        """Create or update function alias"""
        try:
            # Get the latest version number
            function_config = self.get_function_config(function_name)
            version = function_config['Version']
            
            # Try to update existing alias
            try:
                response = self.lambda_client.update_alias(
                    FunctionName=function_name,
                    Name=alias_name,
                    FunctionVersion=version,
                    Description=f"Updated on {datetime.now().isoformat()}"
                )
                logger.info(f"✓ Updated alias {alias_name} to version {version}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new alias
                    response = self.lambda_client.create_alias(
                        FunctionName=function_name,
                        Name=alias_name,
                        FunctionVersion=version,
                        Description=f"Created on {datetime.now().isoformat()}"
                    )
                    logger.info(f"✓ Created alias {alias_name} for version {version}")
                else:
                    raise
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error managing function alias: {e}")
            return False
    
    def add_function_tags(self, function_name: str, tickers: str, data_hash: str) -> bool:
        """Add tags to Lambda function"""
        try:
            # Get function ARN
            config = self.get_function_config(function_name)
            function_arn = config['FunctionArn']
            
            # Prepare tags
            tags = {
                'LastTrainingDate': datetime.now().strftime('%Y-%m-%d'),
                'TrainedTickers': tickers.replace(',', '_'),  # Replace commas for tag values
                'DataHash': data_hash[:16],  # Truncate for tag length limits
                'UpdatedBy': 'RoyFinAI-Workflow'
            }
            
            # Add tags
            self.lambda_client.tag_resource(
                Resource=function_arn,
                Tags=tags
            )
            
            logger.info(f"✓ Added tags to Lambda function {function_name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error adding tags to function: {e}")
            return False
    
    def validate_function_update(self, function_name: str, expected_tickers: str) -> bool:
        """Validate that the function was updated correctly"""
        try:
            config = self.get_function_config(function_name)
            env_vars = config.get('Environment', {}).get('Variables', {})
            
            # Check if expected tickers are in environment
            trained_tickers = env_vars.get('TRAINED_TICKERS', '')
            
            if expected_tickers in trained_tickers or trained_tickers in expected_tickers:
                logger.info(f"✓ Function update validation passed")
                return True
            else:
                logger.warning(f"✗ Validation failed: expected {expected_tickers}, got {trained_tickers}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error validating function update: {e}")
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Update Lambda function")
    
    parser.add_argument(
        '--function-name',
        type=str,
        required=True,
        help='Lambda function name'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated list of trained tickers'
    )
    
    parser.add_argument(
        '--data-hash',
        type=str,
        required=True,
        help='Data hash used for training'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region'
    )
    
    parser.add_argument(
        '--create-alias',
        action='store_true',
        help='Create/update function alias'
    )
    
    parser.add_argument(
        '--add-tags',
        action='store_true',
        help='Add tags to function'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Updating Lambda function")
    logger.info(f"Function: {args.function_name}")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Data hash: {args.data_hash[:8]}...")
    
    try:
        # Initialize Lambda updater
        updater = LambdaUpdater(args.region)
        
        success = True
        
        # Update environment variables
        if not updater.update_environment_variables(args.function_name, args.tickers, args.data_hash):
            success = False
        
        # Create/update alias if requested
        if args.create_alias:
            if not updater.create_function_alias(args.function_name):
                success = False
        
        # Add tags if requested
        if args.add_tags:
            if not updater.add_function_tags(args.function_name, args.tickers, args.data_hash):
                success = False
        
        # Validate update
        if success:
            if updater.validate_function_update(args.function_name, args.tickers):
                logger.info("✓ Lambda function update completed successfully")
                sys.exit(0)
            else:
                logger.error("✗ Lambda function update validation failed")
                sys.exit(1)
        else:
            logger.error("✗ Lambda function update failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()