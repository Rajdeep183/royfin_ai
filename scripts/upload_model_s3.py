#!/usr/bin/env python3
"""
S3 Model Upload Script for RoyFin AI
Uploads trained models to AWS S3 with proper versioning and metadata
"""

import argparse
import logging
import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S3ModelUploader:
    """Upload models to S3 with versioning and metadata"""
    
    def __init__(self, bucket_name: str, aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region
                )
            else:
                # Use default credentials (environment variables, IAM role, etc.)
                self.s3_client = boto3.client('s3', region_name=region)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"Bucket {bucket_name} not found")
            elif error_code == '403':
                logger.error(f"Access denied to bucket {bucket_name}")
            else:
                logger.error(f"Error accessing bucket {bucket_name}: {e}")
            raise
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_model_files(self, ticker: str, models_dir: str = "./models") -> Dict[str, Path]:
        """Find all model-related files for a ticker"""
        models_path = Path(models_dir)
        safe_ticker = ticker.replace("^", "").replace(".", "_")
        
        files = {}
        
        # Look for various model files
        file_patterns = {
            'model': [f"{safe_ticker}_model.pkl", f"{safe_ticker}_*.pkl"],
            'scaler': [f"{safe_ticker}_scaler.pkl", f"{safe_ticker}_scalers.pkl"], 
            'metadata': [f"{safe_ticker}_metadata.json", f"{safe_ticker}_config.json"],
            'performance': [f"{safe_ticker}_performance.json", f"{safe_ticker}_metrics.json"],
            'training_log': [f"{safe_ticker}_training.log", f"{safe_ticker}.log"]
        }
        
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                found_files = list(models_path.glob(pattern))
                if found_files:
                    # Take the most recent file if multiple matches
                    files[file_type] = max(found_files, key=lambda x: x.stat().st_mtime)
                    break
        
        return files
    
    def create_model_metadata(self, ticker: str, data_hash: str, model_files: Dict[str, Path]) -> Dict:
        """Create comprehensive metadata for the model upload"""
        metadata = {
            'ticker': ticker,
            'upload_timestamp': datetime.now().isoformat(),
            'data_hash': data_hash,
            'model_version': f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'files': {},
            'checksums': {},
            'total_size_bytes': 0
        }
        
        for file_type, file_path in model_files.items():
            if file_path and file_path.exists():
                file_hash = self.calculate_file_hash(file_path)
                file_size = file_path.stat().st_size
                
                metadata['files'][file_type] = {
                    'filename': file_path.name,
                    'size_bytes': file_size,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                metadata['checksums'][file_type] = file_hash
                metadata['total_size_bytes'] += file_size
        
        return metadata
    
    def upload_file_to_s3(self, local_file: Path, s3_key: str, metadata: Optional[Dict] = None) -> bool:
        """Upload a single file to S3"""
        try:
            extra_args = {}
            
            if metadata:
                # Add metadata as S3 object metadata
                extra_args['Metadata'] = {k: str(v) for k, v in metadata.items() if k != 'files'}
            
            # Set content type based on file extension
            if local_file.suffix == '.json':
                extra_args['ContentType'] = 'application/json'
            elif local_file.suffix == '.pkl':
                extra_args['ContentType'] = 'application/octet-stream'
            elif local_file.suffix == '.log':
                extra_args['ContentType'] = 'text/plain'
            
            self.s3_client.upload_file(
                str(local_file),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            logger.info(f"✓ Uploaded {local_file.name} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"✗ Failed to upload {local_file.name}: {e}")
            return False
    
    def check_existing_model(self, ticker: str, data_hash: str) -> Optional[Dict]:
        """Check if a model with the same data hash already exists"""
        try:
            # List objects with the ticker prefix
            safe_ticker = ticker.replace("^", "").replace(".", "_")
            prefix = f"models/{safe_ticker}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return None
            
            # Look for metadata files
            for obj in response['Contents']:
                if obj['Key'].endswith('_metadata.json'):
                    try:
                        # Download and check metadata
                        metadata_obj = self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        metadata = json.loads(metadata_obj['Body'].read())
                        
                        if metadata.get('data_hash') == data_hash:
                            logger.info(f"Found existing model with same data hash: {obj['Key']}")
                            return metadata
                            
                    except Exception as e:
                        logger.warning(f"Error reading metadata from {obj['Key']}: {e}")
                        continue
            
            return None
            
        except ClientError as e:
            logger.warning(f"Error checking existing models: {e}")
            return None
    
    def upload_model(self, ticker: str, data_hash: str, models_dir: str = "./models", 
                    force_upload: bool = False) -> Dict:
        """Upload model and all related files for a ticker"""
        logger.info(f"Starting upload for {ticker}")
        
        result = {
            'ticker': ticker,
            'success': False,
            'uploaded_files': [],
            'skipped_files': [],
            'errors': [],
            's3_keys': {},
            'total_size_bytes': 0
        }
        
        # Check if model already exists with same data hash
        if not force_upload:
            existing_metadata = self.check_existing_model(ticker, data_hash)
            if existing_metadata:
                logger.info(f"Model for {ticker} with data hash {data_hash[:8]} already exists, skipping upload")
                result['success'] = True
                result['skipped_reason'] = 'Model already exists with same data hash'
                return result
        
        # Find model files
        model_files = self.get_model_files(ticker, models_dir)
        
        if not model_files:
            error_msg = f"No model files found for {ticker}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
            return result
        
        # Create metadata
        upload_metadata = self.create_model_metadata(ticker, data_hash, model_files)
        
        # Prepare S3 keys
        safe_ticker = ticker.replace("^", "").replace(".", "_")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_key = f"models/{safe_ticker}/{timestamp}"
        
        # Upload each file
        upload_success = True
        
        for file_type, file_path in model_files.items():
            if not file_path or not file_path.exists():
                continue
            
            s3_key = f"{base_key}/{file_path.name}"
            
            if self.upload_file_to_s3(file_path, s3_key, upload_metadata):
                result['uploaded_files'].append(file_path.name)
                result['s3_keys'][file_type] = s3_key
                result['total_size_bytes'] += file_path.stat().st_size
            else:
                upload_success = False
                result['errors'].append(f"Failed to upload {file_path.name}")
        
        # Upload metadata file
        metadata_key = f"{base_key}/{safe_ticker}_metadata.json"
        metadata_file = Path(models_dir) / f"{safe_ticker}_upload_metadata.json"
        
        try:
            # Save metadata to temporary file
            with open(metadata_file, 'w') as f:
                json.dump(upload_metadata, f, indent=2)
            
            if self.upload_file_to_s3(metadata_file, metadata_key):
                result['uploaded_files'].append(metadata_file.name)
                result['s3_keys']['upload_metadata'] = metadata_key
            
            # Clean up temporary file
            metadata_file.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error creating/uploading metadata: {e}")
            result['errors'].append(f"Metadata upload failed: {str(e)}")
            upload_success = False
        
        # Create a "latest" symlink by uploading a reference file
        if upload_success:
            try:
                latest_key = f"models/{safe_ticker}/latest.json"
                latest_data = {
                    'latest_version': upload_metadata['model_version'],
                    'timestamp': upload_metadata['upload_timestamp'],
                    'data_hash': data_hash,
                    'files': result['s3_keys']
                }
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=latest_key,
                    Body=json.dumps(latest_data, indent=2),
                    ContentType='application/json'
                )
                
                result['s3_keys']['latest'] = latest_key
                logger.info(f"✓ Updated latest reference for {ticker}")
                
            except Exception as e:
                logger.warning(f"Failed to update latest reference: {e}")
        
        result['success'] = upload_success
        
        if upload_success:
            logger.info(f"✓ Successfully uploaded model for {ticker}")
            logger.info(f"  Files uploaded: {len(result['uploaded_files'])}")
            logger.info(f"  Total size: {result['total_size_bytes'] / 1024 / 1024:.2f} MB")
        else:
            logger.error(f"✗ Upload failed for {ticker}")
            for error in result['errors']:
                logger.error(f"  Error: {error}")
        
        return result

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Upload models to S3")
    
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Ticker symbol'
    )
    
    parser.add_argument(
        '--bucket',
        type=str,
        required=True,
        help='S3 bucket name'
    )
    
    parser.add_argument(
        '--data-hash',
        type=str,
        required=True,
        help='Data hash for versioning'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help='Directory containing model files'
    )
    
    parser.add_argument(
        '--aws-access-key-id',
        type=str,
        help='AWS access key ID (if not using environment variables)'
    )
    
    parser.add_argument(
        '--aws-secret-access-key',
        type=str,
        help='AWS secret access key (if not using environment variables)'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region'
    )
    
    parser.add_argument(
        '--force-upload',
        action='store_true',
        help='Force upload even if model exists'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Starting S3 model upload")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Data hash: {args.data_hash[:8]}...")
    
    try:
        # Initialize uploader
        uploader = S3ModelUploader(
            bucket_name=args.bucket,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            region=args.region
        )
        
        # Upload model
        result = uploader.upload_model(
            ticker=args.ticker,
            data_hash=args.data_hash,
            models_dir=args.models_dir,
            force_upload=args.force_upload
        )
        
        # Print summary
        logger.info("="*50)
        logger.info("UPLOAD SUMMARY")
        logger.info("="*50)
        logger.info(f"Ticker: {args.ticker}")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Files uploaded: {len(result['uploaded_files'])}")
        
        if result['uploaded_files']:
            logger.info("Uploaded files:")
            for filename in result['uploaded_files']:
                logger.info(f"  - {filename}")
        
        if result['errors']:
            logger.error("Errors encountered:")
            for error in result['errors']:
                logger.error(f"  - {error}")
        
        # Exit with appropriate code
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()