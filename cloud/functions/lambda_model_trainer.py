#!/usr/bin/env python3
"""
AWS Lambda Model Training Integration Script
Integrates with the existing RoyFin AI infrastructure for automated model retraining
"""

import os
import sys
import json
import logging
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the model directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

try:
    from stock_lstm import AdvancedStockPredictor
except ImportError:
    # Fallback for cloud environments
    print("Warning: Could not import AdvancedStockPredictor, using basic implementation")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LambdaModelTrainer:
    """
    AWS Lambda-compatible model trainer that integrates with RoyFin AI infrastructure
    """
    
    def __init__(self, event: Dict[str, Any], context: Any):
        self.event = event
        self.context = context
        self.s3_client = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        
        # Configuration from event
        self.workflow_run_id = event.get('workflow_run_id', 'unknown')
        self.timestamp = event.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.tickers = event.get('tickers', ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY'])
        self.force_retrain = event.get('force_retrain', False)
        self.config = event.get('configuration', {})
        
        # AWS configuration
        self.s3_bucket = os.environ.get('S3_BUCKET_NAME', 'royfin-ai-models')
        self.s3_data_prefix = 'data/financial'
        self.s3_models_prefix = 'models'
        
        # Training configuration
        self.ensemble_size = self.config.get('ensemble_size', 5)
        self.epochs = self.config.get('epochs', 100)
        self.optimize_hyperparams = self.config.get('optimize_hyperparams', True)
        self.validation_split = self.config.get('validation_split', 0.2)
        
        logger.info(f"🚀 Lambda Model Trainer initialized for workflow: {self.workflow_run_id}")

    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Main Lambda handler function
        """
        try:
            logger.info(f"📥 Received training request: {json.dumps(event, indent=2)}")
            
            # Initialize trainer
            trainer = LambdaModelTrainer(event, context)
            
            # Execute training pipeline
            results = trainer.execute_training_pipeline()
            
            # Return success response
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'success',
                    'message': 'Model training completed successfully',
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
            
            # Return error response
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            }

    def execute_training_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline
        """
        results = {
            'workflow_run_id': self.workflow_run_id,
            'timestamp': self.timestamp,
            'trained_models': [],
            'performance_metrics': {},
            'artifacts': []
        }
        
        logger.info(f"🎯 Starting training pipeline for {len(self.tickers)} tickers")
        
        for ticker in self.tickers:
            try:
                logger.info(f"🔄 Training model for {ticker}")
                
                # Check if retraining is needed
                if not self.force_retrain and self._is_recent_model_available(ticker):
                    logger.info(f"⏭️ Skipping {ticker} - recent model exists")
                    continue
                
                # Download data from S3 or fetch fresh data
                data_available = self._prepare_training_data(ticker)
                
                if not data_available:
                    logger.warning(f"⚠️ No data available for {ticker}, skipping")
                    continue
                
                # Train the model
                model_results = self._train_ticker_model(ticker)
                
                if model_results['success']:
                    results['trained_models'].append(ticker)
                    results['performance_metrics'][ticker] = model_results['metrics']
                    results['artifacts'].extend(model_results['artifacts'])
                    
                    logger.info(f"✅ Successfully trained model for {ticker}")
                else:
                    logger.error(f"❌ Failed to train model for {ticker}: {model_results['error']}")
                
            except Exception as e:
                logger.error(f"❌ Error training {ticker}: {str(e)}")
                continue
        
        # Upload final results
        self._upload_training_results(results)
        
        logger.info(f"🎉 Training pipeline completed. Trained models: {results['trained_models']}")
        return results

    def _is_recent_model_available(self, ticker: str, max_age_hours: int = 24) -> bool:
        """
        Check if a recent model exists for the ticker
        """
        try:
            # List recent models in S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f"{self.s3_models_prefix}/{ticker}/",
                MaxKeys=10
            )
            
            if 'Contents' not in response:
                return False
            
            # Check if any model is recent enough
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) > cutoff_time:
                    logger.info(f"🕐 Recent model found for {ticker}: {obj['Key']}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking for recent models for {ticker}: {e}")
            return False

    def _prepare_training_data(self, ticker: str) -> bool:
        """
        Prepare training data for the ticker
        """
        try:
            # Try to download processed data from S3 first
            s3_key = f"{self.s3_data_prefix}/processed/{ticker}_latest.csv"
            local_path = f"/tmp/{ticker}_data.csv"
            
            try:
                self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                logger.info(f"📥 Downloaded data for {ticker} from S3")
                return True
            except Exception as e:
                logger.info(f"📊 S3 data not available for {ticker}, will fetch fresh data: {e}")
            
            # If S3 data not available, fetch fresh data
            # This is a fallback - in production, data should come from the ingestion job
            import yfinance as yf
            import pandas as pd
            
            logger.info(f"🔄 Fetching fresh data for {ticker}")
            
            # Fetch 5 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)
            
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d"
            )
            
            if df.empty:
                logger.error(f"❌ No data available for {ticker}")
                return False
            
            # Save locally for training
            df.to_csv(local_path)
            logger.info(f"💾 Fresh data saved for {ticker} ({len(df)} records)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data for {ticker}: {e}")
            return False

    def _train_ticker_model(self, ticker: str) -> Dict[str, Any]:
        """
        Train the model for a specific ticker
        """
        try:
            logger.info(f"🤖 Starting model training for {ticker}")
            
            # Initialize the predictor with enhanced configuration
            predictor = AdvancedStockPredictor(
                ticker=ticker,
                seq_length=60,
                ensemble_size=self.ensemble_size,
                hidden_dim=256,
                num_layers=3,
                learning_rate=0.001,
                batch_size=64,
                epochs=self.epochs,
                dropout=0.3,
                use_attention=True,
                use_residual=True,
                optimize_hyperparams=self.optimize_hyperparams
            )
            
            # Train the ensemble
            predictor.train_ensemble()
            
            # Generate performance metrics
            metrics = self._evaluate_model_performance(predictor)
            
            # Save model to S3
            artifacts = self._save_model_to_s3(predictor, ticker)
            
            return {
                'success': True,
                'metrics': metrics,
                'artifacts': artifacts
            }
            
        except Exception as e:
            logger.error(f"❌ Model training failed for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {},
                'artifacts': []
            }

    def _evaluate_model_performance(self, predictor) -> Dict[str, Any]:
        """
        Evaluate model performance and return metrics
        """
        try:
            # Get training history
            training_history = predictor.training_history
            
            if not training_history:
                return {'error': 'No training history available'}
            
            # Calculate aggregate metrics
            best_val_losses = [h['best_val_loss'] for h in training_history]
            avg_best_val_loss = sum(best_val_losses) / len(best_val_losses)
            min_val_loss = min(best_val_losses)
            
            # Generate a test prediction to validate model
            try:
                test_predictions = predictor.predict_advanced(days_ahead=5, use_monte_carlo=False)
                prediction_success = len(test_predictions) > 0
            except Exception as e:
                logger.warning(f"⚠️ Test prediction failed: {e}")
                prediction_success = False
            
            metrics = {
                'ensemble_size': len(predictor.models),
                'best_models_count': len(predictor.best_models),
                'avg_validation_loss': float(avg_best_val_loss),
                'min_validation_loss': float(min_val_loss),
                'prediction_test_passed': prediction_success,
                'model_size_mb': self._estimate_model_size(predictor),
                'training_completed_at': datetime.now().isoformat()
            }
            
            logger.info(f"📊 Performance metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model performance: {e}")
            return {'error': str(e)}

    def _estimate_model_size(self, predictor) -> float:
        """
        Estimate model size in MB
        """
        try:
            import pickle
            import io
            
            # Serialize model to estimate size
            buffer = io.BytesIO()
            save_dict = {
                'models': predictor.models,
                'best_models': predictor.best_models,
                'feature_scalers': predictor.feature_scalers,
                'target_scalers': predictor.target_scalers,
                'input_dim': predictor.input_dim,
                'feature_names': predictor.feature_names
            }
            pickle.dump(save_dict, buffer)
            size_bytes = buffer.tell()
            size_mb = size_bytes / (1024 * 1024)
            
            return round(size_mb, 2)
            
        except Exception as e:
            logger.warning(f"⚠️ Could not estimate model size: {e}")
            return 0.0

    def _save_model_to_s3(self, predictor, ticker: str) -> List[str]:
        """
        Save the trained model to S3
        """
        artifacts = []
        
        try:
            # Save the model locally first
            local_model_path = f"/tmp/{ticker}_model_{self.timestamp}.pkl"
            predictor.model_save_path = local_model_path
            predictor.save_advanced_model()
            
            # Upload to S3
            s3_key = f"{self.s3_models_prefix}/{ticker}/{ticker}_model_{self.timestamp}.pkl"
            self.s3_client.upload_file(local_model_path, self.s3_bucket, s3_key)
            artifacts.append(s3_key)
            
            # Also save as latest
            latest_s3_key = f"{self.s3_models_prefix}/{ticker}/{ticker}_model_latest.pkl"
            self.s3_client.copy_object(
                Bucket=self.s3_bucket,
                CopySource={'Bucket': self.s3_bucket, 'Key': s3_key},
                Key=latest_s3_key
            )
            artifacts.append(latest_s3_key)
            
            logger.info(f"💾 Model saved to S3: {s3_key}")
            
            # Clean up local file
            try:
                os.remove(local_model_path)
            except:
                pass
            
            return artifacts
            
        except Exception as e:
            logger.error(f"❌ Error saving model to S3: {e}")
            return []

    def _upload_training_results(self, results: Dict[str, Any]) -> None:
        """
        Upload training results summary to S3
        """
        try:
            results_json = json.dumps(results, indent=2, default=str)
            
            # Save to S3
            s3_key = f"training_results/workflow_{self.workflow_run_id}_{self.timestamp}.json"
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=results_json,
                ContentType='application/json'
            )
            
            logger.info(f"📄 Training results uploaded to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"❌ Error uploading training results: {e}")


# Lambda handler function (entry point)
def lambda_handler(event, context):
    """
    AWS Lambda entry point
    """
    trainer = LambdaModelTrainer(event, context)
    return trainer.lambda_handler(event, context)


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "source": "github_actions",
        "workflow_run_id": "test_run_123",
        "timestamp": "20240724_120000",
        "trigger_type": "manual",
        "tickers": ["AAPL", "GOOGL"],
        "force_retrain": True,
        "market_status": "closed",
        "data_ingestion": {
            "completed": True,
            "artifacts_available": True
        },
        "configuration": {
            "ensemble_size": 3,
            "epochs": 50,
            "optimize_hyperparams": False,
            "validation_split": 0.2
        }
    }
    
    # Mock context
    class MockContext:
        def __init__(self):
            self.function_name = "test_function"
            self.memory_limit_in_mb = 1024
            self.remaining_time_in_millis = lambda: 300000
    
    # Run test
    result = lambda_handler(test_event, MockContext())
    print("Test result:", json.dumps(result, indent=2))