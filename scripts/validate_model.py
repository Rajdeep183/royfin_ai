#!/usr/bin/env python3
"""
Model Validation Script for RoyFin AI
Validates trained models and their performance
"""

import argparse
import logging
import os
import sys
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Validate trained models and their performance"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'models_validated': [],
            'passed_validations': [],
            'failed_validations': [],
            'performance_metrics': {},
            'summary': {}
        }
    
    def find_model_files(self, ticker: str) -> Dict[str, Optional[Path]]:
        """Find model files for a given ticker"""
        safe_ticker = ticker.replace("^", "").replace(".", "_")
        
        files = {
            'model': None,
            'scaler': None,
            'metadata': None,
            'performance': None
        }
        
        # Look for model files with various patterns
        patterns = {
            'model': [f"{safe_ticker}_model.pkl", f"{safe_ticker}_*.pkl"],
            'scaler': [f"{safe_ticker}_scaler.pkl", f"{safe_ticker}_scalers.pkl"],
            'metadata': [f"{safe_ticker}_metadata.json", f"{safe_ticker}_config.json"],
            'performance': [f"{safe_ticker}_performance.json", f"{safe_ticker}_metrics.json"]
        }
        
        for file_type, file_patterns in patterns.items():
            for pattern in file_patterns:
                found_files = list(self.models_dir.glob(pattern))
                if found_files:
                    files[file_type] = found_files[0]  # Take the first match
                    break
        
        return files
    
    def validate_model_files(self, ticker: str) -> Tuple[bool, List[str]]:
        """Validate that required model files exist and are loadable"""
        issues = []
        
        model_files = self.find_model_files(ticker)
        
        # Check if model file exists
        if not model_files['model'] or not model_files['model'].exists():
            issues.append(f"Model file not found for {ticker}")
            return False, issues
        
        # Try to load the model file
        try:
            with open(model_files['model'], 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if it's a valid model structure
            if not isinstance(model_data, (dict, list)) and not hasattr(model_data, 'predict'):
                issues.append(f"Invalid model format for {ticker}")
        except Exception as e:
            issues.append(f"Cannot load model file for {ticker}: {str(e)}")
        
        # Check scaler file if it exists
        if model_files['scaler'] and model_files['scaler'].exists():
            try:
                with open(model_files['scaler'], 'rb') as f:
                    pickle.load(f)
            except Exception as e:
                issues.append(f"Cannot load scaler file for {ticker}: {str(e)}")
        
        return len(issues) == 0, issues
    
    def validate_model_metadata(self, ticker: str) -> Tuple[bool, List[str]]:
        """Validate model metadata and configuration"""
        issues = []
        
        model_files = self.find_model_files(ticker)
        
        if not model_files['metadata'] or not model_files['metadata'].exists():
            issues.append(f"No metadata file found for {ticker}")
            return False, issues
        
        try:
            with open(model_files['metadata'], 'r') as f:
                metadata = json.load(f)
            
            # Check required metadata fields
            required_fields = ['ticker', 'created_at', 'model_type']
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                issues.append(f"Missing metadata fields for {ticker}: {missing_fields}")
            
            # Check if model is recent enough (within last 30 days)
            if 'created_at' in metadata:
                try:
                    created_date = datetime.fromisoformat(metadata['created_at'].replace('Z', '+00:00'))
                    days_old = (datetime.now() - created_date.replace(tzinfo=None)).days
                    if days_old > 30:
                        issues.append(f"Model for {ticker} is {days_old} days old")
                except Exception:
                    issues.append(f"Invalid created_at format in metadata for {ticker}")
            
        except Exception as e:
            issues.append(f"Cannot read metadata file for {ticker}: {str(e)}")
        
        return len(issues) == 0, issues
    
    def validate_model_performance(self, ticker: str, min_accuracy: float = 0.6) -> Tuple[bool, List[str]]:
        """Validate model performance metrics"""
        issues = []
        
        model_files = self.find_model_files(ticker)
        
        if not model_files['performance'] or not model_files['performance'].exists():
            issues.append(f"No performance file found for {ticker}")
            return False, issues
        
        try:
            with open(model_files['performance'], 'r') as f:
                performance = json.load(f)
            
            # Check for required performance metrics
            required_metrics = ['accuracy', 'mse', 'mae']
            available_metrics = []
            
            # Look for metrics in various possible structures
            if isinstance(performance, dict):
                if 'test_metrics' in performance:
                    metrics = performance['test_metrics']
                elif 'validation_metrics' in performance:
                    metrics = performance['validation_metrics']
                else:
                    metrics = performance
                
                # Extract available metrics
                for metric in required_metrics:
                    if metric in metrics:
                        available_metrics.append(metric)
                    elif f'test_{metric}' in metrics:
                        available_metrics.append(metric)
                        metrics[metric] = metrics[f'test_{metric}']
                    elif f'val_{metric}' in metrics:
                        available_metrics.append(metric)
                        metrics[metric] = metrics[f'val_{metric}']
                
                # Validate accuracy if available
                if 'accuracy' in metrics:
                    accuracy = float(metrics['accuracy'])
                    if accuracy < min_accuracy:
                        issues.append(f"Model accuracy {accuracy:.3f} < minimum {min_accuracy} for {ticker}")
                    
                    self.validation_results['performance_metrics'][ticker] = {
                        'accuracy': accuracy,
                        'mse': metrics.get('mse'),
                        'mae': metrics.get('mae')
                    }
                else:
                    # Try to infer accuracy from other metrics
                    if 'directional_accuracy' in metrics:
                        accuracy = float(metrics['directional_accuracy'])
                        if accuracy < min_accuracy:
                            issues.append(f"Directional accuracy {accuracy:.3f} < minimum {min_accuracy} for {ticker}")
                    else:
                        issues.append(f"No accuracy metric found for {ticker}")
                
                # Check for reasonable MSE and MAE values
                if 'mse' in metrics:
                    mse = float(metrics['mse'])
                    if mse > 1.0:  # Very high MSE might indicate poor performance
                        issues.append(f"High MSE {mse:.3f} for {ticker}")
                
                if 'mae' in metrics:
                    mae = float(metrics['mae'])
                    if mae > 0.5:  # Very high MAE might indicate poor performance
                        issues.append(f"High MAE {mae:.3f} for {ticker}")
            
            else:
                issues.append(f"Invalid performance data format for {ticker}")
        
        except Exception as e:
            issues.append(f"Cannot read performance file for {ticker}: {str(e)}")
        
        return len(issues) == 0, issues
    
    def test_model_predictions(self, ticker: str) -> Tuple[bool, List[str]]:
        """Test that model can make predictions"""
        issues = []
        
        try:
            # Import the model module
            import sys
            model_path = Path(__file__).parent.parent / "model"
            if str(model_path) not in sys.path:
                sys.path.append(str(model_path))
            
            from stock_lstm import StockPredictor
            
            # Initialize predictor
            predictor = StockPredictor(ticker=ticker)
            
            # Try to load the model
            if not predictor.load_model():
                issues.append(f"Failed to load model for {ticker}")
                return False, issues
            
            # Try to make a simple prediction
            try:
                predictions = predictor.predict(days_ahead=5, temperature=1.0)
                
                if predictions is None or len(predictions) == 0:
                    issues.append(f"Model returned empty predictions for {ticker}")
                elif len(predictions) != 5:
                    issues.append(f"Model returned {len(predictions)} predictions instead of 5 for {ticker}")
                else:
                    # Check if predictions are reasonable
                    pred_values = predictions['predicted_price'].values if 'predicted_price' in predictions.columns else predictions.iloc[:, 0].values
                    
                    if np.any(np.isnan(pred_values)):
                        issues.append(f"Model returned NaN predictions for {ticker}")
                    elif np.any(pred_values <= 0):
                        issues.append(f"Model returned non-positive predictions for {ticker}")
                    else:
                        logger.info(f"✓ Model prediction test passed for {ticker}")
                
            except Exception as e:
                issues.append(f"Error during prediction test for {ticker}: {str(e)}")
        
        except ImportError as e:
            issues.append(f"Cannot import model module: {str(e)}")
        except Exception as e:
            issues.append(f"Unexpected error during prediction test for {ticker}: {str(e)}")
        
        return len(issues) == 0, issues
    
    def validate_single_model(self, ticker: str, min_accuracy: float = 0.6, 
                            test_predictions: bool = False) -> Dict:
        """Validate a single model"""
        logger.info(f"Validating model for {ticker}")
        
        result = {
            'ticker': ticker,
            'status': 'UNKNOWN',
            'validations': {},
            'issues': [],
            'metadata': {}
        }
        
        # Run validations
        validations = [
            ('file_integrity', self.validate_model_files, (ticker,)),
            ('metadata', self.validate_model_metadata, (ticker,)),
            ('performance', self.validate_model_performance, (ticker, min_accuracy))
        ]
        
        if test_predictions:
            validations.append(('predictions', self.test_model_predictions, (ticker,)))
        
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
        logger.info(f"Model validation for {ticker}: {result['status']}")
        
        return result
    
    def validate_all_models(self, tickers: List[str], min_accuracy: float = 0.6, 
                          test_predictions: bool = False) -> Dict:
        """Validate models for all specified tickers"""
        logger.info(f"Starting model validation for {len(tickers)} tickers")
        
        results = []
        passed_count = 0
        failed_count = 0
        
        for ticker in tickers:
            try:
                result = self.validate_single_model(ticker, min_accuracy, test_predictions)
                results.append(result)
                
                if result['status'] == 'PASSED':
                    passed_count += 1
                    self.validation_results['passed_validations'].append(ticker)
                else:
                    failed_count += 1
                    self.validation_results['failed_validations'].append(ticker)
                
                self.validation_results['models_validated'].append(ticker)
                
            except Exception as e:
                logger.error(f"Unexpected error validating model for {ticker}: {str(e)}")
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
            'total_models': len(tickers),
            'passed': passed_count,
            'failed': failed_count,
            'success_rate': passed_count / len(tickers) * 100 if tickers else 0
        }
        
        self.validation_results['detailed_results'] = results
        
        return self.validation_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Validation for RoyFin AI")
    
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Ticker to validate'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help='Directory containing model files'
    )
    
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=0.6,
        help='Minimum required accuracy'
    )
    
    parser.add_argument(
        '--test-predictions',
        action='store_true',
        help='Test model prediction functionality'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        help='Output file for validation report'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Starting model validation process")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Minimum accuracy: {args.min_accuracy}")
    
    # Initialize validator
    validator = ModelValidator(args.models_dir)
    
    # Run validation for single ticker
    results = validator.validate_all_models([args.ticker], args.min_accuracy, args.test_predictions)
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation report saved to {args.output_report}")
    
    # Print summary
    summary = results['summary']
    logger.info("="*50)
    logger.info("MODEL VALIDATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Status: {results['overall_status']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    
    if results['failed_validations']:
        logger.warning("Failed validations found")
        for result in results['detailed_results']:
            if result['status'] == 'FAILED':
                logger.warning(f"Issues for {result['ticker']}: {result['issues']}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'FAILED':
        logger.error("Model validation failed")
        sys.exit(1)
    else:
        logger.info("Model validation completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()