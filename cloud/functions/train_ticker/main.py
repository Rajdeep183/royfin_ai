import subprocess
import json
import os
import sys
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(request, context):
    """
    Enhanced HTTP Cloud Function to train advanced stock model.
    Query parameters: ticker, start_date, optimize_hyperparams, ensemble_size
    Example: ?ticker=AAPL&start_date=2015-01-01&optimize_hyperparams=true&ensemble_size=5
    """
    request_args = request.args

    ticker = request_args.get('ticker')
    start_date = request_args.get('start_date', '2015-01-01')
    optimize_hyperparams = request_args.get('optimize_hyperparams', 'true').lower() == 'true'
    ensemble_size = int(request_args.get('ensemble_size', '5'))
    n_trials = int(request_args.get('n_trials', '50'))  # For hyperparameter optimization

    if not ticker:
        return ({"error": "Missing required parameter: ticker"}, 400)

    try:
        # Add the lib directory to Python path
        lib_path = os.path.join(os.path.dirname(__file__), '..', 'lib')
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
        
        # Enhanced training command with better error handling
        command = [
            "python3", 
            os.path.join(lib_path, "model", "stock_lstm.py"), 
            "train",
            "--ticker", ticker,
            "--start_date", start_date,
            "--optimize_hyperparams", str(optimize_hyperparams),
            "--ensemble_size", str(ensemble_size),
            "--n_trials", str(n_trials),
            "--output_format", "json"  # Request JSON output for better parsing
        ]

        logger.info(f"🚀 Starting enhanced training for {ticker}")
        logger.info(f"Parameters: start_date={start_date}, optimize={optimize_hyperparams}, ensemble_size={ensemble_size}")
        logger.info(f"Command: {' '.join(command)}")

        # Set environment variables for the subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = lib_path
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=3600,  # 1 hour timeout for training
            env=env,
            cwd=lib_path
        )

        # Enhanced result parsing
        training_info = {
            "ticker": ticker,
            "start_date": start_date,
            "training_completed": datetime.now().isoformat(),
            "hyperparameter_optimization": optimize_hyperparams,
            "ensemble_size": ensemble_size,
            "n_trials": n_trials,
            "command": " ".join(command),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0
        }

        # Enhanced metrics extraction with JSON parsing fallback
        try:
            # Try to parse as JSON first (if --output_format json was used)
            if result.stdout.strip().startswith('{'):
                json_output = json.loads(result.stdout)
                training_info.update(json_output)
            else:
                # Fallback to text parsing
                metrics = parse_training_metrics(result.stdout)
                if metrics:
                    training_info['performance_metrics'] = metrics
                    
        except json.JSONDecodeError:
            # Try text parsing as fallback
            metrics = parse_training_metrics(result.stdout)
            if metrics:
                training_info['performance_metrics'] = metrics
        except Exception as e:
            logger.warning(f"Could not parse training metrics: {e}")

        # Add helpful status messages
        if result.returncode == 0:
            training_info['message'] = f"✅ Training completed successfully for {ticker}"
        else:
            training_info['message'] = f"❌ Training failed for {ticker}"
            training_info['error_details'] = result.stderr

        status_code = 200 if result.returncode == 0 else 500
        return (training_info, status_code)

    except subprocess.TimeoutExpired:
        error_msg = f"Training timeout for {ticker} - consider reducing ensemble_size ({ensemble_size}) or n_trials ({n_trials})"
        logger.error(error_msg)
        return ({"error": error_msg, "ticker": ticker, "timeout": True}, 408)
        
    except FileNotFoundError as e:
        error_msg = f"Training script not found: {str(e)}"
        logger.error(error_msg)
        return ({"error": error_msg, "ticker": ticker}, 404)
        
    except Exception as e:
        error_msg = f"Training failed for {ticker}: {str(e)}"
        logger.error(error_msg)
        return ({"error": error_msg, "ticker": ticker}, 500)

def parse_training_metrics(stdout_text):
    """Parse training metrics from stdout text"""
    try:
        metrics = {}
        lines = stdout_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if "Price MAE:" in line:
                try:
                    metrics['price_mae'] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif "Price R²:" in line or "Price R2:" in line:
                try:
                    metrics['price_r2'] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif "Trend Accuracy:" in line:
                try:
                    metrics['trend_accuracy'] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif "Training Loss:" in line:
                try:
                    metrics['training_loss'] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif "Validation Loss:" in line:
                try:
                    metrics['validation_loss'] = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif "Best Hyperparameters:" in line:
                # Try to parse hyperparameters if they're in the next lines
                metrics['hyperparameters_optimized'] = True
        
        return metrics if metrics else None
        
    except Exception as e:
        logger.warning(f"Error parsing metrics: {e}")
        return None
