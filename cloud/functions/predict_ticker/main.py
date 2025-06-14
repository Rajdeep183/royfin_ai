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
    Enhanced HTTP Cloud Function to predict stock prices using advanced ML model.
    Query parameters: ticker, days_ahead, confidence_level, ensemble_size
    Example: ?ticker=AAPL&days_ahead=30&confidence_level=0.95&ensemble_size=5
    """
    request_args = request.args

    ticker = request_args.get('ticker')
    days_ahead = request_args.get('days_ahead', '30')
    confidence_level = request_args.get('confidence_level', '0.95')
    ensemble_size = request_args.get('ensemble_size', '5')
    include_uncertainty = request_args.get('include_uncertainty', 'false').lower() == 'true'

    if not ticker:
        return ({"error": "Missing required parameter: ticker"}, 400)

    try:
        # Add the lib directory to Python path
        lib_path = os.path.join(os.path.dirname(__file__), '..', 'lib')
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
        
        # Enhanced prediction command
        command = [
            "python3", 
            os.path.join(lib_path, "model", "stock_lstm.py"), 
            "predict",
            "--ticker", ticker,
            "--days_ahead", str(days_ahead),
            "--confidence_level", str(confidence_level),
            "--ensemble_size", str(ensemble_size),
            "--include_uncertainty", str(include_uncertainty),
            "--output_format", "json"
        ]

        logger.info(f"🔮 Starting enhanced prediction for {ticker}")
        logger.info(f"Parameters: days_ahead={days_ahead}, confidence={confidence_level}, ensemble_size={ensemble_size}")

        # Set environment variables for the subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = lib_path
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=600,  # 10 minute timeout for prediction
            env=env,
            cwd=lib_path
        )

        # Enhanced result parsing
        prediction_info = {
            "ticker": ticker,
            "days_ahead": days_ahead,
            "confidence_level": confidence_level,
            "ensemble_size": ensemble_size,
            "prediction_generated": datetime.now().isoformat(),
            "command": " ".join(command),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0
        }

        # Parse JSON output or fallback to text parsing
        try:
            if result.stdout.strip().startswith('{') or result.stdout.strip().startswith('['):
                json_output = json.loads(result.stdout)
                if isinstance(json_output, dict):
                    prediction_info.update(json_output)
                else:
                    prediction_info['predictions'] = json_output
            else:
                # Fallback parsing for text output
                predictions = parse_prediction_output(result.stdout)
                if predictions:
                    prediction_info['predictions'] = predictions
                    
        except json.JSONDecodeError:
            predictions = parse_prediction_output(result.stdout)
            if predictions:
                prediction_info['predictions'] = predictions
        except Exception as e:
            logger.warning(f"Could not parse prediction output: {e}")

        # Add helpful status messages
        if result.returncode == 0:
            prediction_info['message'] = f"✅ Predictions generated successfully for {ticker}"
            if 'predictions' in prediction_info:
                prediction_info['prediction_count'] = len(prediction_info['predictions'])
        else:
            prediction_info['message'] = f"❌ Prediction failed for {ticker}"
            prediction_info['error_details'] = result.stderr

        status_code = 200 if result.returncode == 0 else 500
        return (prediction_info, status_code)

    except subprocess.TimeoutExpired:
        error_msg = f"Prediction timeout for {ticker} - model may need optimization"
        logger.error(error_msg)
        return ({"error": error_msg, "ticker": ticker, "timeout": True}, 408)
        
    except FileNotFoundError as e:
        error_msg = f"Prediction script not found: {str(e)}"
        logger.error(error_msg)
        return ({"error": error_msg, "ticker": ticker}, 404)
        
    except Exception as e:
        error_msg = f"Prediction failed for {ticker}: {str(e)}"
        logger.error(error_msg)
        return ({"error": error_msg, "ticker": ticker}, 500)

def parse_prediction_output(stdout_text):
    """Parse prediction output from stdout text"""
    try:
        predictions = []
        lines = stdout_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if "Prediction for" in line and ":" in line:
                try:
                    # Parse lines like "Prediction for 2025-06-13: $150.25 (confidence: 0.85)"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        date_part = parts[0].replace("Prediction for", "").strip()
                        value_part = parts[1].strip()
                        
                        # Extract price
                        price_str = value_part.split("$")[1].split()[0] if "$" in value_part else None
                        if price_str:
                            price = float(price_str)
                            
                            # Extract confidence if available
                            confidence = None
                            if "confidence:" in value_part:
                                conf_str = value_part.split("confidence:")[1].strip().rstrip(")")
                                confidence = float(conf_str)
                            
                            predictions.append({
                                "date": date_part,
                                "predicted_price": price,
                                "confidence": confidence
                            })
                            
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse prediction line: {line}, error: {e}")
                    continue
        
        return predictions if predictions else None
        
    except Exception as e:
        logger.warning(f"Error parsing predictions: {e}")
        return None
