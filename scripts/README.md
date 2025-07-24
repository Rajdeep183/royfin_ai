# RoyFin AI Scripts

This directory contains automated scripts for the RoyFin AI model training pipeline.

## Scripts Overview

### Data Management
- **`data_ingestion.py`** - Fetches market data from multiple sources using yfinance
- **`validate_data.py`** - Validates data quality, completeness, and freshness
- **`market_hours_check.py`** - Checks market hours and trading days for different markets

### Model Management
- **`validate_model.py`** - Validates trained models and their performance
- **`check_model_freshness.py`** - Checks if models need retraining based on data changes
- **`upload_model_s3.py`** - Uploads trained models to AWS S3 with versioning

### Infrastructure
- **`update_lambda.py`** - Updates AWS Lambda function with new model information
- **`send_metrics.py`** - Sends workflow metrics to AWS CloudWatch
- **`send_notification.py`** - Sends notifications about workflow status via Slack

### Reporting
- **`create_summary.py`** - Creates comprehensive deployment summaries
- **`update_status_badge.py`** - Updates workflow status badges

## Usage Examples

### Data Ingestion
```bash
# Fetch data for specific tickers
python scripts/data_ingestion.py \
  --tickers "AAPL,MSFT,GOOGL" \
  --output-dir "./data" \
  --include-indices \
  --include-volatility

# Fetch data for Indian stocks
python scripts/data_ingestion.py \
  --tickers "RELIANCE.NS,TCS.NS" \
  --include-indian-stocks
```

### Data Validation
```bash
# Validate data quality
python scripts/validate_data.py \
  --data-dir "./data" \
  --tickers "AAPL,MSFT" \
  --min-days 200 \
  --max-missing-days 5
```

### Market Hours Check
```bash
# Check if markets are open
python scripts/market_hours_check.py --market ALL
```

### Model Validation
```bash
# Validate a trained model
python scripts/validate_model.py \
  --ticker "AAPL" \
  --min-accuracy 0.6 \
  --test-predictions
```

### Model Freshness Check
```bash
# Check if model needs retraining
python scripts/check_model_freshness.py \
  --ticker "AAPL" \
  --data-hash "abc123..." \
  --s3-bucket "my-models-bucket"
```

### S3 Upload
```bash
# Upload model to S3
python scripts/upload_model_s3.py \
  --ticker "AAPL" \
  --bucket "my-models-bucket" \
  --data-hash "abc123..."
```

### AWS Integration
```bash
# Update Lambda function
python scripts/update_lambda.py \
  --function-name "royfin-ai-predictor" \
  --tickers "AAPL,MSFT" \
  --data-hash "abc123..."

# Send CloudWatch metrics
python scripts/send_metrics.py \
  --workflow-run-id "123456789" \
  --tickers "AAPL,MSFT" \
  --training-results "success"
```

### Notifications
```bash
# Send Slack notification
python scripts/send_notification.py \
  --status "success" \
  --workflow-run-id "123456789" \
  --tickers "AAPL,MSFT" \
  --webhook-url "https://hooks.slack.com/..."
```

### Reporting
```bash
# Create deployment summary
python scripts/create_summary.py \
  --workflow-run-id "123456789" \
  --tickers "AAPL,MSFT" \
  --data-hash "abc123..." \
  --results "success"

# Update status badge
python scripts/update_status_badge.py \
  --status "success" \
  --workflow-run-id "123456789"
```

## Configuration

Scripts use the configuration file `config/data_sources.yml` for:
- Data source definitions
- Market hours and trading days
- Default parameters
- AWS settings
- Notification settings

## Dependencies

All scripts require the dependencies listed in:
- `requirements_streamlit.txt` - Core dependencies
- `model/requirements.txt` - ML-specific dependencies

Key dependencies include:
- `yfinance` - Market data fetching
- `boto3` - AWS integration
- `requests` - HTTP requests and notifications
- `pandas`, `numpy` - Data processing
- `pytz` - Timezone handling

## Error Handling

All scripts include:
- Comprehensive logging
- Retry logic for network operations
- Graceful error handling
- Appropriate exit codes for CI/CD integration

## GitHub Actions Integration

These scripts are designed to work seamlessly with the GitHub Actions workflow defined in `.github/workflows/retrain.yml`. They support:
- Environment variable configuration
- GitHub Actions output format
- Artifact handling
- Status reporting

## Environment Variables

Scripts can be configured using environment variables:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `AWS_REGION` - AWS region
- `S3_BUCKET` - S3 bucket name
- `LAMBDA_FUNCTION_NAME` - Lambda function name
- `SLACK_WEBHOOK_URL` - Slack webhook for notifications