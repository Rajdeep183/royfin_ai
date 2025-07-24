# RoyFin AI - Automated Data Ingestion & Model Retraining

This directory contains the GitHub Actions workflow for automated financial data ingestion using yfinance and nightly model retraining using AWS Lambda.

## 🎯 Overview

The workflow provides:
- **Automated Data Ingestion**: Scheduled fetching of financial data using yfinance
- **Nightly Model Retraining**: AWS Lambda-based model training on fresh data
- **Comprehensive Monitoring**: Error handling, logging, and notifications
- **Parallel Processing**: Efficient handling of multiple ticker symbols
- **Data Validation**: Quality checks and validation of ingested data

## 📋 Workflow Features

### 1. Scheduled Execution
- Runs nightly at midnight UTC (during market closed hours)
- Manual trigger support for testing and on-demand runs
- Configurable ticker symbols and data parameters

### 2. Data Ingestion
- Fetches financial data using yfinance with retry logic
- Processes multiple tickers in parallel (default: AAPL, GOOGL, MSFT, TSLA, SPY)
- Calculates technical indicators (SMA, RSI, volatility, beta)
- Validates data quality and completeness
- Stores both raw and processed data as artifacts

### 3. AWS Lambda Integration
- Triggers AWS Lambda function for model retraining
- Passes comprehensive payload with training configuration
- Supports asynchronous execution for long-running training jobs
- Monitors execution status and provides logging

### 4. Error Handling & Monitoring
- Comprehensive error handling with retry mechanisms
- Detailed logging at each step
- Workflow summary with status reporting
- Optional Slack notifications
- Artifact storage for debugging and analysis

## ⚙️ Configuration

### Required GitHub Secrets

```yaml
# AWS Configuration
AWS_ACCESS_KEY_ID: "your-aws-access-key"
AWS_SECRET_ACCESS_KEY: "your-aws-secret-key"
AWS_REGION: "us-east-1"  # Optional, defaults to us-east-1
AWS_LAMBDA_FUNCTION_NAME: "royfin-ai-model-trainer"  # Optional, defaults to royfin-ai-model-trainer

# Optional Notifications
SLACK_WEBHOOK_URL: "https://hooks.slack.com/services/..."  # Optional Slack notifications
```

### Environment Variables

The workflow uses several configurable environment variables:

```yaml
# Python configuration
PYTHON_VERSION: '3.11'

# Data storage
DATA_STORAGE_PATH: 'data/financial'
MODELS_STORAGE_PATH: 'models'
ARTIFACTS_RETENTION_DAYS: 30

# Market configuration
MARKET_TIMEZONE: 'America/New_York'
DEFAULT_TICKERS: 'AAPL,GOOGL,MSFT,TSLA,SPY,QQQ,VTI,BND,GLD,^VIX'
```

## 🚀 Usage

### Manual Trigger

You can manually trigger the workflow with custom parameters:

1. Go to the "Actions" tab in your GitHub repository
2. Select "Automated Data Ingestion & Model Retraining"
3. Click "Run workflow"
4. Configure parameters:
   - **ticker_symbols**: Comma-separated list (e.g., "AAPL,GOOGL,MSFT")
   - **days_of_data**: Number of historical days to fetch (default: 1825)
   - **force_retrain**: Force retraining even if recent models exist
   - **skip_lambda**: Skip AWS Lambda trigger (for testing data ingestion only)

### Scheduled Execution

The workflow automatically runs nightly at midnight UTC. This timing is chosen because:
- US markets are closed (data is stable)
- Reduced API rate limiting
- Optimal time for batch processing

## 📊 Data Processing

### Input Data Sources
- **Primary**: Yahoo Finance via yfinance
- **Market Context**: SPY (S&P 500), VIX (Volatility Index)
- **Time Range**: Configurable (default: 5 years of historical data)

### Processed Features
- **Price Data**: Open, High, Low, Close, Volume
- **Technical Indicators**: SMA (20, 50), RSI, Volatility
- **Market Context**: Beta calculation vs SPY
- **Metadata**: Symbol, data source, ingestion timestamp

### Data Validation
- Minimum data points check (>100 records)
- Missing value validation (<10% missing prices)
- Trading volume validation
- Date range verification

### Output Artifacts
- **Raw Data**: `data/financial/raw/{ticker}_{timestamp}.csv`
- **Processed Data**: `data/financial/processed/{ticker}_latest.csv`
- **Validation Results**: `data/financial/validation/{ticker}_{timestamp}_validation.json`
- **Logs**: `logs/ingestion_{ticker}_{timestamp}.log`

## 🤖 AWS Lambda Integration

### Lambda Function Requirements

Your AWS Lambda function should:
- Accept the payload format defined in the workflow
- Have sufficient timeout (15+ minutes for training)
- Have access to S3 for model storage
- Include the required Python packages

### Payload Structure

```json
{
  "source": "github_actions",
  "workflow_run_id": "123456789",
  "timestamp": "20240724_120000",
  "trigger_type": "scheduled|manual",
  "tickers": ["AAPL", "GOOGL", "MSFT"],
  "force_retrain": false,
  "market_status": "closed",
  "data_ingestion": {
    "completed": true,
    "artifacts_available": true
  },
  "configuration": {
    "ensemble_size": 5,
    "epochs": 100,
    "optimize_hyperparams": true,
    "validation_split": 0.2
  }
}
```

### Lambda Function Setup

The included `lambda_model_trainer.py` provides a complete Lambda function implementation:

1. **Setup AWS Lambda Function**:
   ```bash
   # Create deployment package
   pip install -r cloud/functions/requirements_lambda.txt -t lambda_package/
   cp cloud/functions/lambda_model_trainer.py lambda_package/
   cp model/stock_lstm.py lambda_package/
   
   # Create ZIP file
   cd lambda_package && zip -r ../lambda_function.zip .
   ```

2. **Configure Lambda**:
   - Runtime: Python 3.11
   - Memory: 3008 MB (maximum)
   - Timeout: 15 minutes
   - Environment variables: S3_BUCKET_NAME

3. **IAM Permissions**:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:PutObject",
           "s3:DeleteObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::your-bucket-name",
           "arn:aws:s3:::your-bucket-name/*"
         ]
       },
       {
         "Effect": "Allow",
         "Action": [
           "logs:CreateLogGroup",
           "logs:CreateLogStream",
           "logs:PutLogEvents"
         ],
         "Resource": "arn:aws:logs:*:*:*"
       }
     ]
   }
   ```

## 📈 Monitoring & Troubleshooting

### Workflow Status

Monitor workflow execution through:
- GitHub Actions interface
- Workflow summary in job output
- Downloaded artifacts
- Optional Slack notifications

### Common Issues

1. **Data Ingestion Failures**:
   - Check yfinance API availability
   - Verify ticker symbols are valid
   - Review rate limiting

2. **Lambda Invocation Failures**:
   - Verify AWS credentials
   - Check Lambda function exists
   - Review IAM permissions

3. **Model Training Failures**:
   - Check Lambda CloudWatch logs
   - Verify S3 bucket permissions
   - Review memory/timeout limits

### Debugging

1. **Download Artifacts**:
   - Financial data files
   - Validation results
   - Training logs
   - Lambda payload/response

2. **Check Logs**:
   - GitHub Actions job logs
   - AWS CloudWatch logs
   - Artifact log files

## 🔒 Security Considerations

- All sensitive data stored in GitHub Secrets
- AWS credentials with minimal required permissions
- No hardcoded API keys or credentials
- Secure artifact handling with limited retention

## 🚀 Deployment Checklist

- [ ] Configure GitHub Secrets (AWS credentials)
- [ ] Set up AWS Lambda function
- [ ] Configure S3 bucket for model storage
- [ ] Test manual workflow trigger
- [ ] Verify data ingestion artifacts
- [ ] Confirm Lambda function execution
- [ ] Set up monitoring/notifications (optional)

## 📝 Customization

### Adding New Tickers
Modify the `DEFAULT_TICKERS` environment variable or use manual trigger with custom ticker list.

### Adjusting Schedule
Modify the cron expression in the workflow file:
```yaml
schedule:
  - cron: '0 0 * * *'  # Daily at midnight UTC
```

### Custom Data Processing
Extend the data ingestion Python script to add:
- Additional technical indicators
- Alternative data sources
- Custom validation rules

### Enhanced Notifications
Add integrations for:
- Discord webhooks
- Microsoft Teams
- Email notifications
- Custom monitoring systems

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS Lambda Python Documentation](https://docs.aws.amazon.com/lambda/latest/dg/python-programming-model.html)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [RoyFin AI Model Documentation](../model/README.md)