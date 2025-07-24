# 🚀 RoyFin AI - GitHub Actions Setup & Testing Guide

This guide provides step-by-step instructions for setting up and testing the automated data ingestion and model retraining workflow.

## 📋 Prerequisites

Before setting up the workflow, ensure you have:

1. **GitHub Repository Access**: Admin access to the RoyFin AI repository
2. **AWS Account**: Active AWS account with appropriate permissions
3. **AWS CLI**: Installed and configured locally (for Lambda deployment)
4. **Python 3.11+**: For local testing and development

## 🔧 Setup Process

### Step 1: AWS Infrastructure Setup

#### Option A: Automated Setup (Recommended)
```bash
# Run the automated deployment script
./deploy_lambda.sh
```

This script will:
- Create IAM roles and policies
- Set up S3 bucket for model storage
- Deploy the Lambda function
- Test the deployment

#### Option B: Manual Setup

1. **Create IAM Role**:
   ```bash
   aws iam create-role \
     --role-name royfin-ai-lambda-role \
     --assume-role-policy-document file://trust-policy.json
   ```

2. **Create S3 Bucket**:
   ```bash
   aws s3 mb s3://royfin-ai-models-YOUR_ACCOUNT_ID
   ```

3. **Deploy Lambda Function**:
   ```bash
   # Package dependencies
   pip install -r cloud/functions/requirements_lambda.txt -t lambda_package/
   
   # Create deployment package
   cd lambda_package && zip -r ../lambda_function.zip .
   
   # Deploy function
   aws lambda create-function \
     --function-name royfin-ai-model-trainer \
     --runtime python3.11 \
     --role arn:aws:iam::YOUR_ACCOUNT:role/royfin-ai-lambda-role \
     --handler lambda_model_trainer.lambda_handler \
     --zip-file fileb://lambda_function.zip \
     --timeout 900 \
     --memory-size 3008
   ```

### Step 2: GitHub Secrets Configuration

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

#### Required Secrets
```
AWS_ACCESS_KEY_ID          # Your AWS access key
AWS_SECRET_ACCESS_KEY      # Your AWS secret key
AWS_REGION                 # AWS region (e.g., us-east-1)
```

#### Optional Secrets
```
AWS_LAMBDA_FUNCTION_NAME   # Lambda function name (defaults to royfin-ai-model-trainer)
SLACK_WEBHOOK_URL          # For Slack notifications (optional)
```

#### Setting up AWS Credentials

1. **Create IAM User for GitHub Actions**:
   ```bash
   aws iam create-user --user-name github-actions-royfin
   ```

2. **Create Access Keys**:
   ```bash
   aws iam create-access-key --user-name github-actions-royfin
   ```

3. **Attach Policies**:
   ```bash
   # Lambda invocation permission
   aws iam attach-user-policy \
     --user-name github-actions-royfin \
     --policy-arn arn:aws:iam::aws:policy/AWSLambdaRole
   ```

### Step 3: Workflow Configuration

The workflow is pre-configured with sensible defaults, but you can customize:

#### Environment Variables in Workflow
```yaml
env:
  PYTHON_VERSION: '3.11'                    # Python version
  DEFAULT_TICKERS: 'AAPL,GOOGL,MSFT,TSLA,SPY'  # Default ticker symbols
  ARTIFACTS_RETENTION_DAYS: 30              # How long to keep artifacts
```

#### Scheduling
```yaml
schedule:
  - cron: '0 0 * * *'  # Daily at midnight UTC
```

Change the cron expression to modify the schedule:
- `0 0 * * *` - Daily at midnight
- `0 0 * * 1` - Weekly on Mondays
- `0 0 1 * *` - Monthly on the 1st

## 🧪 Testing

### Test 1: Manual Workflow Trigger

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Select "Automated Data Ingestion & Model Retraining"
4. Click "Run workflow"
5. Configure test parameters:
   - **ticker_symbols**: `AAPL,GOOGL`
   - **days_of_data**: `365`
   - **force_retrain**: `true`
   - **skip_lambda**: `false`
6. Click "Run workflow"

### Test 2: Local Data Ingestion Test

```bash
# Test yfinance connectivity
python3 << 'EOF'
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test basic data fetch
ticker = yf.Ticker("AAPL")
data = ticker.history(period="5d")
print(f"✅ Retrieved {len(data)} records")
print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
EOF
```

### Test 3: Lambda Function Test

```bash
# Test Lambda function directly
aws lambda invoke \
  --function-name royfin-ai-model-trainer \
  --payload '{
    "source": "test",
    "tickers": ["AAPL"],
    "force_retrain": true,
    "configuration": {
      "ensemble_size": 2,
      "epochs": 5
    }
  }' \
  response.json

cat response.json
```

### Test 4: End-to-End Integration Test

1. **Trigger Workflow**: Run workflow manually with test parameters
2. **Monitor Progress**: Check GitHub Actions logs
3. **Verify Data Artifacts**: Download and inspect data artifacts
4. **Check Lambda Execution**: View CloudWatch logs
5. **Validate S3 Storage**: Confirm models saved to S3

## 📊 Monitoring & Troubleshooting

### GitHub Actions Monitoring

1. **Workflow Status**: Check Actions tab for run status
2. **Job Logs**: Click on failed jobs to see detailed logs
3. **Artifacts**: Download artifacts for debugging
4. **Summary**: View workflow summary in job output

### AWS CloudWatch Monitoring

1. **Lambda Logs**: `/aws/lambda/royfin-ai-model-trainer`
2. **Error Tracking**: Filter by ERROR level
3. **Performance**: Monitor duration and memory usage

### Common Issues & Solutions

#### Issue: Data Ingestion Fails
```
Error: No data retrieved for ticker AAPL
```
**Solutions**:
- Check yfinance service status
- Verify ticker symbols are correct
- Check rate limiting

#### Issue: Lambda Invocation Fails
```
Error: AccessDeniedException
```
**Solutions**:
- Verify AWS credentials in GitHub Secrets
- Check IAM permissions
- Confirm Lambda function exists

#### Issue: Model Training Timeout
```
Error: Task timed out after 900.00 seconds
```
**Solutions**:
- Increase Lambda timeout (max 15 minutes)
- Reduce ensemble size or epochs
- Optimize training parameters

#### Issue: S3 Access Denied
```
Error: Access Denied when saving to S3
```
**Solutions**:
- Check S3 bucket permissions
- Verify IAM role has S3 access
- Confirm bucket exists

### Debugging Commands

```bash
# Check workflow syntax
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/data-ingestion-training.yml'))"

# Validate Lambda function
python3 -m py_compile cloud/functions/lambda_model_trainer.py

# Test AWS connectivity
aws sts get-caller-identity

# Check Lambda function status
aws lambda get-function --function-name royfin-ai-model-trainer

# List S3 bucket contents
aws s3 ls s3://royfin-ai-models-YOUR_ACCOUNT_ID/
```

## 🔄 Maintenance

### Regular Tasks

1. **Weekly**: Review workflow execution logs
2. **Monthly**: Check S3 storage costs and cleanup old models
3. **Quarterly**: Update dependencies and test workflow

### Updates

#### Update Python Dependencies
```bash
# Update requirements
pip-review --local --auto

# Test locally
pip install -r requirements_streamlit.txt

# Redeploy Lambda
./deploy_lambda.sh
```

#### Update Workflow Configuration
1. Modify `.github/workflows/data-ingestion-training.yml`
2. Test with manual trigger
3. Monitor scheduled runs

#### Scale Lambda Resources
```bash
# Increase memory for better performance
aws lambda update-function-configuration \
  --function-name royfin-ai-model-trainer \
  --memory-size 3008

# Increase timeout for larger datasets
aws lambda update-function-configuration \
  --function-name royfin-ai-model-trainer \
  --timeout 900
```

## 📈 Performance Optimization

### Data Ingestion Optimization
- Use parallel processing for multiple tickers
- Implement intelligent data caching
- Optimize API call frequency

### Model Training Optimization
- Use GPU instances for Lambda (when available)
- Implement model checkpointing
- Optimize hyperparameters automatically

### Cost Optimization
- Monitor Lambda execution costs
- Implement lifecycle policies for S3 storage
- Use spot instances for training (future enhancement)

## 🎯 Next Steps

After successful setup:

1. **Monitor Performance**: Track model accuracy over time
2. **Expand Tickers**: Add more stocks to the analysis
3. **Enhanced Features**: Add more technical indicators
4. **Real-time Updates**: Implement real-time data ingestion
5. **Model Comparison**: A/B test different model architectures

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Review GitHub Actions logs
3. Check AWS CloudWatch logs
4. Create an issue in the repository with:
   - Error messages
   - Workflow run ID
   - Configuration details

## 🎉 Success Metrics

Your setup is successful when:

- ✅ Workflow runs without errors
- ✅ Data is successfully ingested for all tickers
- ✅ Lambda function executes and trains models
- ✅ Models are saved to S3
- ✅ Workflow completes within expected timeframes
- ✅ Monitoring and alerts are working

Congratulations! Your automated ML pipeline is now operational. 🚀