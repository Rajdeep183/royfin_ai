# RoyFin AI Enhanced Workflow Deployment Guide

This guide explains how to deploy and configure the enhanced GitHub Actions workflow for automated model training.

## Overview

The enhanced workflow provides:
- **Automated Data Ingestion**: Multi-source market data fetching with yfinance
- **Intelligent Scheduling**: Market-aware training that respects trading hours
- **Advanced Validation**: Data quality, model performance, and freshness checks
- **AWS Integration**: S3 model storage, Lambda updates, CloudWatch monitoring
- **Comprehensive Notifications**: Slack alerts with detailed status reporting
- **Parallel Training**: Matrix strategy for efficient multi-ticker processing

## Prerequisites

### GitHub Repository Setup
1. Ensure the repository has the enhanced workflow files:
   - `.github/workflows/retrain.yml`
   - `scripts/` directory with all supporting scripts
   - `config/data_sources.yml` configuration file

### Required GitHub Secrets
Configure the following secrets in your repository settings:

#### AWS Configuration
```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET=your-royfin-ai-bucket
LAMBDA_FUNCTION_NAME=royfin-ai-predictor
```

#### Notification Configuration
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### AWS Infrastructure Setup

#### 1. Create S3 Bucket
```bash
aws s3 mb s3://your-royfin-ai-bucket
aws s3api put-bucket-versioning \
  --bucket your-royfin-ai-bucket \
  --versioning-configuration Status=Enabled
```

#### 2. Create Lambda Function (Optional)
```bash
# Create a basic Lambda function for model serving
aws lambda create-function \
  --function-name royfin-ai-predictor \
  --runtime python3.11 \
  --role arn:aws:iam::account:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip
```

#### 3. CloudWatch Setup
The workflow automatically creates metrics in the `RoyFinAI/ModelTraining` namespace.

## Workflow Configuration

### Schedule Configuration
The workflow runs:
- **Nightly**: 18:30 UTC (00:00 IST) on weekdays
- **Manual Trigger**: Via GitHub Actions UI with custom parameters

### Data Sources
Configure data sources in `config/data_sources.yml`:

#### Default Tickers
- **US Stocks**: AAPL, MSFT, GOOGL, TSLA, META, NVDA
- **Indian Stocks**: RELIANCE.NS, TCS.NS, HDFCBANK.NS
- **Indices**: SPY, QQQ, ^GSPC, ^DJI, ^IXIC
- **Volatility**: ^VIX, ^VXN, ^RVX

#### Custom Ticker Lists
Use manual workflow trigger with custom ticker parameter:
```
AAPL,MSFT,GOOGL,TSLA,RELIANCE.NS
```

## Workflow Jobs

### 1. Data Ingestion Job
**Purpose**: Fetch and validate market data
**Features**:
- Multi-source data fetching (US, Indian, indices, volatility)
- Concurrent processing with ThreadPoolExecutor
- Market hours validation
- Data quality checks
- Artifact caching

**Outputs**:
- Market data artifacts
- Data hash for versioning
- Market status information

### 2. Model Training Job
**Purpose**: Train models for each ticker in parallel
**Features**:
- Matrix strategy for parallel execution
- Data freshness checks
- Model validation
- Performance metrics collection
- S3 upload with versioning

**Configuration**:
- Max parallel jobs: 2 (configurable)
- Timeout: 120 minutes per ticker
- Retry logic: Built into individual scripts

### 3. Deployment Job
**Purpose**: Deploy models and update infrastructure
**Features**:
- Lambda function updates
- CloudWatch metrics reporting
- Deployment summary generation
- Status badge updates

### 4. Notification Job
**Purpose**: Send status notifications
**Features**:
- Slack notifications with rich formatting
- Success/failure/partial failure handling
- Error detail reporting
- Status badge integration

## Manual Workflow Execution

### GitHub Actions UI
1. Go to repository → Actions → "Nightly Model Retraining"
2. Click "Run workflow"
3. Configure parameters:
   - **Tickers**: Comma-separated list (e.g., "AAPL,MSFT,GOOGL")
   - **Force Retrain**: Override freshness checks
   - **Skip Validation**: Skip data validation (not recommended)

### API Trigger
```bash
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/Rajdeep183/royfin_ai/actions/workflows/retrain.yml/dispatches \
  -d '{"ref":"main","inputs":{"tickers":"AAPL,MSFT","force_retrain":"false"}}'
```

## Monitoring and Troubleshooting

### GitHub Actions Logs
- View detailed logs for each job in the Actions tab
- Download artifacts for debugging
- Check individual script outputs

### AWS CloudWatch
- Metrics namespace: `RoyFinAI/ModelTraining`
- Key metrics:
  - `WorkflowRuns`: Total workflow executions
  - `SuccessfulRuns`/`FailedRuns`: Success/failure counts
  - `TickersProcessed`: Number of tickers processed
  - `ModelAccuracy`: Model performance metrics

### Slack Notifications
- Real-time status updates
- Error summaries with actionable information
- Links to workflow runs and logs

### Common Issues and Solutions

#### 1. Data Fetching Failures
**Symptoms**: yfinance errors, network timeouts
**Solutions**:
- Check Yahoo Finance service status
- Verify ticker symbols are correct
- Use manual trigger with `force_retrain=true`

#### 2. Model Training Failures
**Symptoms**: Training timeouts, memory errors
**Solutions**:
- Reduce parallel jobs in workflow
- Check data quality
- Review model hyperparameters

#### 3. AWS Integration Issues
**Symptoms**: S3 upload failures, Lambda update errors
**Solutions**:
- Verify AWS credentials and permissions
- Check S3 bucket exists and is accessible
- Validate Lambda function exists

#### 4. Notification Failures
**Symptoms**: Missing Slack messages
**Solutions**:
- Verify webhook URL is correct
- Check Slack workspace permissions
- Review notification script logs

## Performance Optimization

### Workflow Efficiency
- **Data Caching**: Reduces redundant data fetching
- **Parallel Processing**: Matrix strategy for concurrent training
- **Conditional Execution**: Skip unnecessary steps based on data freshness
- **Artifact Management**: Efficient storage and retrieval

### Cost Optimization
- **S3 Lifecycle Policies**: Automatically archive old models
- **CloudWatch Retention**: Set appropriate log retention periods
- **Lambda Optimization**: Right-size function memory and timeout

### Scaling Considerations
- **Matrix Parallelism**: Adjust `max-parallel` based on GitHub Actions limits
- **Ticker Batching**: Group tickers for large-scale processing
- **Resource Limits**: Monitor and adjust timeouts and memory limits

## Security Best Practices

### Secrets Management
- Use GitHub Secrets for all sensitive data
- Rotate AWS credentials regularly
- Limit IAM permissions to minimum required

### Data Protection
- Enable S3 bucket encryption
- Use VPC endpoints for AWS services (production)
- Implement access logging

### Code Security
- Regular dependency updates
- Security scanning with GitHub CodeQL
- Review third-party integrations

## Maintenance

### Regular Tasks
- **Weekly**: Review workflow logs and performance metrics
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and optimize AWS costs

### Updates and Improvements
- Monitor GitHub Actions feature updates
- Update yfinance library for new data sources
- Enhance notification templates based on team feedback

## Support and Documentation

### Internal Documentation
- `scripts/README.md`: Detailed script documentation
- `config/data_sources.yml`: Configuration reference
- Workflow comments: Inline documentation

### External Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [AWS SDK Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

## Deployment Checklist

- [ ] Repository secrets configured
- [ ] AWS infrastructure deployed
- [ ] S3 bucket created and accessible
- [ ] Lambda function deployed (if used)
- [ ] Slack webhook configured
- [ ] Test manual workflow execution
- [ ] Verify notifications work
- [ ] Monitor first automated run
- [ ] Set up CloudWatch alerts
- [ ] Document any customizations

## Next Steps

After successful deployment:
1. **Monitor Performance**: Track success rates and execution times
2. **Optimize Configuration**: Adjust parameters based on actual usage
3. **Expand Data Sources**: Add new tickers and markets as needed
4. **Enhance Notifications**: Customize messages for different stakeholders
5. **Scale Infrastructure**: Upgrade AWS resources as volume increases