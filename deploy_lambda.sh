#!/bin/bash

# 🚀 AWS Lambda Deployment Script for RoyFin AI Model Trainer
# This script helps deploy the model training Lambda function

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Configuration
FUNCTION_NAME="royfin-ai-model-trainer"
RUNTIME="python3.11"
HANDLER="lambda_model_trainer.lambda_handler"
TIMEOUT=900  # 15 minutes
MEMORY=3008  # Maximum memory
PACKAGE_DIR="lambda_package"
ZIP_FILE="lambda_function.zip"

print_header "RoyFin AI Lambda Deployment Script"

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install it first."
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install it first."
    exit 1
fi

print_success "Prerequisites check passed"

# Check AWS configuration
print_status "Checking AWS configuration..."
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")
print_success "AWS configured for account: $AWS_ACCOUNT_ID in region: $AWS_REGION"

# Create S3 bucket if it doesn't exist
BUCKET_NAME="royfin-ai-models-${AWS_ACCOUNT_ID}"
print_status "Checking S3 bucket: $BUCKET_NAME"

if ! aws s3 ls "s3://$BUCKET_NAME" &> /dev/null; then
    print_status "Creating S3 bucket: $BUCKET_NAME"
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3 mb "s3://$BUCKET_NAME"
    else
        aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION"
    fi
    print_success "S3 bucket created: $BUCKET_NAME"
else
    print_success "S3 bucket exists: $BUCKET_NAME"
fi

# Clean up previous builds
print_status "Cleaning up previous builds..."
rm -rf "$PACKAGE_DIR"
rm -f "$ZIP_FILE"

# Create package directory
print_status "Creating deployment package..."
mkdir -p "$PACKAGE_DIR"

# Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install -r cloud/functions/requirements_lambda.txt -t "$PACKAGE_DIR/" --quiet

# Copy function code
print_status "Copying function code..."
cp cloud/functions/lambda_model_trainer.py "$PACKAGE_DIR/"
cp model/stock_lstm.py "$PACKAGE_DIR/" 2>/dev/null || print_warning "stock_lstm.py not found, will be included in deployment"

# Create ZIP file
print_status "Creating deployment ZIP file..."
cd "$PACKAGE_DIR"
zip -r "../$ZIP_FILE" . -q
cd ..
print_success "Deployment package created: $ZIP_FILE"

# Create IAM role if it doesn't exist
ROLE_NAME="royfin-ai-lambda-role"
print_status "Checking IAM role: $ROLE_NAME"

if ! aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    print_status "Creating IAM role: $ROLE_NAME"
    
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create role
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file://trust-policy.json

    # Attach basic execution policy
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

    # Create and attach S3 policy
    cat > s3-policy.json << EOF
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
        "arn:aws:s3:::$BUCKET_NAME",
        "arn:aws:s3:::$BUCKET_NAME/*"
      ]
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "RoyFinAIS3Access" \
        --policy-document file://s3-policy.json

    # Clean up policy files
    rm -f trust-policy.json s3-policy.json

    print_success "IAM role created: $ROLE_NAME"
    
    # Wait for role to be available
    print_status "Waiting for IAM role to be available..."
    sleep 10
else
    print_success "IAM role exists: $ROLE_NAME"
fi

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)

# Deploy Lambda function
print_status "Deploying Lambda function: $FUNCTION_NAME"

if aws lambda get-function --function-name "$FUNCTION_NAME" &> /dev/null; then
    print_status "Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://$ZIP_FILE"
    
    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --handler "$HANDLER" \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY" \
        --environment Variables="{S3_BUCKET_NAME=$BUCKET_NAME}"
    
    print_success "Lambda function updated"
else
    print_status "Creating new Lambda function..."
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime "$RUNTIME" \
        --role "$ROLE_ARN" \
        --handler "$HANDLER" \
        --zip-file "fileb://$ZIP_FILE" \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY" \
        --environment Variables="{S3_BUCKET_NAME=$BUCKET_NAME}" \
        --description "RoyFin AI automated model training function"
    
    print_success "Lambda function created"
fi

# Test the function
print_status "Testing Lambda function..."
cat > test-event.json << EOF
{
  "source": "github_actions",
  "workflow_run_id": "test_run_$(date +%s)",
  "timestamp": "$(date -u +%Y%m%d_%H%M%S)",
  "trigger_type": "manual",
  "tickers": ["AAPL"],
  "force_retrain": true,
  "market_status": "closed",
  "data_ingestion": {
    "completed": true,
    "artifacts_available": true
  },
  "configuration": {
    "ensemble_size": 2,
    "epochs": 5,
    "optimize_hyperparams": false,
    "validation_split": 0.2
  }
}
EOF

print_status "Invoking test function..."
aws lambda invoke \
    --function-name "$FUNCTION_NAME" \
    --payload file://test-event.json \
    response.json

if [ $? -eq 0 ]; then
    print_success "Lambda function test invocation successful"
    echo "Response:"
    cat response.json | python3 -m json.tool 2>/dev/null || cat response.json
else
    print_error "Lambda function test invocation failed"
fi

# Clean up
print_status "Cleaning up temporary files..."
rm -rf "$PACKAGE_DIR"
rm -f "$ZIP_FILE"
rm -f test-event.json
rm -f response.json

print_header "Deployment Summary"
echo -e "${GREEN}✅ Lambda Function:${NC} $FUNCTION_NAME"
echo -e "${GREEN}✅ S3 Bucket:${NC} $BUCKET_NAME"
echo -e "${GREEN}✅ IAM Role:${NC} $ROLE_NAME"
echo -e "${GREEN}✅ Runtime:${NC} $RUNTIME"
echo -e "${GREEN}✅ Memory:${NC} ${MEMORY}MB"
echo -e "${GREEN}✅ Timeout:${NC} ${TIMEOUT}s"

print_header "Next Steps"
echo "1. Add the following secrets to your GitHub repository:"
echo "   - AWS_ACCESS_KEY_ID"
echo "   - AWS_SECRET_ACCESS_KEY"
echo "   - AWS_REGION (optional, defaults to your configured region)"
echo "   - AWS_LAMBDA_FUNCTION_NAME (optional, defaults to '$FUNCTION_NAME')"
echo ""
echo "2. Test the GitHub Actions workflow:"
echo "   - Go to Actions tab in your repository"
echo "   - Run 'Automated Data Ingestion & Model Retraining' manually"
echo ""
echo "3. Monitor the workflow execution:"
echo "   - Check GitHub Actions logs"
echo "   - Check AWS CloudWatch logs for Lambda function"
echo "   - Verify models are saved to S3 bucket: $BUCKET_NAME"

print_success "Deployment completed successfully! 🎉"