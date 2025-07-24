# GitHub Actions Workflow Status Badges

Add these badges to your repository README.md to show the status of the automated data ingestion and model retraining workflow:

## Main Workflow Badge

```markdown
[![Data Ingestion & Model Training](https://github.com/Rajdeep183/royfin_ai/actions/workflows/data-ingestion-training.yml/badge.svg)](https://github.com/Rajdeep183/royfin_ai/actions/workflows/data-ingestion-training.yml)
```

## Individual Job Badges

For more granular status reporting, you can create custom badges:

### Setup and Validation
```markdown
![Setup Status](https://img.shields.io/github/actions/workflow/status/Rajdeep183/royfin_ai/data-ingestion-training.yml?label=Setup%20%26%20Validation&style=flat-square)
```

### Data Ingestion
```markdown
![Data Ingestion](https://img.shields.io/github/actions/workflow/status/Rajdeep183/royfin_ai/data-ingestion-training.yml?label=Data%20Ingestion&style=flat-square)
```

### Model Training
```markdown
![Model Training](https://img.shields.io/github/actions/workflow/status/Rajdeep183/royfin_ai/data-ingestion-training.yml?label=Model%20Training&style=flat-square)
```

## Last Run Information

```markdown
![Last Run](https://img.shields.io/github/actions/workflow/status/Rajdeep183/royfin_ai/data-ingestion-training.yml?label=Last%20Run&style=for-the-badge)
```

## Complete Status Section

Here's a complete status section you can add to your README:

```markdown
## 🔄 Automated Pipeline Status

| Component | Status | Last Run |
|-----------|--------|----------|
| **Data Ingestion & Training** | [![Workflow Status](https://github.com/Rajdeep183/royfin_ai/actions/workflows/data-ingestion-training.yml/badge.svg)](https://github.com/Rajdeep183/royfin_ai/actions/workflows/data-ingestion-training.yml) | ![Last Run](https://img.shields.io/github/last-commit/Rajdeep183/royfin_ai?label=Last%20Update&style=flat-square) |
| **Data Quality** | ![Data Quality](https://img.shields.io/badge/Data%20Quality-Monitored-green?style=flat-square) | Continuous |
| **Model Performance** | ![Model Performance](https://img.shields.io/badge/Model%20Performance-Tracked-blue?style=flat-square) | Nightly |

### Pipeline Features
- 🕐 **Schedule**: Runs nightly at midnight UTC
- 📊 **Data Sources**: Yahoo Finance (yfinance)
- 🤖 **ML Platform**: AWS Lambda with PyTorch
- 📈 **Tickers**: AAPL, GOOGL, MSFT, TSLA, SPY, and more
- ⚡ **Processing**: Parallel ingestion and training
- 🔍 **Monitoring**: Comprehensive logging and validation
```

## Custom Badges

You can also create custom badges for specific metrics:

### Data Freshness
```markdown
![Data Freshness](https://img.shields.io/badge/Data%20Freshness-24h-brightgreen?style=flat-square)
```

### Model Accuracy
```markdown
![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-Tracking-blue?style=flat-square)
```

### Supported Tickers
```markdown
![Supported Tickers](https://img.shields.io/badge/Supported%20Tickers-10+-orange?style=flat-square)
```

## Implementation

To add these badges to your README.md:

1. Copy the desired badge markdown
2. Paste it into your README.md file
3. The badges will automatically update based on workflow status
4. Click badges to view detailed workflow runs

The workflow badges will show:
- ✅ **Green**: Last run was successful
- ❌ **Red**: Last run failed
- 🟡 **Yellow**: Workflow is currently running
- ⚪ **Gray**: No recent runs or workflow disabled