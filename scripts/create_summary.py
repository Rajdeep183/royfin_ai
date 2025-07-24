#!/usr/bin/env python3
"""
Deployment Summary Script for RoyFin AI
Creates a comprehensive summary of the deployment process
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentSummarizer:
    """Create deployment summaries and reports"""
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_summary(self, workflow_run_id: str, tickers: str, 
                      data_hash: str, results: str) -> Dict:
        """Create comprehensive deployment summary"""
        
        ticker_list = [t.strip() for t in tickers.split(',')]
        
        summary = {
            'deployment_info': {
                'workflow_run_id': workflow_run_id,
                'timestamp': datetime.now().isoformat(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'data_hash': data_hash,
                'results': results
            },
            'training_summary': {
                'total_tickers': len(ticker_list),
                'tickers': ticker_list,
                'status': results,
                'success_rate': 100.0 if results == 'success' else 0.0
            },
            'data_summary': {
                'data_hash': data_hash,
                'hash_short': data_hash[:8] if data_hash else 'unknown',
                'ingestion_date': datetime.now().strftime('%Y-%m-%d')
            },
            'deployment_steps': [],
            'next_actions': [],
            'links': {
                'workflow_url': f"https://github.com/Rajdeep183/royfin_ai/actions/runs/{workflow_run_id}",
                'repository_url': "https://github.com/Rajdeep183/royfin_ai"
            }
        }
        
        # Add deployment steps based on results
        if results == 'success':
            summary['deployment_steps'] = [
                {'step': 'Data Ingestion', 'status': 'completed', 'timestamp': datetime.now().isoformat()},
                {'step': 'Model Training', 'status': 'completed', 'timestamp': datetime.now().isoformat()},
                {'step': 'Model Validation', 'status': 'completed', 'timestamp': datetime.now().isoformat()},
                {'step': 'S3 Upload', 'status': 'completed', 'timestamp': datetime.now().isoformat()},
                {'step': 'Lambda Update', 'status': 'completed', 'timestamp': datetime.now().isoformat()}
            ]
            summary['next_actions'] = [
                'Models are ready for prediction',
                'Monitor model performance in production',
                'Schedule next training cycle'
            ]
        else:
            summary['deployment_steps'] = [
                {'step': 'Data Ingestion', 'status': 'completed', 'timestamp': datetime.now().isoformat()},
                {'step': 'Model Training', 'status': 'failed', 'timestamp': datetime.now().isoformat()}
            ]
            summary['next_actions'] = [
                'Review training logs for errors',
                'Check data quality',
                'Retry training with adjusted parameters'
            ]
        
        return summary
    
    def generate_markdown_report(self, summary: Dict) -> str:
        """Generate markdown report"""
        
        md_content = f"""# RoyFin AI Deployment Summary

## Deployment Information
- **Workflow Run ID**: {summary['deployment_info']['workflow_run_id']}
- **Date**: {summary['deployment_info']['date']}
- **Status**: {summary['deployment_info']['results'].upper()}
- **Data Hash**: `{summary['data_summary']['hash_short']}`

## Training Summary
- **Total Tickers**: {summary['training_summary']['total_tickers']}
- **Tickers Processed**: {', '.join(summary['training_summary']['tickers'])}
- **Success Rate**: {summary['training_summary']['success_rate']:.1f}%

## Deployment Steps
"""
        
        for step in summary['deployment_steps']:
            status_emoji = '✅' if step['status'] == 'completed' else '❌'
            md_content += f"- {status_emoji} **{step['step']}**: {step['status']}\n"
        
        md_content += f"""
## Next Actions
"""
        for action in summary['next_actions']:
            md_content += f"- {action}\n"
        
        md_content += f"""
## Links
- [Workflow Run]({summary['links']['workflow_url']})
- [Repository]({summary['links']['repository_url']})

---
*Generated on {summary['deployment_info']['timestamp']}*
"""
        
        return md_content
    
    def generate_html_report(self, summary: Dict) -> str:
        """Generate HTML report"""
        
        status_color = '#28a745' if summary['deployment_info']['results'] == 'success' else '#dc3545'
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RoyFin AI Deployment Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: {status_color}; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .step {{ margin: 10px 0; }}
        .success {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .info-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RoyFin AI Deployment Summary</h1>
        <p>Status: {summary['deployment_info']['results'].upper()}</p>
    </div>
    
    <div class="info-grid">
        <div class="info-card">
            <h3>Deployment Information</h3>
            <p><strong>Workflow Run ID:</strong> {summary['deployment_info']['workflow_run_id']}</p>
            <p><strong>Date:</strong> {summary['deployment_info']['date']}</p>
            <p><strong>Data Hash:</strong> <code>{summary['data_summary']['hash_short']}</code></p>
        </div>
        
        <div class="info-card">
            <h3>Training Summary</h3>
            <p><strong>Total Tickers:</strong> {summary['training_summary']['total_tickers']}</p>
            <p><strong>Success Rate:</strong> {summary['training_summary']['success_rate']:.1f}%</p>
            <p><strong>Tickers:</strong> {', '.join(summary['training_summary']['tickers'])}</p>
        </div>
    </div>
    
    <div class="section">
        <h3>Deployment Steps</h3>
"""
        
        for step in summary['deployment_steps']:
            status_class = 'success' if step['status'] == 'completed' else 'failed'
            status_emoji = '✅' if step['status'] == 'completed' else '❌'
            html_content += f'        <div class="step {status_class}">{status_emoji} <strong>{step["step"]}</strong>: {step["status"]}</div>\n'
        
        html_content += f"""    </div>
    
    <div class="section">
        <h3>Next Actions</h3>
        <ul>
"""
        
        for action in summary['next_actions']:
            html_content += f"            <li>{action}</li>\n"
        
        html_content += f"""        </ul>
    </div>
    
    <div class="section">
        <h3>Links</h3>
        <p><a href="{summary['links']['workflow_url']}" target="_blank">View Workflow Run</a></p>
        <p><a href="{summary['links']['repository_url']}" target="_blank">View Repository</a></p>
    </div>
    
    <footer>
        <p><em>Generated on {summary['deployment_info']['timestamp']}</em></p>
    </footer>
</body>
</html>"""
        
        return html_content
    
    def save_reports(self, summary: Dict) -> List[str]:
        """Save all report formats"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        # Save JSON summary
        json_file = self.output_dir / f"deployment_summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files.append(str(json_file))
        logger.info(f"Saved JSON summary: {json_file}")
        
        # Save Markdown report
        md_content = self.generate_markdown_report(summary)
        md_file = self.output_dir / f"deployment_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
        saved_files.append(str(md_file))
        logger.info(f"Saved Markdown report: {md_file}")
        
        # Save HTML report
        html_content = self.generate_html_report(summary)
        html_file = self.output_dir / f"deployment_report_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        saved_files.append(str(html_file))
        logger.info(f"Saved HTML report: {html_file}")
        
        # Save latest versions (without timestamp)
        latest_json = self.output_dir / "latest_deployment.json"
        with open(latest_json, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files.append(str(latest_json))
        
        return saved_files

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create deployment summary")
    
    parser.add_argument(
        '--workflow-run-id',
        type=str,
        required=True,
        help='GitHub Actions workflow run ID'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated list of tickers'
    )
    
    parser.add_argument(
        '--data-hash',
        type=str,
        required=True,
        help='Data hash'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        choices=['success', 'failure', 'partial'],
        required=True,
        help='Deployment results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./reports',
        help='Output directory for reports'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Creating deployment summary")
    logger.info(f"Workflow Run ID: {args.workflow_run_id}")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Results: {args.results}")
    
    try:
        # Initialize summarizer
        summarizer = DeploymentSummarizer(args.output_dir)
        
        # Create summary
        summary = summarizer.create_summary(
            workflow_run_id=args.workflow_run_id,
            tickers=args.tickers,
            data_hash=args.data_hash,
            results=args.results
        )
        
        # Save reports
        saved_files = summarizer.save_reports(summary)
        
        logger.info("="*50)
        logger.info("DEPLOYMENT SUMMARY CREATED")
        logger.info("="*50)
        logger.info(f"Status: {args.results.upper()}")
        logger.info(f"Tickers: {args.tickers}")
        logger.info(f"Files saved: {len(saved_files)}")
        
        for file_path in saved_files:
            logger.info(f"  - {file_path}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error creating deployment summary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()