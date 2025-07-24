#!/usr/bin/env python3
"""
Notification Script for RoyFin AI
Sends notifications about workflow status via Slack or other channels
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Optional
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NotificationSender:
    """Send notifications about workflow status"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
    
    def create_slack_message(self, status: str, workflow_run_id: str, 
                           tickers: str, error_details: Optional[str] = None) -> Dict:
        """Create Slack message payload"""
        
        # Color coding
        color_map = {
            'success': '#36a64f',  # Green
            'failure': '#ff0000',  # Red
            'warning': '#ffaa00',  # Orange
            'info': '#0099ff'      # Blue
        }
        
        color = color_map.get(status, '#808080')
        
        # Create message
        if status == 'success':
            title = "✅ RoyFin AI Model Training - SUCCESS"
            message = f"Model training completed successfully for tickers: {tickers}"
        elif status == 'failure':
            title = "❌ RoyFin AI Model Training - FAILURE"
            message = f"Model training failed for tickers: {tickers}"
        else:
            title = f"🔔 RoyFin AI Model Training - {status.upper()}"
            message = f"Model training status update for tickers: {tickers}"
        
        # Build Slack payload
        payload = {
            "username": "RoyFin AI Bot",
            "icon_emoji": ":chart_with_upwards_trend:",
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "text": message,
                    "fields": [
                        {
                            "title": "Workflow Run ID",
                            "value": workflow_run_id,
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        },
                        {
                            "title": "Tickers",
                            "value": tickers,
                            "short": False
                        }
                    ],
                    "footer": "RoyFin AI",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }
        
        # Add error details if provided
        if error_details and status == 'failure':
            try:
                error_data = json.loads(error_details)
                error_summary = []
                
                for job_name, job_result in error_data.items():
                    if isinstance(job_result, dict) and job_result.get('result') == 'failure':
                        error_summary.append(f"• {job_name}: Failed")
                
                if error_summary:
                    payload["attachments"][0]["fields"].append({
                        "title": "Failed Jobs",
                        "value": "\n".join(error_summary),
                        "short": False
                    })
                    
            except Exception as e:
                logger.warning(f"Could not parse error details: {e}")
                payload["attachments"][0]["fields"].append({
                    "title": "Error Details",
                    "value": str(error_details)[:500] + "..." if len(str(error_details)) > 500 else str(error_details),
                    "short": False
                })
        
        return payload
    
    def send_slack_notification(self, payload: Dict) -> bool:
        """Send notification to Slack"""
        if not self.webhook_url:
            logger.warning("No webhook URL provided, skipping Slack notification")
            return False
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✓ Slack notification sent successfully")
                return True
            else:
                logger.error(f"✗ Slack notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Error sending Slack notification: {e}")
            return False
    
    def send_email_notification(self, status: str, workflow_run_id: str, 
                              tickers: str, email_config: Optional[Dict] = None) -> bool:
        """Send email notification (placeholder - would need SMTP config)"""
        logger.info("Email notifications not implemented yet")
        return False
    
    def send_notification(self, status: str, workflow_run_id: str, 
                         tickers: str, error_details: Optional[str] = None) -> bool:
        """Send notification via available channels"""
        
        success = False
        
        # Send Slack notification
        if self.webhook_url:
            payload = self.create_slack_message(status, workflow_run_id, tickers, error_details)
            if self.send_slack_notification(payload):
                success = True
        
        # Could add other notification methods here (email, Discord, etc.)
        
        return success

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Send workflow notifications")
    
    parser.add_argument(
        '--status',
        choices=['success', 'failure', 'warning', 'info'],
        required=True,
        help='Notification status'
    )
    
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
        '--webhook-url',
        type=str,
        help='Slack webhook URL'
    )
    
    parser.add_argument(
        '--error-details',
        type=str,
        help='Error details for failure notifications'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Sending workflow notification")
    logger.info(f"Status: {args.status}")
    logger.info(f"Workflow Run ID: {args.workflow_run_id}")
    logger.info(f"Tickers: {args.tickers}")
    
    # Initialize notification sender
    sender = NotificationSender(args.webhook_url)
    
    # Send notification
    success = sender.send_notification(
        status=args.status,
        workflow_run_id=args.workflow_run_id,
        tickers=args.tickers,
        error_details=args.error_details
    )
    
    if success:
        logger.info("✓ Notification sent successfully")
        sys.exit(0)
    else:
        logger.warning("✗ No notifications were sent successfully")
        sys.exit(1)

if __name__ == "__main__":
    main()