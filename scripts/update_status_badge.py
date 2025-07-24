#!/usr/bin/env python3
"""
Status Badge Update Script for RoyFin AI
Updates workflow status badges
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatusBadgeUpdater:
    """Update workflow status badges"""
    
    def __init__(self, output_dir: str = "./badges"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_badge_data(self, status: str, workflow_run_id: str) -> Dict:
        """Create badge data"""
        
        # Color mapping
        color_map = {
            'success': 'brightgreen',
            'failure': 'red',
            'running': 'yellow',
            'unknown': 'lightgrey'
        }
        
        # Label mapping
        label_map = {
            'success': 'passing',
            'failure': 'failing',
            'running': 'running',
            'unknown': 'unknown'
        }
        
        color = color_map.get(status, 'lightgrey')
        label = label_map.get(status, 'unknown')
        
        badge_data = {
            'schemaVersion': 1,
            'label': 'model training',
            'message': label,
            'color': color,
            'cacheSeconds': 300,
            'namedLogo': 'github',
            'logoColor': 'white'
        }
        
        return badge_data
    
    def create_shields_url(self, status: str) -> str:
        """Create shields.io badge URL"""
        
        color_map = {
            'success': 'brightgreen',
            'failure': 'red', 
            'running': 'yellow',
            'unknown': 'lightgrey'
        }
        
        label_map = {
            'success': 'passing',
            'failure': 'failing',
            'running': 'running',
            'unknown': 'unknown'
        }
        
        color = color_map.get(status, 'lightgrey')
        label = label_map.get(status, 'unknown')
        
        url = f"https://img.shields.io/badge/model%20training-{label}-{color}?style=flat-square&logo=github"
        
        return url
    
    def create_status_json(self, status: str, workflow_run_id: str) -> Dict:
        """Create comprehensive status JSON"""
        
        status_data = {
            'status': status,
            'workflow_run_id': workflow_run_id,
            'last_updated': datetime.now().isoformat(),
            'last_updated_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'badge_url': self.create_shields_url(status),
            'workflow_url': f"https://github.com/Rajdeep183/royfin_ai/actions/runs/{workflow_run_id}",
            'repository_url': "https://github.com/Rajdeep183/royfin_ai"
        }
        
        return status_data
    
    def create_readme_badge(self, status: str, workflow_run_id: str) -> str:
        """Create README badge markdown"""
        
        badge_url = self.create_shields_url(status)
        workflow_url = f"https://github.com/Rajdeep183/royfin_ai/actions/runs/{workflow_run_id}"
        
        badge_markdown = f"[![Model Training]({badge_url})]({workflow_url})"
        
        return badge_markdown
    
    def save_badge_files(self, status: str, workflow_run_id: str) -> List[str]:
        """Save badge files"""
        
        saved_files = []
        
        # Save badge data JSON
        badge_data = self.create_badge_data(status, workflow_run_id)
        badge_file = self.output_dir / "training_badge.json"
        with open(badge_file, 'w') as f:
            json.dump(badge_data, f, indent=2)
        saved_files.append(str(badge_file))
        
        # Save status JSON
        status_data = self.create_status_json(status, workflow_run_id)
        status_file = self.output_dir / "training_status.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        saved_files.append(str(status_file))
        
        # Save README badge markdown
        badge_markdown = self.create_readme_badge(status, workflow_run_id)
        readme_file = self.output_dir / "readme_badge.md"
        with open(readme_file, 'w') as f:
            f.write(badge_markdown)
        saved_files.append(str(readme_file))
        
        # Save badge URL
        badge_url = self.create_shields_url(status)
        url_file = self.output_dir / "badge_url.txt"
        with open(url_file, 'w') as f:
            f.write(badge_url)
        saved_files.append(str(url_file))
        
        logger.info(f"Saved {len(saved_files)} badge files")
        return saved_files

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Update status badges")
    
    parser.add_argument(
        '--status',
        type=str,
        choices=['success', 'failure', 'running', 'unknown'],
        required=True,
        help='Workflow status'
    )
    
    parser.add_argument(
        '--workflow-run-id',
        type=str,
        required=True,
        help='GitHub Actions workflow run ID'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./badges',
        help='Output directory for badge files'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    logger.info("Updating status badges")
    logger.info(f"Status: {args.status}")
    logger.info(f"Workflow Run ID: {args.workflow_run_id}")
    
    try:
        # Initialize badge updater
        updater = StatusBadgeUpdater(args.output_dir)
        
        # Save badge files
        saved_files = updater.save_badge_files(args.status, args.workflow_run_id)
        
        logger.info("="*50)
        logger.info("STATUS BADGE UPDATED")
        logger.info("="*50)
        logger.info(f"Status: {args.status.upper()}")
        logger.info(f"Files saved: {len(saved_files)}")
        
        for file_path in saved_files:
            logger.info(f"  - {file_path}")
        
        # Print badge URL for easy access
        badge_url = updater.create_shields_url(args.status)
        logger.info(f"Badge URL: {badge_url}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error updating status badge: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()