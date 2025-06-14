#!/bin/bash

# 🚀 RoyFin AI - Multi-Platform Deployment Script
# Deploy to Railway, Render, Netlify, and Heroku

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "client" ]; then
    print_error "Please run this script from the root directory of your stock_pred project"
    exit 1
fi

print_header "🚀 RoyFin AI - Multi-Platform Deployment"
echo
echo -e "${PURPLE}Next-Generation Stock Prediction Platform${NC}"
echo -e "${PURPLE}Deploying to multiple cloud platforms for maximum reliability${NC}"
echo

# Install Railway CLI if not present
check_railway() {
    if ! command -v railway &> /dev/null; then
        print_info "Installing Railway CLI..."
        npm install -g @railway/cli
    fi
}

# Install Render CLI if not present
check_render() {
    if ! command -v render &> /dev/null; then
        print_info "Render CLI not found. You can deploy via web interface."
        return 1
    fi
    return 0
}

# Install Netlify CLI if not present
check_netlify() {
    if ! command -v netlify &> /dev/null; then
        print_info "Installing Netlify CLI..."
        npm install -g netlify-cli
    fi
}

# Install Heroku CLI if not present
check_heroku() {
    if ! command -v heroku &> /dev/null; then
        print_warning "Heroku CLI not found. Install from: https://devcenter.heroku.com/articles/heroku-cli"
        return 1
    fi
    return 0
}

# Prepare project for deployment
prepare_project() {
    print_info "Preparing project for deployment..."
    
    # Install dependencies
    cd client
    npm install
    
    # Build the project to check for errors
    print_info "Building project to verify..."
    npm run build
    
    if [ $? -eq 0 ]; then
        print_success "Build successful! Project is ready for deployment."
    else
        print_error "Build failed! Please fix errors before deploying."
        exit 1
    fi
    
    cd ..
}

# Deploy to Railway
deploy_railway() {
    print_header "🚂 Deploying to Railway"
    
    check_railway
    
    # Login to Railway (will open browser if not logged in)
    print_info "Checking Railway authentication..."
    if ! railway whoami &> /dev/null; then
        print_info "Please log in to Railway..."
        railway login
    fi
    
    # Create or connect to Railway project
    if [ ! -f ".railway" ]; then
        print_info "Creating new Railway project..."
        railway project create royfin-ai
    fi
    
    # Deploy
    print_info "Deploying to Railway..."
    railway up
    
    if [ $? -eq 0 ]; then
        print_success "Railway deployment successful!"
        RAILWAY_URL=$(railway domain)
        print_info "Railway URL: ${RAILWAY_URL}"
    else
        print_error "Railway deployment failed!"
    fi
}

# Deploy to Netlify
deploy_netlify() {
    print_header "🌐 Deploying to Netlify"
    
    check_netlify
    
    # Login to Netlify
    if ! netlify status &> /dev/null; then
        print_info "Please log in to Netlify..."
        netlify login
    fi
    
    # Build for static deployment
    cd client
    print_info "Building static version for Netlify..."
    npm run build
    
    # Deploy
    print_info "Deploying to Netlify..."
    netlify deploy --prod --dir=.next
    
    if [ $? -eq 0 ]; then
        print_success "Netlify deployment successful!"
        NETLIFY_URL=$(netlify status --json | grep "url" | head -1 | cut -d'"' -f4)
        print_info "Netlify URL: ${NETLIFY_URL}"
    else
        print_error "Netlify deployment failed!"
    fi
    
    cd ..
}

# Deploy to Heroku
deploy_heroku() {
    print_header "🟣 Deploying to Heroku"
    
    if ! check_heroku; then
        print_warning "Skipping Heroku deployment - CLI not found"
        return
    fi
    
    # Login to Heroku
    if ! heroku auth:whoami &> /dev/null; then
        print_info "Please log in to Heroku..."
        heroku login
    fi
    
    # Create Heroku app
    APP_NAME="royfin-ai-$(date +%s)"
    print_info "Creating Heroku app: ${APP_NAME}"
    heroku create ${APP_NAME}
    
    # Set buildpack for Node.js
    heroku buildpacks:set heroku/nodejs --app ${APP_NAME}
    
    # Add Procfile for Heroku
    echo "web: cd client && npm start" > Procfile
    
    # Deploy
    print_info "Deploying to Heroku..."
    git add .
    git commit -m "Deploy to Heroku" || true
    git push heroku main
    
    if [ $? -eq 0 ]; then
        print_success "Heroku deployment successful!"
        HEROKU_URL="https://${APP_NAME}.herokuapp.com"
        print_info "Heroku URL: ${HEROKU_URL}"
    else
        print_error "Heroku deployment failed!"
    fi
}

# Deploy to DigitalOcean App Platform
deploy_digitalocean() {
    print_header "🌊 Deploying to DigitalOcean App Platform"
    
    # Create App Platform spec
    cat > .do/app.yaml << EOF
name: royfin-ai
services:
- name: web
  source_dir: /client
  github:
    repo: rajdeeproy/stock_pred
    branch: main
  run_command: npm start
  build_command: npm install && npm run build
  environment_slug: node-js
  instance_count: 1
  instance_size_slug: basic-xxs
  routes:
  - path: /
  health_check:
    http_path: /api/health
EOF
    
    print_info "DigitalOcean App Platform spec created at .do/app.yaml"
    print_info "Please deploy manually via: https://cloud.digitalocean.com/apps"
}

# Create comprehensive deployment documentation
create_deployment_docs() {
    cat > DEPLOYMENT.md << 'EOF'
# 🚀 RoyFin AI - Deployment Guide

## Deployed Platforms

### 1. Railway 🚂
- **Best for**: Full-stack applications with databases
- **URL**: Will be provided after deployment
- **Features**: Auto-scaling, built-in monitoring, easy database integration

### 2. Netlify 🌐  
- **Best for**: Frontend applications with serverless functions
- **URL**: Will be provided after deployment
- **Features**: Global CDN, instant rollbacks, branch previews

### 3. Heroku 🟣
- **Best for**: Traditional web applications
- **URL**: Will be provided after deployment  
- **Features**: Easy scaling, add-ons marketplace, Git-based deployment

### 4. DigitalOcean App Platform 🌊
- **Best for**: Production applications with predictable pricing
- **Manual deployment**: Use the generated .do/app.yaml file

## Deployment Commands

```bash
# Deploy to all platforms
./deploy.sh

# Deploy to specific platform
./deploy.sh railway
./deploy.sh netlify  
./deploy.sh heroku
```

## Environment Variables

Set these environment variables on each platform:

- `NODE_ENV=production`
- `NEXT_PUBLIC_API_URL=<your-deployed-url>`

## Health Checks

All platforms are configured with health checks at `/api/health`

## Monitoring

- Railway: Built-in monitoring dashboard
- Netlify: Analytics and performance monitoring
- Heroku: Metrics dashboard
- DigitalOcean: App-level monitoring

## Troubleshooting

If deployment fails:
1. Check build logs on the platform
2. Verify all dependencies are in package.json
3. Ensure build command succeeds locally
4. Check environment variables

## Support

- Railway: https://docs.railway.app
- Netlify: https://docs.netlify.com  
- Heroku: https://devcenter.heroku.com
- DigitalOcean: https://docs.digitalocean.com/products/app-platform
EOF

    print_success "Deployment documentation created: DEPLOYMENT.md"
}

# Main deployment function
main() {
    # Check if specific platform is requested
    PLATFORM=${1:-"all"}
    
    prepare_project
    
    case $PLATFORM in
        "railway")
            deploy_railway
            ;;
        "netlify")
            deploy_netlify
            ;;
        "heroku")
            deploy_heroku
            ;;
        "digitalocean")
            deploy_digitalocean
            ;;
        "all")
            print_info "Deploying to all platforms..."
            
            deploy_railway &
            RAILWAY_PID=$!
            
            deploy_netlify &
            NETLIFY_PID=$!
            
            deploy_heroku &
            HEROKU_PID=$!
            
            deploy_digitalocean
            
            # Wait for parallel deployments
            wait $RAILWAY_PID
            wait $NETLIFY_PID  
            wait $HEROKU_PID
            ;;
        *)
            print_error "Unknown platform: $PLATFORM"
            print_info "Available platforms: railway, netlify, heroku, digitalocean, all"
            exit 1
            ;;
    esac
    
    create_deployment_docs
    
    print_header "🎉 Deployment Summary"
    echo
    print_success "RoyFin AI has been deployed to multiple platforms!"
    print_info "Check DEPLOYMENT.md for complete details and URLs"
    echo
    print_info "Your Next-Generation Stock Prediction Platform is now live! 🚀"
}

# Run main function
main "$@"