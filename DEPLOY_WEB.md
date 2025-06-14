# 🚀 RoyFin AI - Web Deployment Guide

Since CLI installations have permission issues, use these **web-based deployment methods** instead:

## 🚂 Option 1: Railway (RECOMMENDED - Most Reliable)

### Steps:
1. Go to **https://railway.app**
2. Sign up/Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Connect your GitHub account and select your `stock_pred` repository
5. Railway will auto-detect your Next.js app and deploy it
6. Your app will be live at: `https://yourapp.railway.app`

### Configuration:
- **Root Directory**: `/client` (set this in Railway dashboard)
- **Build Command**: `npm run build`
- **Start Command**: `npm start`
- **Environment Variables**: Set `NODE_ENV=production`

---

## 🌐 Option 2: Netlify (Great for Static Sites)

### Steps:
1. Go to **https://app.netlify.com**
2. Sign up/Login with GitHub
3. Click "Add new site" → "Import an existing project"
4. Choose GitHub and select your `stock_pred` repository
5. Set build settings:
   - **Base directory**: `client`
   - **Build command**: `npm run build`
   - **Publish directory**: `client/.next`

---

## 🟣 Option 3: Heroku (Traditional Hosting)

### Steps:
1. Go to **https://dashboard.heroku.com**
2. Create new app: `royfin-ai-yourname`
3. Connect to GitHub repository
4. Enable automatic deploys from `main` branch
5. Add this `Procfile` to your root directory:
   ```
   web: cd client && npm start
   ```

---

## 🌊 Option 4: DigitalOcean App Platform

### Steps:
1. Go to **https://cloud.digitalocean.com/apps**
2. Create new app from GitHub
3. Select your repository
4. Configure:
   - **Source Directory**: `/client`
   - **Build Command**: `npm install && npm run build`
   - **Run Command**: `npm start`

---

## ⚡ Option 5: Render (Simple & Reliable)

### Steps:
1. Go to **https://render.com**
2. Sign up with GitHub
3. New Web Service → Connect repository
4. Settings:
   - **Root Directory**: `client`
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npm start`

---

## 🎯 QUICKEST DEPLOYMENT (Railway)

**Railway is the fastest and most reliable option:**

1. **Go to**: https://railway.app
2. **Click**: "Deploy Now" 
3. **Connect**: Your GitHub repo `stock_pred`
4. **Set Root Directory**: `client`
5. **Deploy**: Railway handles everything automatically!

Your app will be live in 2-3 minutes at a URL like:
`https://royfin-ai-production.up.railway.app`

---

## 🔧 Environment Variables (Set on chosen platform)

```bash
NODE_ENV=production
NEXT_PUBLIC_API_URL=https://your-deployed-url.com
```

---

## 📊 Expected Results

Once deployed, your **RoyFin AI** platform will have:

- ✅ **Landing Page**: Professional gradient design with AI branding
- ✅ **Prediction Dashboard**: Real-time stock analysis with charts
- ✅ **AI Models**: Ultra-advanced ML predictions (transformer, quantum, LSTM, ensemble)
- ✅ **APIs**: All 5 API endpoints working (`/api/health`, `/api/stock`, `/api/predict`, etc.)
- ✅ **Interactive Charts**: Stock data visualization with predictions
- ✅ **Mobile Responsive**: Works on all devices

---

## 🚀 DEPLOY NOW!

**I recommend starting with Railway** - just go to railway.app and deploy your GitHub repo. It's the most reliable platform for Next.js applications and will have your RoyFin AI system live in minutes!

Let me know which platform you choose and I'll help you configure it! 🎯