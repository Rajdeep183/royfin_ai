# 🚀 RoyFin AI - Next-Generation Stock Prediction Platform

RoyFin AI is a full-stack, production-ready platform for intelligent stock market forecasting. It combines advanced machine learning (LSTM, Transformer, Quantum-inspired, Ensemble), real-time data, and a modern web UI to deliver actionable insights for investors and analysts.

---
## Live Demo

https://royfinai.vercel.app/predict

---
## 🌟 Features

- **AI-Powered Predictions:** LSTM, Transformer, Quantum-inspired, and Ensemble models
- **Real-Time Data:** Live price updates, earnings, and market sentiment
- **Interactive Dashboard:** Modern Next.js/React UI with charts and analytics
- **Streamlit App:** Python dashboard for rapid prototyping and analysis
- **Cloud Functions:** Scalable model training and prediction APIs
- **Multi-Platform Deployment:** Railway, Netlify, Heroku, DigitalOcean, Render
- **Mobile Responsive:** Works on all devices

---

## 🗂️ Project Structure

```
.
├── client/           # Next.js frontend (UI, API routes, components)
├── cloud/            # Cloud functions (model training, prediction APIs)
├── lib/              # Advanced ML/AI modules (TypeScript)
├── model/            # Python ML models, training scripts, requirements
├── streamlit_app.py  # Streamlit dashboard (Python)
├── manage_system.sh  # Full system management script
├── deploy.sh         # Multi-platform deployment script
├── requirements_streamlit.txt  # Python requirements for Streamlit
├── setup.py          # Python package setup
├── README.md         # (You are here)
└── ...               # Configs, docs, and deployment files
```

---

## ⚡ Quick Start

### 1. **Install Dependencies**

```sh
# Frontend (Next.js)
cd client
npm install
cd ..

# Python (ML/Streamlit)
pip install -r requirements_streamlit.txt
```

### 2. **Run Locally**

```sh
# Start Next.js API & UI
cd client
npm run dev
# Visit http://localhost:3001

# OR Start Streamlit Dashboard
streamlit run streamlit_app.py
# Visit http://localhost:8501
```

### 3. **Use System Manager**

```sh
# Full setup, training, and deployment
./manage_system.sh

# Available options:
# 1. Setup & Install Dependencies
# 2. Run Development Servers
# 3. Run Streamlit Dashboard  
# 4. Train & Evaluate Models
```

---

## 🧠 AI & ML Models

- **Ultra-Advanced ML:** See [`lib/ultra-advanced-ml.ts`](lib/ultra-advanced-ml.ts)
- **Python LSTM/Ensemble:** See [`model/`](model/)
- **Cloud Training:** See [`cloud/functions/train_ticker/main.py`](cloud/functions/train_ticker/main.py)

---

## 📊 Real-Time Data

- Yahoo Finance, Alpha Vantage, Finnhub, Polygon.io integrations
- WebSocket support for live updates
- See [REALTIME_SETUP.md](REALTIME_SETUP.md) for API key setup

---

## 🛠️ System Management

- Use [`manage_system.sh`](manage_system.sh) for:
  - Full setup & deployment
  - Running Streamlit dashboard
  - Testing APIs
  - Training models

---

Made with ❤️ by Rajdeep Roy
