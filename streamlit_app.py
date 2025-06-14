import streamlit as st

# MUST be the first Streamlit command - Configure page before anything else
st.set_page_config(
    page_title="RoyFin AI - Next-Generation Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import sys
import os
from datetime import datetime, timedelta
import asyncio
import yfinance as yf

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cloud', 'functions', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# Initialize global variables before any other Streamlit commands
ADVANCED_MODEL_AVAILABLE = False
API_BASE_URL = "http://localhost:3001/api"  # Your Next.js API

# Try to import your advanced ML model (do this quietly)
try:
    from cloud.functions.lib.model.stock_lstm import NextGenerationStockPredictor, ModelConfig
    ADVANCED_MODEL_AVAILABLE = True
except ImportError:
    # Don't show warning immediately, we'll handle this in the UI section
    pass

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc, #f1f5f9);
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🤖 RoyFin AI - Next-Generation Stock Prediction</h1>', unsafe_allow_html=True)

# Show model availability info in a clean way
if not ADVANCED_MODEL_AVAILABLE:
    st.info("ℹ️ Advanced ML models not found locally. Using API-based predictions and statistical fallbacks.")

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Prediction Settings")
    
    # Stock ticker input
    ticker = st.text_input(
        "Enter Stock Ticker",
        value="AAPL",
        help="Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL, RELIANCE.NS for Indian stocks)"
    ).upper()
    
    # Time range selection
    time_range = st.selectbox(
        "Historical Data Range",
        options=["6m", "1y", "2y", "3y"],
        index=1,
        help="Select how much historical data to use for training"
    )
    
    # Prediction parameters
    st.subheader("🎯 Prediction Parameters")
    
    prediction_days = st.slider(
        "Days to Predict",
        min_value=1,
        max_value=90,
        value=30,
        help="Number of future days to predict"
    )
    
    aggressiveness = st.slider(
        "Prediction Aggressiveness",
        min_value=0,
        max_value=100,
        value=50,
        help="Higher values = more volatile predictions, Lower values = more conservative"
    )
    
    # Advanced model settings
    if st.sidebar.checkbox("🚀 Use Ultra-Advanced ML API", value=True):
        st.subheader("🧠 Ultra-Advanced ML Settings")
        
        model_type = st.selectbox(
            "AI Model Type",
            options=["ensemble", "transformer", "quantum", "lstm"],
            index=0,
            help="Choose the AI model architecture"
        )
        
        use_ensemble = st.checkbox("Multi-Model Ensemble", value=True)
        ensemble_size = st.slider("Ensemble Size", 3, 10, 7) if use_ensemble else 1
        
        use_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
        use_macro = st.checkbox("Include Macro Factors", value=True)
        use_uncertainty = st.checkbox("Show Uncertainty Bands", value=True)
        
        confidence_level = st.slider("Confidence Level", 0.1, 0.95, 0.85, 0.05)
    else:
        model_type = "basic"
        use_sentiment = False
        use_macro = False

# Main prediction button
predict_button = st.button("🚀 Generate Predictions", type="primary", use_container_width=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker, time_range):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Calculate period
        end_date = datetime.now()
        if time_range == "6m":
            start_date = end_date - timedelta(days=180)
        elif time_range == "1y":
            start_date = end_date - timedelta(days=365)
        elif time_range == "2y":
            start_date = end_date - timedelta(days=730)
        else:  # 3y
            start_date = end_date - timedelta(days=1095)
        
        # Fetch data
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None, f"No data found for ticker {ticker}"
        
        # Get company info
        try:
            info = stock.info
            company_name = info.get('longName', ticker)
        except:
            company_name = ticker
        
        return {
            'historical': data.reset_index().to_dict('records'),
            'company_name': company_name,
            'ticker': ticker
        }, None
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def create_advanced_predictions(stock_data, days, aggressiveness):
    """Create predictions using advanced ML model if available"""
    if not ADVANCED_MODEL_AVAILABLE:
        return create_simple_predictions(stock_data, days, aggressiveness)
    
    try:
        # Configure the advanced model
        config = ModelConfig(
            hidden_dims=[128, 256, 128] if model_complexity == "Heavy" else [64, 128, 64],
            epochs=100 if model_complexity == "Heavy" else 50,
            num_ensemble_models=ensemble_size,
            sequence_length=60
        )
        
        # Initialize predictor
        predictor = NextGenerationStockPredictor(ticker, config)
        
        # Convert stock data format
        historical_data = []
        for item in stock_data['historical']:
            historical_data.append({
                'date': item['Date'].strftime('%Y-%m-%d') if hasattr(item['Date'], 'strftime') else str(item['Date'])[:10],
                'open': float(item['Open']),
                'high': float(item['High']),
                'low': float(item['Low']),
                'close': float(item['Close']),
                'volume': int(item['Volume'])
            })
        
        # Create features and predictions
        features = predictor.create_next_gen_features({'historical': historical_data})
        X, y = predictor.prepare_sequences()
        
        # Train and predict
        predictor.train_with_modern_techniques(X, y)
        predictions = predictor.predict_with_uncertainty(days_ahead=days)
        
        return predictions, None
        
    except Exception as e:
        st.warning(f"Advanced model failed, falling back to simple predictions: {str(e)}")
        return create_simple_predictions(stock_data, days, aggressiveness)

def create_simple_predictions(stock_data, days, aggressiveness):
    """Create simple predictions using basic statistical methods"""
    try:
        df = pd.DataFrame(stock_data['historical'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Calculate basic statistics
        prices = df['Close'].values
        returns = np.diff(prices) / prices[:-1]
        
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Apply aggressiveness factor
        aggressiveness_factor = (aggressiveness - 50) / 100
        trend_multiplier = 1 + (aggressiveness_factor * 0.3)
        volatility_multiplier = 1 + abs(aggressiveness_factor) * 0.2
        
        # Generate predictions
        predictions = []
        current_price = prices[-1]
        current_date = df['Date'].iloc[-1]
        
        for i in range(1, days + 1):
            # Add business days
            next_date = current_date + timedelta(days=i)
            while next_date.weekday() >= 5:  # Skip weekends
                next_date += timedelta(days=1)
            
            # Calculate predicted price with trend and randomness
            daily_return = avg_return * trend_multiplier + np.random.normal(0, volatility * volatility_multiplier)
            predicted_price = current_price * (1 + daily_return * i * 0.1)
            
            # Add some mean reversion
            mean_price = np.mean(prices[-20:])  # 20-day average
            reversion = (mean_price - predicted_price) * 0.05
            predicted_price += reversion
            
            predictions.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'predicted': max(0.01, predicted_price),  # Ensure positive price
                'confidence': max(0.1, 1 - (i / days) * 0.5)  # Decreasing confidence
            })
        
        return predictions, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def plot_stock_chart(stock_data, predictions=None):
    """Create an interactive stock chart with predictions"""
    df = pd.DataFrame(stock_data['historical'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_width=[0.7, 0.3],
        subplot_titles=('Stock Price & Predictions', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )
    
    # Add predictions if available
    if predictions:
        pred_dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in predictions]
        pred_prices = [p['predicted'] for p in predictions]
        
        # Connect last historical price to first prediction
        connection_date = df['Date'].iloc[-1]
        connection_price = df['Close'].iloc[-1]
        
        pred_dates.insert(0, connection_date)
        pred_prices.insert(0, connection_price)
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_prices,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='#3b82f6', width=3, dash='dash'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add confidence bands if using uncertainty
        if use_uncertainty and 'confidence' in predictions[0]:
            upper_band = [p['predicted'] * (1 + (1 - p['confidence']) * 0.1) for p in predictions[1:]]
            lower_band = [p['predicted'] * (1 - (1 - p['confidence']) * 0.1) for p in predictions[1:]]
            
            fig.add_trace(
                go.Scatter(
                    x=pred_dates[1:] + pred_dates[1:][::-1],
                    y=upper_band + lower_band[::-1],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Add volume chart
    colors = ['#10b981' if close >= open else '#ef4444' 
              for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{stock_data['company_name']} ({stock_data['ticker']}) - Stock Analysis & Predictions",
        height=700,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Main application logic
if predict_button and ticker:
    with st.spinner(f"🔍 Fetching data for {ticker}..."):
        stock_data, error = fetch_stock_data(ticker, time_range)
    
    if error:
        st.error(f"❌ {error}")
    else:
        # Display company info
        col1, col2, col3 = st.columns(3)
        
        df = pd.DataFrame(stock_data['historical'])
        latest_price = df['Close'].iloc[-1]
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
        price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stock_data['company_name']}</h3>
                <h2>${latest_price:.2f}</h2>
                <p style="color: {'green' if price_change >= 0 else 'red'}">
                    {'+' if price_change >= 0 else ''}{price_change:.2f} ({price_change_pct:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            st.metric("52W High", f"${df['High'].max():.2f}")
        
        with col3:
            st.metric("52W Low", f"${df['Low'].min():.2f}")
            st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
        
        # Generate predictions
        with st.spinner("🤖 Generating AI predictions..."):
            if model_type != "basic":
                predictions, pred_error = create_ultra_advanced_predictions(
                    stock_data, prediction_days, model_type, 
                    confidence_level, use_sentiment, use_macro, ensemble_size
                )
            elif ADVANCED_MODEL_AVAILABLE:
                predictions, pred_error = create_advanced_predictions(stock_data, prediction_days, aggressiveness)
            else:
                predictions, pred_error = create_simple_predictions(stock_data, prediction_days, aggressiveness)
        
        if pred_error:
            st.error(f"❌ Prediction Error: {pred_error}")
        else:
            # Display predictions
            st.subheader("📈 Stock Chart & Predictions")
            chart = plot_stock_chart(stock_data, predictions)
            st.plotly_chart(chart, use_container_width=True)
            
            # Prediction summary
            if predictions:
                st.subheader("🎯 AI Prediction Summary")
                
                # Add model information
                col_model1, col_model2 = st.columns(2)
                with col_model1:
                    st.info(f"🤖 **Model**: {model_type.title() if model_type != 'basic' else 'Statistical'}")
                    if model_type != "basic":
                        st.info(f"🧠 **Features**: Sentiment: {use_sentiment}, Macro: {use_macro}")
                
                with col_model2:
                    if model_type == "ensemble":
                        st.info(f"📊 **Ensemble Size**: {ensemble_size} models")
                    st.info(f"🎯 **Confidence Level**: {confidence_level:.0%}")
                
                pred_df = pd.DataFrame(predictions)
                first_pred = predictions[0]['predicted']
                last_pred = predictions[-1]['predicted']
                total_change = ((last_pred - latest_price) / latest_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        f"Price in {prediction_days} days",
                        f"${last_pred:.2f}",
                        f"{total_change:+.2f}%"
                    )
                
                with col2:
                    avg_pred = pred_df['predicted'].mean()
                    avg_change = ((avg_pred - latest_price) / latest_price) * 100
                    st.metric(
                        "Average Predicted Price",
                        f"${avg_pred:.2f}",
                        f"{avg_change:+.2f}%"
                    )
                
                with col3:
                    max_pred = pred_df['predicted'].max()
                    max_change = ((max_pred - latest_price) / latest_price) * 100
                    st.metric(
                        "Highest Prediction",
                        f"${max_pred:.2f}",
                        f"{max_change:+.2f}%"
                    )
                
                with col4:
                    min_pred = pred_df['predicted'].min()
                    min_change = ((min_pred - latest_price) / latest_price) * 100
                    st.metric(
                        "Lowest Prediction",
                        f"${min_pred:.2f}",
                        f"{min_change:+.2f}%"
                    )
                
                # Detailed predictions table
                with st.expander("📊 Detailed Predictions"):
                    display_df = pred_df.copy()
                    display_df['Date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
                    display_df['Predicted Price'] = display_df['predicted'].apply(lambda x: f"${x:.2f}")
                    display_df['Change from Current'] = display_df['predicted'].apply(
                        lambda x: f"{((x - latest_price) / latest_price) * 100:+.2f}%"
                    )
                    
                    if 'confidence' in display_df.columns:
                        display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
                    
                    st.dataframe(
                        display_df[['Date', 'Predicted Price', 'Change from Current'] + 
                                  (['Confidence'] if 'confidence' in display_df.columns else [])],
                        use_container_width=True
                    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 2rem;">
    <p>🤖 <strong>RoyFin AI</strong> - Next-Generation Stock Prediction Platform</p>
    <p>Built with advanced machine learning models and real-time market data</p>
    <p>Made with ❤️ by Rajdeep Roy</p>
</div>
""", unsafe_allow_html=True)