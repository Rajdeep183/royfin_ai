import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta
import warnings
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress warnings
warnings.filterwarnings('ignore')

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for enhanced feature learning"""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection and residual connection
        output = self.w_o(context)
        return self.layer_norm(x + self.dropout(output))

class AdvancedStockLSTMModel(nn.Module):
    """Advanced LSTM model with attention, residual connections, and multi-task learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2, use_attention: bool = True,
                 use_residual: bool = True):
        super(AdvancedStockLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        # LSTM layers with layer normalization
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Multi-head attention
        if use_attention:
            self.attention = MultiHeadAttention(hidden_dim, n_heads=8, dropout=dropout)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 if use_attention else hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task heads
        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Direction prediction head (up/down classification)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        # Magnitude prediction head (absolute change)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Input projection and normalization
        x = self.input_projection(x)
        x = x.transpose(1, 2)  # For BatchNorm1d
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # Back to original shape
        
        # LSTM processing with residual connections
        lstm_outputs = []
        current_input = x
        
        for i, (lstm_layer, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
            
            lstm_out, _ = lstm_layer(current_input, (h0, c0))
            lstm_out = layer_norm(lstm_out)
            
            # Residual connection
            if self.use_residual and i > 0:
                lstm_out = lstm_out + current_input
            
            lstm_out = self.dropout(lstm_out)
            lstm_outputs.append(lstm_out)
            current_input = lstm_out
        
        # Take the last LSTM output
        lstm_final = lstm_outputs[-1]
        
        # Apply attention mechanism
        if self.use_attention:
            attention_out = self.attention(lstm_final)
            # Combine LSTM and attention outputs
            combined = torch.cat([lstm_final[:, -1, :], attention_out[:, -1, :]], dim=1)
        else:
            combined = lstm_final[:, -1, :]
        
        # Feature fusion
        features = self.feature_fusion(combined)
        
        # Multi-task predictions
        price_pred = self.price_head(features)
        volatility_pred = torch.abs(self.volatility_head(features))  # Ensure positive volatility
        direction_pred = self.direction_head(features)
        magnitude_pred = torch.abs(self.magnitude_head(features))  # Ensure positive magnitude
        
        return {
            'price': price_pred,
            'volatility': volatility_pred,
            'direction': direction_pred,
            'magnitude': magnitude_pred
        }

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators without TA-Lib dependency"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = close.rolling(window=period).mean()
        df[f'EMA_{period}'] = close.ewm(span=period).mean()
        df[f'Price_SMA_{period}_Ratio'] = close / df[f'SMA_{period}']
        df[f'Price_EMA_{period}_Ratio'] = close / df[f'EMA_{period}']
    
    # Bollinger Bands
    for period in [10, 20, 50]:
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        df[f'BB_Upper_{period}'] = sma + (std * 2)
        df[f'BB_Lower_{period}'] = sma - (std * 2)
        df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / sma
        df[f'BB_Position_{period}'] = (close - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Stochastic Oscillator
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    df['STOCH_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    
    # Williams %R
    df['WILLIAMS_R'] = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    # Rate of Change
    df['ROC_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
    df['ROC_20'] = ((close - close.shift(20)) / close.shift(20)) * 100
    
    # Momentum
    df['MOM_10'] = close - close.shift(10)
    df['MOM_20'] = close - close.shift(20)
    
    # Average True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_SMA_10'] = volume.rolling(window=10).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_10']
    
    # On Balance Volume (simplified)
    df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    return df

class AdvancedStockPredictor:
    """Advanced stock predictor with state-of-the-art ML techniques"""
    
    def __init__(self, ticker: str, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None, seq_length: int = 60, 
                 ensemble_size: int = 7, hidden_dim: int = 256, 
                 num_layers: int = 3, learning_rate: float = 0.001, 
                 batch_size: int = 64, epochs: int = 200, 
                 dropout: float = 0.3, device: Optional[str] = None,
                 use_attention: bool = True, use_residual: bool = True,
                 optimize_hyperparams: bool = False):
        """Initialize the Advanced Stock Predictor"""
        self.ticker = ticker
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.seq_length = seq_length
        self.ensemble_size = ensemble_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.optimize_hyperparams = optimize_hyperparams
        
        # Advanced scalers for better normalization
        self.price_scaler = RobustScaler()
        self.feature_scalers = {}
        self.target_scalers = {}
        
        # Model storage
        self.models = []
        self.best_models = []
        self.input_dim = None
        self.output_dim = 1
        self.df = None
        self.feature_names = []
        self.model_save_path = f"models/{ticker}_advanced_lstm_ensemble.pkl"
        
        # Performance tracking
        self.training_history = []
        self.best_hyperparams = {}
        
        logger.info(f"Initialized AdvancedStockPredictor for {ticker}")
        logger.info(f"Using device: {self.device}")

    def fetch_data(self) -> pd.DataFrame:
        """Fetch comprehensive data from Yahoo Finance"""
        logger.info(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        try:
            # Fetch main data
            self.df = yf.Ticker(self.ticker).history(
                interval="1d", 
                start=self.start_date, 
                end=self.end_date,
                auto_adjust=True,
                prepost=True
            )
            
            if self.df.empty or len(self.df) < 200:
                raise ValueError(f"Insufficient data for ticker {self.ticker}. Need at least 200 days.")
            
            # Fetch additional market data for context
            try:
                spy = yf.Ticker("SPY").history(interval="1d", start=self.start_date, end=self.end_date)
                vix = yf.Ticker("^VIX").history(interval="1d", start=self.start_date, end=self.end_date)
                
                # Align indices
                common_dates = self.df.index.intersection(spy.index).intersection(vix.index)
                if len(common_dates) > 100:  # Only use if we have sufficient overlap
                    self.df = self.df.loc[common_dates]
                    spy = spy.loc[common_dates]
                    vix = vix.loc[common_dates]
                    
                    # Add market context
                    self.df['SPY_Close'] = spy['Close']
                    self.df['VIX_Close'] = vix['Close']
                    logger.info("Added market context data (SPY, VIX)")
                else:
                    logger.warning("Insufficient market data overlap, proceeding without market context")
            except Exception as e:
                logger.warning(f"Could not fetch market data: {e}")
            
            logger.info(f"Data fetched: {len(self.df)} trading days")
            return self.df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def create_advanced_features(self) -> pd.DataFrame:
        """Create comprehensive technical and fundamental features"""
        if self.df is None:
            self.fetch_data()
        
        logger.info("Creating advanced features...")
        df = self.df.copy()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Advanced volatility measures
        for window in [5, 10, 20, 30, 60]:
            df[f'Volatility_{window}d'] = df['Returns'].rolling(window).std() * np.sqrt(252)
            df[f'RealizedVol_{window}d'] = df['Log_Returns'].rolling(window).std() * np.sqrt(252)
            df[f'GARCH_Vol_{window}d'] = df['Returns'].ewm(span=window).std() * np.sqrt(252)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Market context features (if available)
        if 'SPY_Close' in df.columns:
            df['SPY_Returns'] = df['SPY_Close'].pct_change()
            df['Beta_SPY'] = df['Returns'].rolling(60).cov(df['SPY_Returns']) / df['SPY_Returns'].rolling(60).var()
            df['Correlation_SPY'] = df['Returns'].rolling(60).corr(df['SPY_Returns'])
        
        if 'VIX_Close' in df.columns:
            df['VIX_Change'] = df['VIX_Close'].pct_change()
        
        # Cyclical and seasonal features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day
        
        # Encode cyclical features
        df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Target variables (multi-task)
        df['Next_Return'] = df['Returns'].shift(-1)
        df['Next_Price_Change'] = df['Price_Change'].shift(-1)
        df['Next_Volatility'] = df['Volatility_20d'].shift(-1)
        df['Next_Direction'] = (df['Next_Return'] > 0).astype(int)  # 1 for up, 0 for down
        df['Next_Magnitude'] = np.abs(df['Next_Return'])
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        logger.info(f"Created {len(df.columns)} features")
        return df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str, k: int = 50) -> List[str]:
        """Select the best features using statistical tests and correlation analysis"""
        logger.info(f"Selecting top {k} features for {target_col}")
        
        # Exclude target columns and non-numeric columns
        exclude_cols = ['Next_Return', 'Next_Price_Change', 'Next_Volatility', 'Next_Direction', 'Next_Magnitude']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        # Remove highly correlated features
        correlation_matrix = df[feature_cols].corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than 0.95
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        feature_cols = [col for col in feature_cols if col not in high_corr_features]
        
        # Use SelectKBest for feature selection
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        selector.fit(X, y)
        
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features

    def prepare_multi_task_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare targets for multi-task learning"""
        targets = {
            'price': df['Next_Return'].values.reshape(-1, 1),
            'volatility': df['Next_Volatility'].values.reshape(-1, 1),
            'direction': df['Next_Direction'].values.reshape(-1, 1),
            'magnitude': df['Next_Magnitude'].values.reshape(-1, 1)
        }
        
        # Scale targets
        for key, values in targets.items():
            if key == 'direction':
                continue  # Don't scale binary classification targets
            
            scaler = RobustScaler()  # More robust to outliers
            targets[key] = scaler.fit_transform(values)
            self.target_scalers[key] = scaler
        
        return targets

    def create_advanced_sequences(self, X: np.ndarray, targets: Dict[str, np.ndarray]) -> Tuple:
        """Create sequences for multi-task learning"""
        X_seq = []
        y_seq = {key: [] for key in targets.keys()}
        
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            for key in targets.keys():
                y_seq[key].append(targets[key][i + self.seq_length])
        
        X_seq = np.array(X_seq)
        for key in y_seq.keys():
            y_seq[key] = np.array(y_seq[key])
        
        return X_seq, y_seq

    def _calculate_multi_task_loss(self, predictions: Dict[str, torch.Tensor], 
                                 targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate multi-task loss with appropriate weights"""
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        # Price loss (primary task)
        price_loss = mse_loss(predictions['price'], targets['price'])
        
        # Volatility loss
        vol_loss = mse_loss(predictions['volatility'], targets['volatility'])
        
        # Direction loss (classification)
        direction_loss = ce_loss(predictions['direction'], targets['direction'].long().squeeze())
        
        # Magnitude loss
        magnitude_loss = mse_loss(predictions['magnitude'], targets['magnitude'])
        
        # Weighted combination
        total_loss = (
            2.0 * price_loss +      # Primary task
            1.0 * vol_loss +        # Important for risk assessment
            1.5 * direction_loss +  # Direction is crucial
            0.5 * magnitude_loss    # Auxiliary task
        )
        
        return total_loss

    def train_advanced_model(self, train_loader, val_loader, model_idx: int = 0):
        """Train a single advanced model with all optimizations"""
        logger.info(f"Training advanced model {model_idx + 1}")
        
        # Initialize model
        model = AdvancedStockLSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout,
            use_attention=self.use_attention,
            use_residual=self.use_residual
        ).to(self.device)
        
        # Advanced optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Training loop with advanced techniques
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 15
        counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_data in train_loader:
                X_batch = batch_data[0]
                y_batch = {
                    'price': batch_data[1],
                    'volatility': batch_data[2],
                    'direction': batch_data[3],
                    'magnitude': batch_data[4]
                }
                
                optimizer.zero_grad()
                predictions = model(X_batch)
                
                # Calculate multi-task loss
                loss = self._calculate_multi_task_loss(predictions, y_batch)
                
                # Add L2 regularization
                l2_reg = torch.tensor(0., device=self.device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += 1e-6 * l2_reg
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    X_batch = batch_data[0]
                    y_batch = {
                        'price': batch_data[1],
                        'volatility': batch_data[2],
                        'direction': batch_data[3],
                        'magnitude': batch_data[4]
                    }
                    
                    predictions = model(X_batch)
                    loss = self._calculate_multi_task_loss(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Progress logging
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch + 1}/{self.epochs}], "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
            
            # Early stopping with patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    # Restore best model state
                    model.load_state_dict(best_model_state)
                    break
        
        # Store training history
        self.training_history.append({
            'model_idx': model_idx,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        })
        
        return model

    def train_ensemble(self):
        """Train the complete ensemble with advanced techniques"""
        logger.info("Starting advanced ensemble training...")
        
        # Create comprehensive features
        df = self.create_advanced_features()
        
        # Feature selection for each target
        price_features = self.select_best_features(df, 'Next_Return', k=60)
        self.feature_names = price_features
        self.input_dim = len(price_features)
        
        # Prepare features and targets
        X = df[price_features].values
        
        # Advanced feature scaling
        for i, feature in enumerate(price_features):
            scaler = RobustScaler()  # More robust to outliers
            X[:, i] = scaler.fit_transform(X[:, i].reshape(-1, 1)).flatten()
            self.feature_scalers[feature] = scaler
        
        # Prepare multi-task targets
        targets = self.prepare_multi_task_targets(df)
        
        # Create sequences
        X_seq, y_seq = self.create_advanced_sequences(X, targets)
        
        # Train-validation split
        split_idx = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train = {k: v[:split_idx] for k, v in y_seq.items()}
        y_val = {k: v[split_idx:] for k, v in y_seq.items()}
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        y_train_tensors = []
        y_val_tensors = []
        for key in ['price', 'volatility', 'direction', 'magnitude']:
            y_train_tensors.append(torch.FloatTensor(y_train[key]).to(self.device))
            y_val_tensors.append(torch.FloatTensor(y_val[key]).to(self.device))
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, *y_train_tensors)
        val_dataset = TensorDataset(X_val_tensor, *y_val_tensors)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        # Train ensemble
        self.models = []
        for i in range(self.ensemble_size):
            model = self.train_advanced_model(train_loader, val_loader, i)
            self.models.append(model)
        
        # Select best models based on validation performance
        val_losses = [history['best_val_loss'] for history in self.training_history]
        best_indices = np.argsort(val_losses)[:max(3, self.ensemble_size // 2)]
        self.best_models = [self.models[i] for i in best_indices]
        
        logger.info(f"Training completed. Best models: {best_indices}")
        
        # Save the ensemble
        self.save_advanced_model()

    def save_advanced_model(self):
        """Save the advanced model ensemble"""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        save_dict = {
            'models': self.models,
            'best_models': self.best_models,
            'feature_scalers': self.feature_scalers,
            'target_scalers': self.target_scalers,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seq_length': self.seq_length,
            'feature_names': self.feature_names,
            'ensemble_size': self.ensemble_size,
            'training_history': self.training_history,
            'best_hyperparams': self.best_hyperparams,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Advanced model saved to {self.model_save_path}")

    def load_advanced_model(self) -> bool:
        """Load the advanced model ensemble"""
        try:
            with open(self.model_save_path, 'rb') as f:
                save_dict = pickle.load(f)
            
            self.models = save_dict['models']
            self.best_models = save_dict.get('best_models', self.models[:3])
            self.feature_scalers = save_dict['feature_scalers']
            self.target_scalers = save_dict.get('target_scalers', {})
            self.input_dim = save_dict['input_dim']
            self.output_dim = save_dict['output_dim']
            self.seq_length = save_dict['seq_length']
            self.feature_names = save_dict['feature_names']
            self.ensemble_size = save_dict['ensemble_size']
            self.training_history = save_dict.get('training_history', [])
            self.best_hyperparams = save_dict.get('best_hyperparams', {})
            self.use_attention = save_dict.get('use_attention', True)
            self.use_residual = save_dict.get('use_residual', True)
            
            logger.info(f"Advanced model loaded from {self.model_save_path}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"Model file not found at {self.model_save_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict_advanced(self, days_ahead: int = 30, temperature: float = 1.0, 
                        use_monte_carlo: bool = True, n_simulations: int = 1000) -> pd.DataFrame:
        """Advanced prediction with Monte Carlo simulation and uncertainty quantification"""
        if not self.models and not self.load_advanced_model():
            raise ValueError("No trained models available. Please train the model first.")
        
        logger.info(f"Generating advanced predictions for {days_ahead} days ahead")
        
        # Get latest data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        try:
            df = yf.Ticker(self.ticker).history(interval="1d", start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"Could not fetch data for ticker {self.ticker}")
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
        
        # Create advanced features for prediction
        df_features = self.create_advanced_features()
        
        # Prepare features for prediction
        X_pred = self.prepare_prediction_features(df_features)
        X_pred_tensor = torch.FloatTensor(X_pred).to(self.device)
        
        # Get latest price and date
        latest_date = df.index[-1]
        latest_price = df['Close'].iloc[-1]
        
        logger.info(f"Latest price for {self.ticker} (as of {latest_date}): ${latest_price:.2f}")
        
        if use_monte_carlo:
            return self._monte_carlo_prediction(X_pred_tensor, latest_price, latest_date, 
                                              days_ahead, temperature, n_simulations)
        else:
            return self._ensemble_prediction(X_pred_tensor, latest_price, latest_date, 
                                           days_ahead, temperature)

    def _monte_carlo_prediction(self, X_pred: torch.Tensor, latest_price: float, 
                               latest_date, days_ahead: int, temperature: float, 
                               n_simulations: int) -> pd.DataFrame:
        """Monte Carlo simulation for robust predictions"""
        models_to_use = self.best_models if self.best_models else self.models
        
        all_simulations = []
        
        for sim in range(n_simulations):
            # Random model selection
            model = np.random.choice(models_to_use)
            model.eval()
            
            simulation_prices = [latest_price]
            simulation_returns = []
            simulation_volatilities = []
            simulation_directions = []
            
            X_current = X_pred.clone()
            
            for day in range(days_ahead):
                with torch.no_grad():
                    predictions = model(X_current)
                    
                    # Add noise based on temperature
                    if temperature > 0:
                        noise_scale = temperature * 0.1
                        for key in ['price', 'volatility', 'magnitude']:
                            if key in predictions:
                                noise = torch.normal(0, noise_scale, size=predictions[key].shape).to(self.device)
                                predictions[key] += noise
                    
                    # Extract predictions and inverse transform
                    price_pred = predictions['price'].cpu().numpy().flatten()[0]
                    vol_pred = predictions['volatility'].cpu().numpy().flatten()[0]
                    direction_pred = F.softmax(predictions['direction'], dim=1).cpu().numpy()[0]
                    
                    # Inverse transform
                    price_return = self.target_scalers['price'].inverse_transform([[price_pred]])[0][0]
                    volatility = self.target_scalers['volatility'].inverse_transform([[vol_pred]])[0][0]
                    
                    # Calculate next price
                    next_price = simulation_prices[-1] * (1 + price_return)
                    simulation_prices.append(next_price)
                    simulation_returns.append(price_return)
                    simulation_volatilities.append(volatility)
                    simulation_directions.append(direction_pred[1])  # Probability of up movement
                    
                    # Update input for next prediction (simplified)
                    X_current = torch.cat((X_current[:, 1:, :], X_current[:, -1:, :]), dim=1)
            
            all_simulations.append({
                'prices': simulation_prices[1:],  # Exclude initial price
                'returns': simulation_returns,
                'volatilities': simulation_volatilities,
                'directions': simulation_directions
            })
        
        # Aggregate simulation results
        return self._aggregate_simulations(all_simulations, latest_date, days_ahead)

    def _aggregate_simulations(self, simulations: List[Dict], latest_date, days_ahead: int) -> pd.DataFrame:
        """Aggregate Monte Carlo simulation results"""
        # Convert to arrays for easier computation
        all_prices = np.array([sim['prices'] for sim in simulations])
        all_returns = np.array([sim['returns'] for sim in simulations])
        all_volatilities = np.array([sim['volatilities'] for sim in simulations])
        all_directions = np.array([sim['directions'] for sim in simulations])
        
        # Calculate statistics
        mean_prices = np.mean(all_prices, axis=0)
        std_prices = np.std(all_prices, axis=0)
        median_prices = np.median(all_prices, axis=0)
        
        # Confidence intervals
        lower_5 = np.percentile(all_prices, 5, axis=0)
        lower_25 = np.percentile(all_prices, 25, axis=0)
        upper_75 = np.percentile(all_prices, 75, axis=0)
        upper_95 = np.percentile(all_prices, 95, axis=0)
        
        # Other statistics
        mean_returns = np.mean(all_returns, axis=0)
        mean_volatilities = np.mean(all_volatilities, axis=0)
        prob_up = np.mean(all_directions, axis=0)
        
        # Calculate Value at Risk (VaR)
        var_1 = np.percentile(all_returns, 1, axis=0)
        var_5 = np.percentile(all_returns, 5, axis=0)
        
        # Create prediction dates
        prediction_dates = []
        current_date = latest_date
        for _ in range(days_ahead):
            current_date += timedelta(days=1)
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            prediction_dates.append(current_date)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': prediction_dates,
            'Mean_Price': mean_prices,
            'Median_Price': median_prices,
            'Std_Price': std_prices,
            'Lower_5%': lower_5,
            'Lower_25%': lower_25,
            'Upper_75%': upper_75,
            'Upper_95%': upper_95,
            'Expected_Return': mean_returns,
            'Expected_Volatility': mean_volatilities,
            'Probability_Up': prob_up,
            'VaR_1%': var_1,
            'VaR_5%': var_5,
        })
        
        return results

    def _ensemble_prediction(self, X_pred: torch.Tensor, latest_price: float, 
                           latest_date, days_ahead: int, temperature: float) -> pd.DataFrame:
        """Standard ensemble prediction without Monte Carlo"""
        models_to_use = self.best_models if self.best_models else self.models
        
        all_predictions = []
        X_current = X_pred.clone()
        
        for day in range(days_ahead):
            day_predictions = {'price': [], 'volatility': [], 'direction': []}
            
            for model in models_to_use:
                model.eval()
                with torch.no_grad():
                    predictions = model(X_current)
                    
                    # Add temperature-based noise
                    if temperature > 0:
                        noise_scale = temperature * 0.1
                        for key in ['price', 'volatility']:
                            noise = torch.normal(0, noise_scale, size=predictions[key].shape).to(self.device)
                            predictions[key] += noise
                    
                    day_predictions['price'].append(predictions['price'].cpu().numpy().flatten()[0])
                    day_predictions['volatility'].append(predictions['volatility'].cpu().numpy().flatten()[0])
                    day_predictions['direction'].append(
                        F.softmax(predictions['direction'], dim=1).cpu().numpy()[0][1]
                    )
            
            all_predictions.append(day_predictions)
            
            # Update input for next prediction
            X_current = torch.cat((X_current[:, 1:, :], X_current[:, -1:, :]), dim=1)
        
        # Process ensemble predictions
        prices = [latest_price]
        returns = []
        volatilities = []
        directions = []
        price_stds = []
        
        for day_pred in all_predictions:
            # Aggregate predictions
            mean_price_pred = np.mean(day_pred['price'])
            std_price_pred = np.std(day_pred['price'])
            mean_vol_pred = np.mean(day_pred['volatility'])
            mean_dir_pred = np.mean(day_pred['direction'])
            
            # Inverse transform
            price_return = self.target_scalers['price'].inverse_transform([[mean_price_pred]])[0][0]
            volatility = self.target_scalers['volatility'].inverse_transform([[mean_vol_pred]])[0][0]
            
            next_price = prices[-1] * (1 + price_return)
            prices.append(next_price)
            returns.append(price_return)
            volatilities.append(volatility)
            directions.append(mean_dir_pred)
            price_stds.append(std_price_pred)
        
        # Create prediction dates
        prediction_dates = []
        current_date = latest_date
        for _ in range(days_ahead):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            prediction_dates.append(current_date)
        
        # Calculate confidence intervals
        price_uncertainties = np.array(price_stds) + np.array(volatilities)
        lower_bounds = np.array(prices[1:]) - 1.96 * price_uncertainties
        upper_bounds = np.array(prices[1:]) + 1.96 * price_uncertainties
        
        results = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Price': prices[1:],
            'Expected_Return': returns,
            'Expected_Volatility': volatilities,
            'Probability_Up': directions,
            'Lower_Bound': lower_bounds,
            'Upper_Bound': upper_bounds,
            'Prediction_Uncertainty': price_uncertainties
        })
        
        return results

    def prepare_prediction_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction using the advanced feature set"""
        # Select the last seq_length days
        if len(df) < self.seq_length:
            raise ValueError(f"Insufficient data for prediction. Need at least {self.seq_length} days.")
        
        df_sequence = df.iloc[-self.seq_length:]
        
        # Extract and scale features
        X = np.zeros((1, self.seq_length, self.input_dim))
        
        for i, feature in enumerate(self.feature_names):
            if feature in df_sequence.columns:
                feature_values = df_sequence[feature].values.reshape(-1, 1)
                scaled_values = self.feature_scalers[feature].transform(feature_values).flatten()
                X[0, :, i] = scaled_values
            else:
                logger.warning(f"Feature {feature} not found in data, using zeros")
                X[0, :, i] = np.zeros(self.seq_length)
        
        return X

    def train(self):
        """Complete training pipeline"""
        # Fetch and prepare data
        self.fetch_data()
        X, y_price, y_vol = self.prepare_features()
        X_seq, y_price_seq, y_vol_seq = self.create_sequences(X, y_price, y_vol)
        train_loader, test_loader = self.train_test_split(X_seq, y_price_seq, y_vol_seq)
        
        # Train the model
        self.train_model(train_loader, test_loader)
        
        return "Training completed successfully!"

    def save_to_gcs(self, bucket_name, file_name, local_path):
        """Save a local file to Google Cloud Storage."""
        try:
            # Initialize the GCS client
            client = storage.Client()

            # Get the bucket and blob
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            # Upload the file to GCS
            blob.upload_from_filename(local_path)
            print(f"File {local_path} uploaded to bucket {bucket_name} as {file_name}")
        except Exception as e:
            print(f"An error occurred while uploading the file: {e}")

    def load_from_gcs(self, bucket_name, file_name, local_path):
        """Load a file from Google Cloud Storage and save it locally."""
        try:
            # Initialize the GCS client
            client = storage.Client()

            # Get the bucket and blob
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            # Download the file to the specified local path
            blob.download_to_filename(local_path)
            print(f"File {file_name} downloaded from bucket {bucket_name} to {local_path}")
        except Exception as e:
            print(f"An error occurred while downloading the file: {e}")
            """Load a file from Google Cloud Storage and save it locally."""

    def plot_predictions(self, predictions):
        """Plot price predictions with confidence intervals"""
        plt.figure(figsize=(12, 6))
        
        # Plot the predicted price
        plt.plot(predictions['Date'], predictions['Predicted_Price'], label='Predicted Price', color='blue')
        
        # Plot confidence intervals
        plt.fill_between(
            predictions['Date'],
            predictions['Lower_Bound'],
            predictions['Upper_Bound'],
            color='blue',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Format the plot
        plt.title(f'{self.ticker} Price Prediction with Volatility')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save and show the plot
        plt.savefig(f"{self.ticker}_prediction.png")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Stock Ticker LSTM Model')
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run the script in')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    train_parser.add_argument('--start_date', type=str, required=True, help='Start date for training data (YYYY-MM-DD)')
    train_parser.add_argument('--end_date', type=str, help='End date for training data (YYYY-MM-DD)')
    train_parser.add_argument('--seq_length', type=int, default=30, help='Sequence length for LSTM input')
    train_parser.add_argument('--ensemble_size', type=int, default=5, help='Number of models in the ensemble')
    train_parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for LSTM')
    train_parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Prediction arguments
    predict_parser = subparsers.add_parser('predict', help='Make predictions using a trained model')
    predict_parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    predict_parser.add_argument('--days', type=int, default=30, help='Number of days to predict')
    predict_parser.add_argument('--temperature', type=float, default=1.0, 
                               help='Temperature parameter for prediction randomness (0.0-2.0)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'train':
        predictor = StockPredictor(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            seq_length=args.seq_length,
            ensemble_size=args.ensemble_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            dropout=args.dropout
        )
        
        result = predictor.train()
        print(result)
        
    elif args.mode == 'predict':
        predictor = StockPredictor(ticker=args.ticker)
        
        if predictor.load_model():
            predictions = predictor.predict(days_ahead=args.days, temperature=args.temperature)
            
            # Display prediction results
            pd.set_option('display.max_rows', None)
            print("\nPrediction Results:")
            print(predictions)
            
            # Plot predictions
            predictor.plot_predictions(predictions)
            
            # Save predictions to CSV
            csv_filename = f"{args.ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            predictions.to_csv(csv_filename, index=False)
            print(f"\nPredictions saved to {csv_filename}")
        else:
            print(f"No trained model found for {args.ticker}. Please train the model first.")
    
    else:
        print("Please specify a mode: train or predict")
        print("Example usage for training:")
        print("  python stock_lstm.py train --ticker AAPL --start_date 2015-01-01")
        print("\nExample usage for prediction:")
        print("  python stock_lstm.py predict --ticker AAPL --days 30 --temperature 1.0")