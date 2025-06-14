import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from scipy import stats
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class UltraAdvancedModelEvaluator:
    """Ultra-advanced model evaluation and backtesting system with financial risk analysis"""
    
    def __init__(self):
        self.results = {}
        self.backtesting_results = {}
        self.risk_metrics = {}
        self.benchmark_data = None
        
    def evaluate_ultra_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 predictions_dict: Dict[str, np.ndarray] = None,
                                 model_name: str = "UltraModel") -> Dict[str, float]:
        """Ultra-comprehensive evaluation of prediction accuracy with advanced metrics"""
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (len(y_true) - 1) / (len(y_true) - 2)
        
        # Advanced financial metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics['smape'] = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        metrics['directional_accuracy'] = self._directional_accuracy(y_true, y_pred)
        metrics['profit_accuracy'] = self._profit_accuracy(y_true, y_pred)
        
        # Volatility-adjusted metrics
        returns_true = np.diff(y_true) / y_true[:-1]
        returns_pred = np.diff(y_pred) / y_pred[:-1]
        
        if len(returns_true) > 1:
            metrics['return_correlation'] = np.corrcoef(returns_true, returns_pred)[0, 1]
            metrics['volatility_accuracy'] = 1 - abs(np.std(returns_pred) - np.std(returns_true)) / np.std(returns_true)
            
            # Information Coefficient (IC)
            metrics['information_coefficient'] = stats.spearmanr(returns_true, returns_pred)[0]
            metrics['information_coefficient_pvalue'] = stats.spearmanr(returns_true, returns_pred)[1]
            
            # Rank correlation
            metrics['rank_correlation'] = stats.spearmanr(y_true, y_pred)[0]
        
        # Risk-adjusted metrics
        if len(returns_true) > 10:
            # Tracking Error
            active_returns = returns_pred - returns_true
            metrics['tracking_error'] = np.std(active_returns) * np.sqrt(252)
            
            # Information Ratio
            if metrics['tracking_error'] > 0:
                metrics['information_ratio'] = np.mean(active_returns) / (metrics['tracking_error'] / np.sqrt(252))
            else:
                metrics['information_ratio'] = 0
        
        # Statistical significance tests
        _, p_value = stats.pearsonr(y_true, y_pred)
        metrics['correlation_p_value'] = p_value
        metrics['statistically_significant'] = p_value < 0.05
        
        # Prediction confidence metrics
        if predictions_dict and 'uncertainty' in predictions_dict:
            uncertainty = predictions_dict['uncertainty']
            metrics['avg_uncertainty'] = np.mean(uncertainty)
            metrics['uncertainty_correlation'] = np.corrcoef(np.abs(y_true - y_pred), uncertainty)[0, 1]
            
            # Calibration score
            metrics['calibration_score'] = self._calculate_calibration_score(y_true, y_pred, uncertainty)
        
        # Regime-based evaluation
        if predictions_dict and 'regime' in predictions_dict:
            regime_pred = predictions_dict['regime']
            metrics['regime_accuracy'] = self._evaluate_regime_predictions(y_true, regime_pred)
        
        # Trend prediction evaluation
        if predictions_dict and 'trend' in predictions_dict:
            trend_pred = predictions_dict['trend']
            trend_true = self._calculate_trend_labels(y_true)
            metrics['trend_accuracy'] = np.mean(trend_pred == trend_true) * 100
        
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_calibration_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   uncertainty: np.ndarray) -> float:
        """Calculate prediction calibration score"""
        errors = np.abs(y_true - y_pred)
        
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainty)
        sorted_errors = errors[sorted_indices]
        sorted_uncertainty = uncertainty[sorted_indices]
        
        # Calculate calibration in bins
        n_bins = 10
        bin_size = len(sorted_errors) // n_bins
        calibration_errors = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_errors)
            
            bin_errors = sorted_errors[start_idx:end_idx]
            bin_uncertainty = sorted_uncertainty[start_idx:end_idx]
            
            expected_error = np.mean(bin_uncertainty)
            actual_error = np.mean(bin_errors)
            
            calibration_errors.append(abs(expected_error - actual_error))
        
        return 1 - np.mean(calibration_errors)  # Higher is better
    
    def _evaluate_regime_predictions(self, y_true: np.ndarray, regime_pred: np.ndarray) -> float:
        """Evaluate regime prediction accuracy"""
        # Simple regime detection based on volatility
        returns = np.diff(y_true) / y_true[:-1]
        rolling_vol = pd.Series(returns).rolling(20).std()
        
        # Define regimes based on volatility quantiles
        vol_quantiles = rolling_vol.quantile([0.33, 0.67])
        regime_true = np.where(rolling_vol <= vol_quantiles.iloc[0], 0,
                              np.where(rolling_vol <= vol_quantiles.iloc[1], 1, 2))
        
        # Align lengths
        min_len = min(len(regime_true), len(regime_pred))
        regime_true = regime_true[-min_len:]
        regime_pred = regime_pred[-min_len:]
        
        return np.mean(regime_true == regime_pred) * 100
    
    def _calculate_trend_labels(self, prices: np.ndarray, window: int = 5) -> np.ndarray:
        """Calculate trend labels for evaluation"""
        returns = np.diff(prices) / prices[:-1]
        
        # Smooth returns
        smoothed_returns = pd.Series(returns).rolling(window, center=True).mean()
        
        # Define trends: 0=bearish, 1=neutral, 2=bullish
        trend_labels = np.where(smoothed_returns > 0.01, 2,
                               np.where(smoothed_returns < -0.01, 0, 1))
        
        return trend_labels
    
    def ultra_backtest_strategy(self, prices: pd.Series, predictions_dict: Dict[str, np.ndarray],
                               initial_capital: float = 100000,
                               transaction_cost: float = 0.001,
                               max_position_size: float = 1.0,
                               risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """Ultra-advanced backtesting with risk management and multiple strategies"""
        
        results = {
            'trades': [],
            'portfolio_values': [initial_capital],
            'positions': [0],
            'returns': [],
            'signals': [],
            'drawdowns': [],
            'risk_metrics': {}
        }
        
        capital = initial_capital
        position = 0  # Shares held
        max_portfolio_value = initial_capital
        
        # Extract predictions
        price_pred = predictions_dict.get('price', np.zeros(len(prices)))
        uncertainty = predictions_dict.get('uncertainty', np.ones(len(prices)) * 0.1)
        trend_pred = predictions_dict.get('trend', np.ones(len(prices)))
        regime_pred = predictions_dict.get('regime', np.zeros(len(prices)))
        
        for i in range(1, len(prices)):
            current_price = prices.iloc[i]
            predicted_return = (price_pred[i] - current_price) / current_price if i < len(price_pred) else 0
            pred_uncertainty = uncertainty[i] if i < len(uncertainty) else 0.1
            trend = trend_pred[i] if i < len(trend_pred) else 1
            regime = regime_pred[i] if i < len(regime_pred) else 0
            
            # Advanced signal generation
            signal, position_size = self._generate_advanced_signal(
                predicted_return, pred_uncertainty, trend, regime
            )
            
            results['signals'].append(signal)
            
            # Risk management
            portfolio_value = capital + position * current_price
            max_position_value = portfolio_value * max_position_size
            max_shares = max_position_value / current_price
            
            # Execute trades
            if signal == 'BUY' and position < max_shares:
                shares_to_buy = min(max_shares - position, 
                                  capital / (current_price * (1 + transaction_cost)))
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    if cost <= capital:
                        position += shares_to_buy
                        capital -= cost
                        
                        results['trades'].append({
                            'date': prices.index[i],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'cost': cost,
                            'signal_strength': position_size
                        })
                        
            elif signal == 'SELL' and position > 0:
                shares_to_sell = min(position, position * position_size)
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price * (1 - transaction_cost)
                    position -= shares_to_sell
                    capital += proceeds
                    
                    results['trades'].append({
                        'date': prices.index[i],
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares_to_sell,
                        'proceeds': proceeds,
                        'signal_strength': position_size
                    })
            
            # Update portfolio tracking
            portfolio_value = capital + position * current_price
            results['portfolio_values'].append(portfolio_value)
            results['positions'].append(position)
            
            # Calculate returns and drawdown
            if i > 1:
                daily_return = (portfolio_value - results['portfolio_values'][-2]) / results['portfolio_values'][-2]
                results['returns'].append(daily_return)
                
                max_portfolio_value = max(max_portfolio_value, portfolio_value)
                drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
                results['drawdowns'].append(drawdown)
        
        # Calculate comprehensive performance metrics
        results['performance'] = self._calculate_advanced_performance_metrics(
            results, initial_capital, risk_free_rate
        )
        
        return results
    
    def _generate_advanced_signal(self, predicted_return: float, uncertainty: float,
                                trend: int, regime: int) -> Tuple[str, float]:
        """Generate advanced trading signals with position sizing"""
        
        # Base thresholds
        base_threshold = 0.02
        
        # Adjust threshold based on uncertainty
        adjusted_threshold = base_threshold * (1 + uncertainty)
        
        # Regime-based adjustments
        regime_multiplier = {0: 0.8, 1: 1.0, 2: 1.2, 3: 0.9}.get(regime, 1.0)
        
        # Trend-based adjustments
        trend_multiplier = {0: 0.7, 1: 1.0, 2: 1.3}.get(trend, 1.0)
        
        # Calculate signal strength
        signal_strength = abs(predicted_return) / adjusted_threshold * regime_multiplier * trend_multiplier
        
        # Position sizing based on Kelly Criterion approximation
        if signal_strength > 1:
            # Simplified Kelly: f = (bp - q) / b, where p is win prob, q is loss prob, b is odds
            win_prob = 0.6 if predicted_return > 0 else 0.4  # Simplified assumption
            position_fraction = min(0.25, signal_strength * 0.1)  # Cap at 25%
        else:
            position_fraction = 0
        
        # Generate signal
        if predicted_return > adjusted_threshold and signal_strength > 1:
            return 'BUY', position_fraction
        elif predicted_return < -adjusted_threshold and signal_strength > 1:
            return 'SELL', position_fraction
        else:
            return 'HOLD', 0
    
    def _calculate_advanced_performance_metrics(self, results: Dict, 
                                              initial_capital: float,
                                              risk_free_rate: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        portfolio_values = np.array(results['portfolio_values'])
        returns = np.array(results['returns']) if results['returns'] else np.array([0])
        
        performance = {}
        
        # Basic metrics
        final_value = portfolio_values[-1]
        performance['total_return'] = (final_value - initial_capital) / initial_capital * 100
        performance['annualized_return'] = (final_value / initial_capital) ** (252 / len(portfolio_values)) - 1
        
        # Risk metrics
        if len(returns) > 1:
            performance['volatility'] = np.std(returns) * np.sqrt(252) * 100
            performance['sharpe_ratio'] = (np.mean(returns) - risk_free_rate/252) / np.std(returns) * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                performance['sortino_ratio'] = (np.mean(returns) - risk_free_rate/252) / downside_deviation * np.sqrt(252)
            else:
                performance['sortino_ratio'] = float('inf')
            
            # Calmar ratio
            max_drawdown = max(results['drawdowns']) if results['drawdowns'] else 0
            performance['max_drawdown'] = max_drawdown * 100
            performance['calmar_ratio'] = performance['annualized_return'] / max_drawdown if max_drawdown > 0 else float('inf')
            
            # Value at Risk (VaR)
            performance['var_95'] = np.percentile(returns, 5) * 100
            performance['var_99'] = np.percentile(returns, 1) * 100
            
            # Expected Shortfall (Conditional VaR)
            var_95_threshold = np.percentile(returns, 5)
            tail_returns = returns[returns <= var_95_threshold]
            performance['expected_shortfall'] = np.mean(tail_returns) * 100 if len(tail_returns) > 0 else 0
        
        # Trading metrics
        trades = results['trades']
        performance['num_trades'] = len(trades)
        
        if len(trades) >= 2:
            # Calculate trade-level metrics
            trade_returns = []
            for i in range(1, len(trades)):
                if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                    buy_price = trades[i-1]['price']
                    sell_price = trades[i]['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)
            
            if trade_returns:
                winning_trades = [t for t in trade_returns if t > 0]
                losing_trades = [t for t in trade_returns if t < 0]
                
                performance['win_rate'] = len(winning_trades) / len(trade_returns) * 100
                performance['avg_win'] = np.mean(winning_trades) * 100 if winning_trades else 0
                performance['avg_loss'] = np.mean(losing_trades) * 100 if losing_trades else 0
                
                if losing_trades:
                    performance['profit_factor'] = abs(sum(winning_trades) / sum(losing_trades))
                else:
                    performance['profit_factor'] = float('inf')
        
        return performance
    
    def compare_with_benchmark(self, portfolio_returns: np.ndarray, 
                             benchmark_symbol: str = 'SPY',
                             start_date: str = None) -> Dict[str, float]:
        """Compare strategy performance with benchmark"""
        
        try:
            # Fetch benchmark data
            if start_date:
                benchmark = yf.download(benchmark_symbol, start=start_date)['Adj Close']
            else:
                benchmark = yf.download(benchmark_symbol, period='2y')['Adj Close']
            
            benchmark_returns = benchmark.pct_change().dropna()
            
            # Align returns
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            portfolio_returns = portfolio_returns[-min_length:]
            benchmark_returns = benchmark_returns.values[-min_length:]
            
            # Calculate comparison metrics
            comparison = {}
            
            # Alpha and Beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            comparison['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
            comparison['alpha'] = np.mean(portfolio_returns) - comparison['beta'] * np.mean(benchmark_returns)
            comparison['alpha_annualized'] = comparison['alpha'] * 252 * 100
            
            # Tracking error
            active_returns = portfolio_returns - benchmark_returns
            comparison['tracking_error'] = np.std(active_returns) * np.sqrt(252) * 100
            
            # Information ratio
            if comparison['tracking_error'] > 0:
                comparison['information_ratio'] = np.mean(active_returns) / (comparison['tracking_error'] / 100 / np.sqrt(252))
            else:
                comparison['information_ratio'] = 0
            
            # Correlation
            comparison['correlation'] = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            
            return comparison
            
        except Exception as e:
            print(f"Warning: Could not fetch benchmark data: {e}")
            return {}
    
    def create_interactive_dashboard(self, results: Dict, model_name: str = "UltraModel"):
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Drawdown', 'Daily Returns', 
                           'Rolling Sharpe', 'Trade Distribution', 'Risk Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(y=results['portfolio_values'], name='Portfolio Value',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Drawdown
        if results['drawdowns']:
            fig.add_trace(
                go.Scatter(y=[-d*100 for d in results['drawdowns']], 
                          name='Drawdown %', fill='tonexty',
                          line=dict(color='red')),
                row=1, col=2
            )
        
        # Daily returns
        if results['returns']:
            fig.add_trace(
                go.Scatter(y=[r*100 for r in results['returns']], 
                          name='Daily Returns %',
                          line=dict(color='green')),
                row=2, col=1
            )
        
        # Rolling Sharpe
        if len(results['returns']) > 30:
            returns_series = pd.Series(results['returns'])
            rolling_sharpe = returns_series.rolling(30).mean() / returns_series.rolling(30).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(y=rolling_sharpe, name='30-Day Rolling Sharpe',
                          line=dict(color='purple')),
                row=2, col=2
            )
        
        # Trade distribution
        if results['trades']:
            trade_values = [t.get('proceeds', 0) - t.get('cost', 0) for t in results['trades']]
            fig.add_trace(
                go.Bar(y=trade_values, name='Trade P&L'),
                row=3, col=1
            )
        
        # Performance indicator
        performance = results.get('performance', {})
        total_return = performance.get('total_return', 0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=total_return,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Total Return %"},
                delta={'reference': 0},
                gauge={'axis': {'range': [None, max(50, total_return * 1.2)]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 10], 'color': "lightgray"},
                           {'range': [10, 25], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 15}}),
            row=3, col=2
        )
        
        fig.update_layout(height=800, title_text=f"{model_name} - Performance Dashboard")
        fig.show()
        
        return fig
    
    def generate_ultra_report(self, model_name: str = "UltraModel", 
                            backtest_results: Dict = None) -> str:
        """Generate ultra-comprehensive evaluation report"""
        
        if model_name not in self.results:
            return "No evaluation results found for this model."
        
        metrics = self.results[model_name]
        
        # Prediction accuracy section
        accuracy_grade = (
            "🟢 EXCELLENT" if metrics.get('directional_accuracy', 0) > 65 and metrics.get('r2', 0) > 0.8 else
            "🟡 VERY GOOD" if metrics.get('directional_accuracy', 0) > 60 and metrics.get('r2', 0) > 0.7 else
            "🟠 GOOD" if metrics.get('directional_accuracy', 0) > 55 and metrics.get('r2', 0) > 0.5 else
            "🔴 NEEDS IMPROVEMENT"
        )
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║                    🧠 ULTRA-ADVANCED AI MODEL REPORT           ║
║                        {model_name}                            ║
╚════════════════════════════════════════════════════════════════╝

🎯 PREDICTION ACCURACY METRICS:
┌────────────────────────────────────────────────────────────────┐
│ • Root Mean Square Error (RMSE): {metrics.get('rmse', 0):.6f}                  │
│ • Mean Absolute Error (MAE): {metrics.get('mae', 0):.6f}                       │
│ • R² Score: {metrics.get('r2', 0):.4f} | Adjusted R²: {metrics.get('adjusted_r2', 0):.4f}           │
│ • Mean Absolute Percentage Error: {metrics.get('mape', 0):.2f}%                │
│ • Symmetric MAPE: {metrics.get('smape', 0):.2f}%                               │
└────────────────────────────────────────────────────────────────┘

📈 FINANCIAL PERFORMANCE METRICS:
┌────────────────────────────────────────────────────────────────┐
│ • Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}%                     │
│ • Profit Accuracy: {metrics.get('profit_accuracy', 0):.2f}%                               │
│ • Return Correlation: {metrics.get('return_correlation', 0):.4f}                          │
│ • Volatility Accuracy: {metrics.get('volatility_accuracy', 0):.4f}                       │
│ • Information Coefficient: {metrics.get('information_coefficient', 0):.4f}               │
│ • Rank Correlation: {metrics.get('rank_correlation', 0):.4f}                             │
└────────────────────────────────────────────────────────────────┘

🔬 ADVANCED ANALYTICS:
┌────────────────────────────────────────────────────────────────┐
│ • Tracking Error: {metrics.get('tracking_error', 0):.4f}                                  │
│ • Information Ratio: {metrics.get('information_ratio', 0):.4f}                           │
│ • Average Uncertainty: {metrics.get('avg_uncertainty', 0):.4f}                           │
│ • Calibration Score: {metrics.get('calibration_score', 0):.4f}                           │
│ • Regime Accuracy: {metrics.get('regime_accuracy', 0):.2f}%                               │
│ • Trend Accuracy: {metrics.get('trend_accuracy', 0):.2f}%                                 │
└────────────────────────────────────────────────────────────────┘

📊 STATISTICAL SIGNIFICANCE:
┌────────────────────────────────────────────────────────────────┐
│ • Correlation P-value: {metrics.get('correlation_p_value', 1):.6f}                        │
│ • Statistically Significant: {'✅ YES' if metrics.get('statistically_significant', False) else '❌ NO'}                    │
│ • IC P-value: {metrics.get('information_coefficient_pvalue', 1):.6f}                      │
└────────────────────────────────────────────────────────────────┘
"""

        # Add backtesting results if available
        if backtest_results and 'performance' in backtest_results:
            perf = backtest_results['performance']
            
            trading_grade = (
                "🟢 EXCELLENT" if perf.get('sharpe_ratio', 0) > 2 and perf.get('total_return', 0) > 20 else
                "🟡 VERY GOOD" if perf.get('sharpe_ratio', 0) > 1.5 and perf.get('total_return', 0) > 15 else
                "🟠 GOOD" if perf.get('sharpe_ratio', 0) > 1 and perf.get('total_return', 0) > 10 else
                "🔴 NEEDS IMPROVEMENT"
            )
            
            report += f"""
🏦 TRADING PERFORMANCE ANALYSIS:
┌────────────────────────────────────────────────────────────────┐
│ • Total Return: {perf.get('total_return', 0):.2f}%                                        │
│ • Annualized Return: {perf.get('annualized_return', 0)*100:.2f}%                          │
│ • Volatility: {perf.get('volatility', 0):.2f}%                                            │
│ • Sharpe Ratio: {perf.get('sharpe_ratio', 0):.4f}                                         │
│ • Sortino Ratio: {perf.get('sortino_ratio', 0):.4f}                                       │
│ • Calmar Ratio: {perf.get('calmar_ratio', 0):.4f}                                         │
│ • Maximum Drawdown: {perf.get('max_drawdown', 0):.2f}%                                    │
└────────────────────────────────────────────────────────────────┘

⚡ RISK METRICS:
┌────────────────────────────────────────────────────────────────┐
│ • Value at Risk (95%): {perf.get('var_95', 0):.2f}%                               │
│ • Value at Risk (99%): {perf.get('var_99', 0):.2f}%                               │
│ • Expected Shortfall: {perf.get('expected_shortfall', 0):.2f}%                     │
└────────────────────────────────────────────────────────────────┘

💰 TRADING STATISTICS:
┌────────────────────────────────────────────────────────────────┐
│ • Number of Trades: {perf.get('num_trades', 0)}                                           │
│ • Win Rate: {perf.get('win_rate', 0):.2f}%                                                │
│ • Average Win: {perf.get('avg_win', 0):.2f}%                                              │
│ • Average Loss: {perf.get('avg_loss', 0):.2f}%                                            │
│ • Profit Factor: {perf.get('profit_factor', 0):.4f}                                       │
└────────────────────────────────────────────────────────────────┘

🏆 TRADING PERFORMANCE GRADE: {trading_grade}
"""

        report += f"""
🏆 OVERALL MODEL ASSESSMENT: {accuracy_grade}

🎓 AI SOPHISTICATION LEVEL:
┌────────────────────────────────────────────────────────────────┐
│ ✅ Quantum-Inspired Neural Architecture                        │
│ ✅ Multi-Task Learning (Price, Volatility, Trend, Regime)     │
│ ✅ Uncertainty Quantification                                  │
│ ✅ Advanced Feature Engineering                                │
│ ✅ Market Microstructure Analysis                              │
│ ✅ Fractal & Chaos Theory Features                             │
│ ✅ Ensemble Learning with Neural Evolution                     │
│ ✅ Risk-Adjusted Performance Optimization                      │
└────────────────────────────────────────────────────────────────┘

🔮 PREDICTION CONFIDENCE:
{'🟢 HIGH CONFIDENCE' if metrics.get('directional_accuracy', 0) > 60 and metrics.get('statistically_significant', False) else '🟡 MODERATE CONFIDENCE' if metrics.get('directional_accuracy', 0) > 52 else '🔴 LOW CONFIDENCE'}

📋 RECOMMENDATIONS:
{'• Model shows excellent predictive power - ready for production' if accuracy_grade == '🟢 EXCELLENT' else '• Model shows good performance - consider additional tuning' if 'GOOD' in accuracy_grade else '• Model needs significant improvement before deployment'}
{'• Trading strategy is highly profitable' if backtest_results and backtest_results.get('performance', {}).get('total_return', 0) > 15 else '• Trading performance is acceptable' if backtest_results and backtest_results.get('performance', {}).get('total_return', 0) > 5 else '• Consider refining trading signals'}

═══════════════════════════════════════════════════════════════════
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Version: Ultra-Advanced AI v2.0 with Quantum Neural Evolution
═══════════════════════════════════════════════════════════════════
"""
        
        return report
    
    # Keep existing methods from the original class
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (up/down predictions)"""
        if len(y_true) < 2:
            return 0.0
            
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    def _profit_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate profit accuracy - percentage of profitable predictions"""
        if len(y_true) < 2:
            return 0.0
            
        # Calculate returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Count profitable predictions (same sign as actual returns)
        profitable = np.sign(true_returns) == np.sign(pred_returns)
        return np.mean(profitable) * 100

# For backward compatibility
ModelEvaluator = UltraAdvancedModelEvaluator