import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas_ta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import datetime as dt

class TradingDashboard:
    def __init__(self):
        self.sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        self.sp500['Symbol'] = self.sp500['Symbol'].str.replace('.', '-')
        
    def initialize_dashboard(self):
        st.set_page_config(layout="wide", page_title="Trading Model Dashboard")
        st.title("S&P 500 Momentum Trading Dashboard")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Model Parameters")
            self.lookback_period = st.slider("Lookback Period (months)", 1, 12, 6)
            self.n_clusters = st.slider("Number of Clusters", 2, 6, 4)
            self.update_button = st.button("Update Analysis")
    
    def fetch_and_process_data(self):
        with st.spinner("Fetching market data..."):
            # Get top 150 liquid stocks
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=365*2)
            
            # Sample 150 stocks for demo (can be modified for production)
            sample_symbols = self.sp500['Symbol'].sample(150).tolist()
            
            # Fetch data
            df = yf.download(tickers=sample_symbols, start=start_date, end=end_date)['Adj Close']
            
            # Calculate features
            features = pd.DataFrame()
            for symbol in df.columns:
                stock_data = df[symbol].dropna()
                if len(stock_data) > 0:
                    features[symbol] = {
                        'returns_1m': stock_data.pct_change(20).iloc[-1],
                        'returns_3m': stock_data.pct_change(60).iloc[-1],
                        'returns_6m': stock_data.pct_change(120).iloc[-1],
                        'volatility': stock_data.pct_change().std() * np.sqrt(252),
                        'rsi': pandas_ta.rsi(stock_data, length=14).iloc[-1],
                    }
            
            return df, features.T
    
    def perform_clustering(self, features):
        with st.spinner("Performing cluster analysis..."):
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            features['Cluster'] = clusters
            return features
    
    def optimize_portfolio(self, prices, cluster_features):
        with st.spinner("Optimizing portfolio..."):
            # Select stocks from the momentum cluster
            momentum_cluster = cluster_features[cluster_features['rsi'] > 60].index.tolist()
            
            if len(momentum_cluster) > 0:
                portfolio_prices = prices[momentum_cluster]
                
                # Calculate optimization inputs
                mu = expected_returns.mean_historical_return(portfolio_prices)
                S = risk_models.sample_cov(portfolio_prices)
                
                # Optimize for maximum Sharpe Ratio
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe()
                clean_weights = ef.clean_weights()
                
                return pd.Series(clean_weights)
            return pd.Series()
    
    def calculate_performance_metrics(self, prices, weights):
        if len(weights) > 0:
            returns = prices[weights.index].pct_change()
            portfolio_returns = (returns * weights).sum(axis=1)
            
            metrics = {
                'Total Return': f"{(np.exp(portfolio_returns.sum()) - 1)*100:.2f}%",
                'Annual Volatility': f"{portfolio_returns.std() * np.sqrt(252)*100:.2f}%",
                'Sharpe Ratio': f"{portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252):.2f}",
                'Max Drawdown': f"{(portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()*100:.2f}%"
            }
            return metrics
        return {}
    
    def render_dashboard(self):
        # Initialize dashboard
        self.initialize_dashboard()
        
        if self.update_button:
            # Fetch and process data
            prices, features = self.fetch_and_process_data()
            
            # Perform clustering
            clustered_features = self.perform_clustering(features)
            
            # Optimize portfolio
            weights = self.optimize_portfolio(prices, clustered_features)
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(prices, weights)
            
            # Display dashboard components
            col1, col2, col3, col4 = st.columns(4)
            
            if metrics:
                with col1:
                    st.metric("Total Return", metrics['Total Return'])
                with col2:
                    st.metric("Annual Volatility", metrics['Annual Volatility'])
                with col3:
                    st.metric("Sharpe Ratio", metrics['Sharpe Ratio'])
                with col4:
                    st.metric("Max Drawdown", metrics['Max Drawdown'])
            
            # Display cluster analysis
            st.subheader("Cluster Analysis")
            fig = px.scatter(clustered_features, 
                           x='rsi', y='returns_1m', 
                           color='Cluster',
                           hover_data=['volatility'])
            st.plotly_chart(fig)
            
            # Display portfolio composition
            if len(weights) > 0:
                st.subheader("Portfolio Composition")
                weights_df = pd.DataFrame(weights).reset_index()
                weights_df.columns = ['Stock', 'Weight']
                fig = px.pie(weights_df, values='Weight', names='Stock')
                st.plotly_chart(fig)
            
            # Display performance chart
            if len(weights) > 0:
                st.subheader("Performance Chart")
                portfolio_returns = (prices[weights.index].pct_change() * weights).sum(axis=1)
                cumulative_returns = (1 + portfolio_returns).cumprod()
                fig = px.line(cumulative_returns)
                st.plotly_chart(fig)

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.render_dashboard()