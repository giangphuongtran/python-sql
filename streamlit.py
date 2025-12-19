import sqlite3
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import PowerTransformer, StandardScaler

from features import compute_features, compute_technical_indicators

warnings.filterwarnings("ignore")
from sklearn_extra.cluster import KMedoids

@st.cache_data
def load_features(db_path: str):
    feats = compute_features(db_path)
    return feats

def prepare_features(feats: pd.DataFrame):
    features = feats.copy()
    skew = features.skew()
    skewed_cols = skew[abs(skew) > 2].index.tolist()
    features_transformed = features.copy()
    for col in skewed_cols:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        features_transformed[col] = pt.fit_transform(features[[col]]).flatten()
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features_transformed),
        index=features_transformed.index,
        columns=features_transformed.columns,
    )
    return scaled
    
@st.cache_data
def run_clustering(
    method: str,
    n_clusters: int,
    features_scaled: pd.DataFrame,
):
    tickers = features_scaled.index.tolist()
    
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(features_scaled)
    elif method == "PAM":
        model = KMedoids(n_clusters=n_clusters, metric="euclidean", random_state=42)
        labels = model.fit_predict(features_scaled)
    else:
        # Hierarchical clustering with cosine distance and average linkage (as in clustering.py)
        distances = pdist(features_scaled, metric="cosine")
        linkage_matrix = linkage(distances, method="average")
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust") - 1
        model = None
    
    clusters_df = pd.DataFrame({"Ticker": tickers, "Cluster": labels + 1})
    
    # Calculate multiple clustering metrics
    sil = silhouette_score(features_scaled, labels)
    db = davies_bouldin_score(features_scaled, labels)
    ch = calinski_harabasz_score(features_scaled, labels)
    
    metrics = {
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch
    }
    
    return clusters_df, metrics, model

@st.cache_data
def load_prices(db_path: str, tickers):
    if not tickers:
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT ticker, date, open, high, low, close, volume FROM daily_bars "
            f"WHERE ticker IN ({','.join(['?'] * len(tickers))}) ORDER BY ticker, date",
            conn,
            params=list(tickers),
            parse_dates=["date"],
        )
    return df
        
@st.cache_data
def load_raw_bars(db_path: str):
    """Load raw daily bars data before aggregation."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT ticker, date, open, high, low, close, volume FROM daily_bars ORDER BY ticker, date",
            conn,
            parse_dates=["date"],
        )
    return df

def get_indicator_description(indicator: str) -> str:
    """Get description for a technical indicator."""
    descriptions = {
        'RSI': "**RSI (Relative Strength Index)**: Momentum oscillator (0-100) measuring speed/magnitude of price changes. "
               "Calculated as: 100 - (100 / (1 + RS)), where RS = average gain / average loss over 14 periods. "
               "Used to identify overbought (>70) or oversold (<30) conditions.",
        'ATR': "**ATR (Average True Range)**: Volatility measure indicating average trading range. "
               "Calculated as 14-period moving average of True Range (max of high-low, |high-prev_close|, |low-prev_close|). "
               "Used to assess market volatility and set stop-loss levels.",
        'ADX': "**ADX (Average Directional Index)**: Trend strength indicator (0-100). "
               "Based on comparison of upward vs downward price movement relative to True Range. "
               "Higher values (>25) indicate stronger trends; used to confirm trend direction.",
        'MACD_Hist': "**MACD Histogram**: Momentum indicator showing difference between MACD line and signal line. "
                    "MACD = 12-period EMA - 26-period EMA; Signal = 9-period EMA of MACD. "
                    "Positive values indicate bullish momentum; used to identify trend changes.",
        'Stoch_K': "**Stochastic K**: Momentum indicator comparing closing price to price range over 14 periods. "
                  "Formula: 100 * (close - lowest_low) / (highest_high - lowest_low). "
                  "Values >80 suggest overbought; <20 suggest oversold conditions.",
        'SMA20': "**SMA20 (20-day Simple Moving Average)**: Average closing price over 20 days. "
                "Used to identify short-term trends; price above SMA20 indicates bullish momentum.",
        'SMA60': "**SMA60 (60-day Simple Moving Average)**: Average closing price over 60 days. "
                "Used to identify medium-term trends and support/resistance levels.",
        'SMA200': "**SMA200 (200-day Simple Moving Average)**: Average closing price over 200 days. "
                 "Major trend indicator; price above SMA200 suggests long-term uptrend.",
        'BB_Upper': "**Bollinger Band Upper**: Upper boundary of Bollinger Bands (SMA20 + 2*standard deviation). "
                   "Price touching upper band may indicate overbought condition.",
        'BB_Lower': "**Bollinger Band Lower**: Lower boundary of Bollinger Bands (SMA20 - 2*standard deviation). "
                   "Price touching lower band may indicate oversold condition.",
        'Volume': "**Volume**: Number of shares traded per day. "
                 "Used to confirm price movements; high volume with price increases suggests strong buying interest."
    }
    return descriptions.get(indicator, "No description available.")
        
def candlestick_chart(ticker_data: pd.DataFrame, ticker: str, selected_indicators: list):
    """Create candlestick chart with selected technical indicators using subplots for RSI, MACD, ADX, Stoch_K, and Volume."""
    # Determine which indicators need separate subplots
    # Indicators with 0-100 scale or different units need separate subplots
    has_rsi = 'RSI' in selected_indicators and 'RSI' in ticker_data.columns
    has_macd = 'MACD_Hist' in selected_indicators and 'MACD_Hist' in ticker_data.columns
    has_stoch = 'Stoch_K' in selected_indicators and 'Stoch_K' in ticker_data.columns
    has_adx = 'ADX' in selected_indicators and 'ADX' in ticker_data.columns
    # Volume is always included
    has_volume = 'Volume' in ticker_data.columns
    
    # Count number of subplots needed and build titles list
    n_subplots = 1  # Main price chart
    row_heights = []
    subplot_titles = [f"{ticker} Price Chart"]
    
    # Determine row assignments (order: Price, RSI, MACD, ADX, Stoch_K, Volume)
    row_assignments = {'price': 1}
    current_row = 2
    
    if has_rsi:
        n_subplots += 1
        row_heights.append(0.12)
        subplot_titles.append("RSI")
        row_assignments['rsi'] = current_row
        current_row += 1
    if has_macd:
        n_subplots += 1
        row_heights.append(0.12)
        subplot_titles.append("MACD Histogram")
        row_assignments['macd'] = current_row
        current_row += 1
    if has_adx:
        n_subplots += 1
        row_heights.append(0.12)
        subplot_titles.append("ADX")
        row_assignments['adx'] = current_row
        current_row += 1
    if has_stoch:
        n_subplots += 1
        row_heights.append(0.12)
        subplot_titles.append("Stochastic K")
        row_assignments['stoch'] = current_row
        current_row += 1
    if has_volume:
        n_subplots += 1
        row_heights.append(0.1)
        subplot_titles.append("Volume")
        row_assignments['volume'] = current_row
    
    # Main chart height is remainder
    main_height = 1.0 - sum(row_heights) if row_heights else 0.5
    row_heights = [main_height] + row_heights
    
    # Create subplots
    fig = make_subplots(
        rows=n_subplots, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Main price chart (row 1)
    fig.add_trace(
            go.Candlestick(
                x=ticker_data['date'],
                open=ticker_data['open'],
                high=ticker_data['high'],
                low=ticker_data['low'],
                close=ticker_data['close'],
            name="Price",
        ),
        row=row_assignments['price'], col=1
    )
    
    # Add overlay indicators on price chart (SMAs, Bollinger Bands, ATR)
    # Note: ADX is 0-100 scale, so it gets its own subplot
    price_row = row_assignments['price']
    if 'SMA20' in selected_indicators and 'SMA20' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['SMA20'], 
                      name='SMA20', line=dict(color='blue', width=1)),
            row=price_row, col=1
        )
    if 'SMA60' in selected_indicators and 'SMA60' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['SMA60'], 
                      name='SMA60', line=dict(color='orange', width=1)),
            row=price_row, col=1
        )
    if 'SMA200' in selected_indicators and 'SMA200' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['SMA200'], 
                      name='SMA200', line=dict(color='red', width=1)),
            row=price_row, col=1
        )
    if 'BB_Upper' in selected_indicators and 'BB_Upper' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['BB_Upper'], 
                      name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
            row=price_row, col=1
        )
    if 'BB_Lower' in selected_indicators and 'BB_Lower' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['BB_Lower'], 
                      name='BB Lower', line=dict(color='gray', width=1, dash='dash')),
            row=price_row, col=1
        )
    if 'ATR' in selected_indicators and 'ATR' in ticker_data.columns:
        # ATR is in price units, so it can be overlaid on price chart
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['ATR'], 
                      name='ATR', line=dict(color='purple', width=1)),
            row=price_row, col=1
        )
    # RSI subplot
    if has_rsi:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['RSI'], 
                      name='RSI', line=dict(color='orange', width=1)),
            row=row_assignments['rsi'], col=1
        )
        # Add RSI reference lines (70 overbought, 30 oversold)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=row_assignments['rsi'], col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=row_assignments['rsi'], col=1)
        fig.update_yaxes(title_text="RSI (0-100)", range=[0, 100], row=row_assignments['rsi'], col=1)
    
    # MACD subplot
    if has_macd:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['MACD_Hist'], 
                      name='MACD Hist', line=dict(color='blue', width=1)),
            row=row_assignments['macd'], col=1
        )
        # Add zero line for MACD
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row_assignments['macd'], col=1)
        fig.update_yaxes(title_text="MACD Histogram", row=row_assignments['macd'], col=1)
    
    # ADX subplot (0-100 scale)
    if has_adx:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['ADX'], 
                      name='ADX', line=dict(color='brown', width=1)),
            row=row_assignments['adx'], col=1
        )
        # Add ADX reference lines (25 strong trend, 50 very strong trend)
        fig.add_hline(y=25, line_dash="dash", line_color="green", opacity=0.5, row=row_assignments['adx'], col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5, row=row_assignments['adx'], col=1)
        fig.update_yaxes(title_text="ADX (0-100)", range=[0, 100], row=row_assignments['adx'], col=1)
    
    # Stochastic K subplot
    if has_stoch:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['Stoch_K'], 
                      name='Stoch_K', line=dict(color='green', width=1)),
            row=row_assignments['stoch'], col=1
        )
        # Add Stochastic K reference lines (80 overbought, 20 oversold)
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=row_assignments['stoch'], col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=row_assignments['stoch'], col=1)
        fig.update_yaxes(title_text="Stoch_K (0-100)", range=[0, 100], row=row_assignments['stoch'], col=1)
    
    # Volume subplot (always included)
    if has_volume:
        fig.add_trace(
            go.Bar(x=ticker_data['date'], y=ticker_data['Volume'], 
                  name='Volume', marker_color='lightblue', opacity=0.6),
            row=row_assignments['volume'], col=1
        )
        fig.update_yaxes(title_text="Volume", row=row_assignments['volume'], col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price Chart with Technical Indicators",
        height=600 + (150 * (n_subplots - 1)),  # Add height for each subplot
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True,
    )
    
    # Update x-axis labels (only on bottom subplot)
    fig.update_xaxes(title_text="Date", row=n_subplots, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    return fig
    
def main():
    st.title("Stock Clustering Analysis")
    
    # Dataset Overview Section
    st.header("Dataset Overview")
    st.write("""
    **What is it?** This dataset contains daily stock price data (open, high, low, close, volume) for 50 major US stocks 
    with high market capitalization over the past 2 years.
    
    **Why we use it:** We use this historical price data to compute technical indicators and risk-return characteristics 
    that help identify patterns and similarities between stocks for investment decision-making.
    
    **Where did we get it:** Data is fetched from Polygon.io API and stored in a local SQLite database.
    """)
    
    # Research Question
    st.header("Research Question")
    st.write("""
    **Objective:** Cluster stocks based on their risk and return characteristics to identify groups of stocks with similar 
    behavioral patterns. This clustering helps investors make informed decisions by grouping stocks with comparable risk profiles, 
    momentum patterns, volatility, and technical indicators.
    """)
    
    db_path = st.text_input("Insert your sql.db path", value="sql.db") 
    if not db_path:
        st.warning("Please provide a valid SQLite database file path.")
        return
    
    try:
        raw_bars = load_raw_bars(db_path)
        feats = load_features(db_path)
        features_scaled = prepare_features(feats)
        price_df = load_prices(db_path, tickers=features_scaled.index.tolist()) 
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    n_cluster = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=4, step=1)
    cluster_method = st.sidebar.selectbox("Clustering method", options=["KMeans", "PAM", "Hierarchical"], index=0, key="method_select")  
    
    # Technical indicator options for visualization
    indicator_options = ['RSI', 'ATR', 'ADX', 'MACD_Hist', 'Stoch_K', 'SMA20', 'SMA60', 'SMA200', 'BB_Upper', 'BB_Lower', 'Volume']
    st.sidebar.header("Technical Indicators for Charts")
    selected_indicators = st.sidebar.multiselect(
        "Select indicators to display",
        options=indicator_options,
        default=['SMA20', 'SMA60'],
        key="indicator_select",
    )
    
    # Main tabs
    tab_overview, tab_analysis, tab_cluster_results = st.tabs(["Overview", "Data Analysis", "Clustering Results"])
    
    with tab_overview:
        st.subheader("Data Before Aggregation")
        st.write("**Raw daily bars data:** This is the original time-series data before feature computation and aggregation.")
        st.dataframe(raw_bars.head(20))
        
        st.subheader("Data After Aggregation")
        st.write("**Aggregated features for clustering:** These are summary statistics (means, ratios, etc.) computed from the raw data. "
                "Each row represents one stock with aggregated metrics. This aggregated data is used for clustering.")
        st.dataframe(feats.head(20))
        
        st.subheader("Scaled Features (Used for Clustering)")
        st.write("**Normalized features:** The aggregated features are transformed and scaled to ensure all variables are on a similar scale, "
                "which is essential for clustering algorithms.")
        st.dataframe(features_scaled.head(20))
        
        # Technical Indicator Descriptions
        if selected_indicators:
            st.subheader("Selected Technical Indicator Descriptions")
            for indicator in selected_indicators:
                with st.expander(f"{indicator}"):
                    st.markdown(get_indicator_description(indicator))
        
        # Candlestick Chart with Indicators
        if not price_df.empty:
            st.subheader("Candlestick Chart with Technical Indicators")
            tick = st.selectbox("Select ticker", options=price_df['ticker'].unique(), key="ticker_select")
            data = price_df[price_df['ticker'] == tick].sort_values(by='date')
            
            if not data.empty:
                col1, col2 = st.columns(2)
                with col1:
                    start = st.date_input("Start date", value=data['date'].min(),
                                        min_value=data['date'].min(),
                                        max_value=data['date'].max(), key="start_date")
                with col2:
                    end = st.date_input("End date", value=data['date'].max(),
                                        min_value=data['date'].min(),
                                        max_value=data['date'].max(), key="end_date")
                
                data_filtered = data[(data['date'] >= pd.to_datetime(start)) & (data['date'] <= pd.to_datetime(end))]
                
                # Always compute indicators (Volume is always included, and selected indicators may include others)
                data_with_indicators = compute_technical_indicators(data_filtered)
                
                # Volume is always included in the chart, so add it to selected_indicators if not already there
                display_indicators = selected_indicators.copy() if selected_indicators else []
                if 'Volume' not in display_indicators:
                    display_indicators.append('Volume')
                
                st.plotly_chart(candlestick_chart(data_with_indicators, tick, display_indicators), use_container_width=True)
    
    with tab_analysis:
        st.subheader("Correlation Matrix")
        st.write("""
        **What is correlation?** Correlation measures the linear relationship between two variables, ranging from -1 (perfect negative) 
        to +1 (perfect positive). A value near 0 indicates no linear relationship.
        
        **Why is it needed for clustering?** High correlation between features means they provide similar information, which can: 
        (1) make clustering less effective, (2) introduce redundancy, and (3) bias results. Understanding correlations helps 
        select meaningful features for clustering.
        """)
        
        # Use all aggregated features (carefully chosen)
        features_for_corr = feats
        
        if len(features_for_corr.columns) > 1:
            corr_matrix = features_for_corr.corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 8},
                colorbar=dict(title="Correlation")
            ))
            fig_corr.update_layout(
                title="Feature Correlation Matrix",
                height=600,
                template="plotly_white"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.write("Select at least 2 features to display correlation matrix.")
        
        st.subheader("Feature Skewness")
        st.write("""
        **What is skewness?** Skewness measures the asymmetry of a distribution. Positive skewness indicates a longer tail on the right 
        (values cluster to the left), negative skewness indicates a longer tail on the left (values cluster to the right).
        
        **Why is it important?** Highly skewed features (|skewness| > 2) can negatively impact clustering algorithms that assume 
        normal distributions. We apply power transformations to normalize skewed features before clustering.
        """)
        
        skewness = feats.skew()
        skew_df = pd.DataFrame({
            'Feature': skewness.index,
            'Skewness': skewness.values
        }).sort_values('Skewness')
        
        fig_skew = px.bar(skew_df, x='Feature', y='Skewness', 
                         title="Feature Skewness Distribution",
                         color='Skewness',
                         color_continuous_scale='RdBu',
                         color_continuous_midpoint=0)
        fig_skew.update_layout(height=500, template="plotly_white")
        fig_skew.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Upper threshold (2)")
        fig_skew.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="Lower threshold (-2)")
        st.plotly_chart(fig_skew, use_container_width=True)
        st.dataframe(skew_df)
    
    with tab_cluster_results:
        st.subheader(f"Clustering Results using {cluster_method}")
        
        # Use all features (carefully chosen after aggregation)
        features_for_clustering = features_scaled
        
        clusters_df, metrics, model = run_clustering(
            method=cluster_method,
            n_clusters=n_cluster,
            features_scaled=features_for_clustering,
        )
        
        # Display clustering metrics
        st.subheader("Clustering Quality Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Silhouette Score", 
                f"{metrics['silhouette']:.4f}",
                help="Higher is better. Range: -1 to 1. Measures how similar objects are to their own cluster vs other clusters."
            )
        with col2:
            st.metric(
                "Davies-Bouldin Score", 
                f"{metrics['davies_bouldin']:.4f}",
                help="Lower is better. Measures the average similarity ratio between clusters."
            )
        with col3:
            st.metric(
                "Calinski-Harabasz Score", 
                f"{metrics['calinski_harabasz']:.2f}",
                help="Higher is better. Ratio of between-cluster dispersion to within-cluster dispersion."
            )
        
        # Cluster distribution
        st.subheader("Cluster Distribution")
        cluster_counts = clusters_df.groupby(by="Cluster").count()
        cluster_counts.columns = ['Number of Stocks']
        
        # Bar chart of cluster sizes
        fig_counts = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts['Number of Stocks'],
            labels={'x': 'Cluster', 'y': 'Number of Stocks'},
            title="Stocks per Cluster"
        )
        fig_counts.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_counts, use_container_width=True)
        st.dataframe(cluster_counts)
        
        # PCA Visualization
        st.subheader("2D Cluster Visualization (PCA)")
        st.write("**PCA (Principal Component Analysis)** projects the high-dimensional feature space onto 2D for visualization. "
                "This helps visualize how well-separated the clusters are.")
        
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_for_clustering)
        
        pca_df = pd.DataFrame({
            'PC1': features_2d[:, 0],
            'PC2': features_2d[:, 1],
            'Cluster': clusters_df['Cluster'].astype(str),
            'Ticker': clusters_df['Ticker']
        })
        
        explained_var = pca.explained_variance_ratio_.sum()
        fig_pca = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=['Ticker'],
            title=f"PCA Visualization (Explained Variance: {explained_var:.2%})",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pca.update_layout(template="plotly_white", height=500)
        fig_pca.update_xaxes(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        fig_pca.update_yaxes(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Cluster characteristics (average features per cluster)
        st.subheader("Cluster Characteristics")
        st.write("**Average feature values per cluster:** This helps interpret what each cluster represents.")
        
        # Merge clusters with original features
        clusters_with_features = clusters_df.set_index('Ticker').join(feats, how='left')
        
        # Calculate mean features per cluster
        cluster_means = clusters_with_features.groupby('Cluster')[feats.columns].mean()
        
        # Display cluster means
        st.dataframe(cluster_means.round(4))
        
        # Visualize key features by cluster
        if len(feats.columns) > 0:
            st.subheader("Feature Comparison Across Clusters")
            selected_features_for_viz = st.multiselect(
                "Select features to visualize",
                options=feats.columns.tolist(),
                default=feats.columns[:min(6, len(feats.columns))].tolist(),
                key="cluster_viz_features"
            )
            
            if selected_features_for_viz:
                cluster_means_selected = cluster_means[selected_features_for_viz]
                
                # Create heatmap
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=cluster_means_selected.values,
                    x=cluster_means_selected.columns,
                    y=cluster_means_selected.index.astype(str),
                    colorscale='RdYlBu_r',
                    text=cluster_means_selected.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Mean Feature Value")
                ))
                fig_heatmap.update_layout(
                    title="Average Feature Values by Cluster",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Cluster assignments table
        st.subheader("Cluster Assignments")
        st.dataframe(clusters_df.sort_values('Cluster'), use_container_width=True)
    
if __name__ == "__main__":
    main()