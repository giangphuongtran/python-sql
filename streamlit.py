import sqlite3
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import PowerTransformer

from features import compute_features, compute_technical_indicators

warnings.filterwarnings("ignore")


@st.cache_data
def load_features(db_path: str):
    feats = compute_features(db_path)
    return feats

def StandardScaler(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(axis=0, ddof=1)
    return (df - mu) / sd.replace(0, np.nan)

def prepare_features(feats: pd.DataFrame):
    """Transform and scale features for clustering."""
    features = feats.copy()
    skew = features.skew()
    skewed_cols = skew[abs(skew) > 2].index.tolist()
    features_transformed = features.copy()
    for col in skewed_cols:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        features_transformed[col] = pt.fit_transform(features[[col]]).flatten()
    scaled = StandardScaler(features_transformed)
    return scaled
    
@st.cache_data
def run_clustering(
    method: str,
    n_clusters: int,
    features_scaled: pd.DataFrame,
    n_pca_components: float = 0.65,
):
    """
    Run clustering with PCA reduction.
    Uses appropriate distance metrics for each method when calculating silhouette scores.
    """
    tickers = features_scaled.index.tolist()
    
    pca = PCA(n_components=n_pca_components, random_state=42)
    features_for_clustering = pd.DataFrame(
        pca.fit_transform(features_scaled),
        index=features_scaled.index
    )
    
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(features_for_clustering)
        linkage_matrix = None
        distance_metric = "euclidean"
    elif method == "PAM":
        model = KMedoids(n_clusters=n_clusters, metric="euclidean", random_state=42)
        labels = model.fit_predict(features_for_clustering)
        linkage_matrix = None
        distance_metric = "euclidean"
    else:  # Hierarchical
        X = features_for_clustering.values
        corr = np.corrcoef(X)
        dist_sq = 1.0 - corr
        np.fill_diagonal(dist_sq, 0.0)

        condensed = squareform(dist_sq, checks=False)
        linkage_matrix = linkage(condensed, method="average")

        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust") - 1
        model = None
        distance_metric = "precomputed_cos"

    
    clusters_df = pd.DataFrame({"Ticker": tickers, "Cluster": labels + 1})
    
    if distance_metric == "precomputed_cos":
        sil_samples = silhouette_samples(dist_sq, labels, metric="precomputed")
        sil = silhouette_score(dist_sq, labels, metric="precomputed")
    elif distance_metric == "cosine":
        dist_matrix = pdist(features_for_clustering, metric="cosine")
        dist_matrix_sq = squareform(dist_matrix)
        sil_samples = silhouette_samples(dist_matrix_sq, labels, metric="precomputed")
        sil = silhouette_score(dist_matrix_sq, labels, metric="precomputed")
    else:
        sil_samples = silhouette_samples(features_for_clustering, labels, metric=distance_metric)
        sil = silhouette_score(features_for_clustering, labels, metric=distance_metric)

    
    clusters_df['Silhouette_Width'] = sil_samples
    
    db = davies_bouldin_score(features_for_clustering, labels)
    ch = calinski_harabasz_score(features_for_clustering, labels)
    
    metrics = {
        'silhouette': sil,
        'davies_bouldin': db,
        'calinski_harabasz': ch
    }
    
    return clusters_df, metrics, model, linkage_matrix, features_for_clustering

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
        'RSI': "**RSI (Relative Strength Index)**: Momentum oscillator (0-100) measuring speed/magnitude of price changes."
               "Calculated as: 100 - (100 / (1 + RS)), where RS = average gain / average loss over 14 periods."
               "Used to identify overbought (>70) or oversold (<30) conditions.",
        'ATR': "**ATR (Average True Range)**: Volatility measure indicating average trading range."
               "Calculated as 14-period moving average of True Range (max of high-low, |high-prev_close|, |low-prev_close|)."
               "Used to assess market volatility and set stop-loss levels.",
        'ADX': "**ADX (Average Directional Index)**: Trend strength indicator (0-100)."
               "Based on comparison of upward vs downward price movement relative to True Range."
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
    # Indicators with 0-100 scale or different units need separate subplots
    has_rsi = 'RSI' in selected_indicators and 'RSI' in ticker_data.columns
    has_macd = 'MACD_Hist' in selected_indicators and 'MACD_Hist' in ticker_data.columns
    has_stoch = 'Stoch_K' in selected_indicators and 'Stoch_K' in ticker_data.columns
    has_adx = 'ADX' in selected_indicators and 'ADX' in ticker_data.columns
    has_volume = 'Volume' in ticker_data.columns
    
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
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['ATR'], 
                      name='ATR', line=dict(color='purple', width=1)),
            row=price_row, col=1
        )
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
    if has_macd:
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['MACD_Hist'], 
                      name='MACD Hist', line=dict(color='blue', width=1)),
            row=row_assignments['macd'], col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row_assignments['macd'], col=1)
        fig.update_yaxes(title_text="MACD Histogram", row=row_assignments['macd'], col=1)
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
    # Volume subplot
    if has_volume:
        fig.add_trace(
            go.Bar(x=ticker_data['date'], y=ticker_data['Volume'], 
                  name='Volume', marker_color='darkblue', opacity=0.6,
                  hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'),
            row=row_assignments['volume'], col=1
        )
        fig.update_yaxes(
            title_text="Volume", 
            row=row_assignments['volume'], 
            col=1,
            tickformat=',.0f'
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price Chart with Technical Indicators",
        height=600 + (150 * (n_subplots - 1)),
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=True,
    )
    fig.update_xaxes(title_text="Date", row=n_subplots, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    return fig
    
def main():
    st.title("Stock Clustering Analysis")
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Overview"
    
    # Dataset Overview Section
    st.header("Dataset Overview")
    st.write("""
    The dataset contains daily stock price data (open, high, low, close, volume) for 50 major US stocks 
    with high market capitalization over the past 2 years.
    
    Historical price data was used to compute technical indicators and risk-return characteristics 
    that help identify patterns and similarities between stocks for investment decision-making.
    
    **Datasource**: Data is fetched from Polygon.io API and stored in a local SQLite database.
    """)
    
    # Research Question
    st.header("Objective")
    st.write("""
    Cluster stocks based on their risk and return characteristics to identify groups of stocks with similar 
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
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    n_cluster = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=4, step=1)
    cluster_method = st.sidebar.selectbox("Clustering method", options=["KMeans", "PAM", "Hierarchical"], index=0, key="method_select")  
    
    # Technical indicator options for visualization
    indicator_options = ['RSI', 'ATR', 'ADX', 'MACD_Hist', 'Stoch_K', 'SMA20', 'SMA60', 'SMA200', 'BB_Upper', 'BB_Lower']
    st.sidebar.header("Technical Indicators for Charts")
    selected_indicators = st.sidebar.multiselect(
        "Select indicators to display",
        options=indicator_options,
        default=['SMA20', 'SMA60'],
        key="indicator_select",
    )
    
    tab_names = ["Overview", "Data Analysis", "Clustering Results"]
    default_tab_idx = tab_names.index(st.session_state.current_tab) if st.session_state.current_tab in tab_names else 0
    tab_overview, tab_analysis, tab_cluster_results = st.tabs(tab_names)
    
    with tab_overview:
        st.subheader("Original Daily Bars Data")
        st.write("Original time-series data from API.")
        raw_bars_display = raw_bars.copy()
        raw_bars_display["date"] = pd.to_datetime(raw_bars_display["date"]).dt.date
        st.dataframe(raw_bars_display.head(5))
        
        st.subheader("Aggregated Features for Clustering")
        st.write("These are summary statistics (means, ratios, etc.) computed from the raw data. "
                "Each row represents one stock with aggregated metrics.")
        st.dataframe(feats.head(5))
        
        st.subheader("Scaled Features (Used for Clustering)")
        st.write("The aggregated features are transformed and scaled to ensure all variables are on a similar scale, "
                "which is essential for clustering algorithms.")
        st.dataframe(features_scaled.head(5))
        
        # Technical Indicator Descriptions
        if selected_indicators:
            st.subheader("Technical Indicator Descriptions")
            for indicator in selected_indicators:
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
                
                data_filtered = data[(data['date'] >= start) & (data['date'] <= end)]
                data_with_indicators = compute_technical_indicators(data_filtered)
                display_indicators = selected_indicators.copy() if selected_indicators else []
                if 'Volume' not in display_indicators:
                    display_indicators.append('Volume')
                
                st.plotly_chart(candlestick_chart(data_with_indicators, tick, display_indicators), use_container_width=True)
    
    with tab_analysis:
        st.subheader("Correlation Matrix")
        st.write("""
        Correlation measures the linear relationship between two variables, ranging from -1 (perfect negative) 
        to +1 (perfect positive). A value near 0 indicates no linear relationship.
        
        **Why is it needed for clustering?** High correlation between features means they provide similar information, which can: 
        (1) make clustering less effective
        (2) introduce redundancy
        (3) bias results
        """)
        
        features_for_corr = feats
        corr_matrix = features_for_corr.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
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
        
        st.subheader("Feature Skewness")
        st.write("""
        Skewness measures the asymmetry of a distribution. Positive skewness indicates a longer tail on the right 
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
        fig_skew.add_hline(y=2, line_dash="dash", line_color="red")
        fig_skew.add_hline(y=-2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_skew, use_container_width=True)
        st.dataframe(skew_df[skew_df.Skewness.abs() > 2].reset_index(drop=True).round(2))
        
        st.subheader("PCA Analysis: Scree Plot & Cumulative Variance")
        st.write("""
        **Principal Component Analysis (PCA)** reduces dimensionality while preserving variance.
        The scree plot shows the variance explained by each component, and the cumulative plot 
        shows how much variance is captured as we add more components.
        
        **Conclusion**: We use 3 principal components for clustering visualization, which captures 
        approximately 67% of the variance while maintaining interpretability.
        """)
        
        # Perform PCA with all components
        pca_full = PCA()
        pca_full.fit(features_scaled)
        
        n_components = min(10, len(features_scaled.columns))
        explained_var = pca_full.explained_variance_ratio_[:n_components] * 100
        cumulative_var = np.cumsum(explained_var)
        
        # Create combined scree and cumulative variance plot
        fig_pca_analysis = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Scree plot (bars)
        fig_pca_analysis.add_trace(
            go.Bar(
                x=list(range(1, n_components + 1)),
                y=explained_var,
                name="Variance Explained (%)",
                marker_color='lightblue',
                text=[f"{v:.1f}%" for v in explained_var],
                textposition='outside'
            ),
            secondary_y=False
        )
        
        # Cumulative variance (line)
        fig_pca_analysis.add_trace(
            go.Scatter(
                x=list(range(1, n_components + 1)),
                y=cumulative_var,
                mode='lines+markers',
                name="Cumulative Variance (%)",
                line=dict(color='red', width=2),
                marker=dict(size=8),
                text=[f"{v:.1f}%" for v in cumulative_var],
                textposition='top center'
            ),
            secondary_y=True
        )
        
        # Add vertical line at 3 components
        fig_pca_analysis.add_vline(
            x=3, 
            line_dash="dash", 
            line_color="green", 
            annotation_text="Selected: 3 components",
            annotation_position="top"
        )
        
        fig_pca_analysis.update_xaxes(title_text="Principal Components")
        fig_pca_analysis.update_yaxes(
            title_text="Percentage of Variance Explained", 
            secondary_y=False,
            range=[0, 100]
        )
        fig_pca_analysis.update_yaxes(
            title_text="Cumulative Variance Explained %", 
            secondary_y=True,
            range=[0, 100]
        )
        
        fig_pca_analysis.update_layout(
            title="Figure 2: Scree Plot & Cumulative Variance",
            height=500,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig_pca_analysis, use_container_width=True)
        
        # Display variance explained table
        pca_summary = pd.DataFrame({
            'Component': range(1, n_components + 1),
            'Variance Explained (%)': explained_var,
            'Cumulative Variance (%)': cumulative_var
        })
        st.dataframe(pca_summary.round(2).head(n_components + 3))
    
    with tab_cluster_results:
        st.subheader(f"Clustering Results using {cluster_method}")
        
        pca_temp = PCA()
        pca_temp.fit(features_scaled)
        cumulative_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_pca_components = np.where(cumulative_var >= 0.65)[0][0] + 1
        
        # Use PCA-reduced features for clustering
        clusters_df, metrics, model, linkage_matrix, features_pca_reduced = run_clustering(
            method=cluster_method,
            n_clusters=n_cluster,
            features_scaled=features_scaled,
            n_pca_components=n_pca_components,
        )
        
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
        
        # Dendrogram for Hierarchical Clustering
        if cluster_method == "Hierarchical" and linkage_matrix is not None:
            st.subheader("Dendrogram")
            st.write("The dendrogram shows the hierarchical clustering structure.")
            
            from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
            threshold_dist = linkage_matrix[-(n_cluster - 1), 2] if n_cluster > 1 else 0

            fig_dendro_data = scipy_dendrogram(
                linkage_matrix,
                labels=features_pca_reduced.index.tolist(),
                leaf_rotation=90,
                no_plot=True,
                color_threshold=threshold_dist
            )
            
            icoord = fig_dendro_data['icoord']
            dcoord = fig_dendro_data['dcoord']
            color_list = fig_dendro_data['color_list']
            ivl = fig_dendro_data['ivl']
            
            fig_dendro = go.Figure()
            
            # Convert matplotlib colors to plotly-compatible colors
            def convert_matplotlib_color(color):
                """Convert matplotlib color codes (like 'C1', 'C2') to valid plotly colors."""
                # Map common matplotlib color codes
                color_map = {
                    'C0': '#1f77b4',  # blue
                    'C1': '#ff7f0e',  # orange
                    'C2': '#2ca02c',  # green
                    'C3': '#d62728',  # red
                    'C4': '#9467bd',  # purple
                    'C5': '#8c564b',  # brown
                    'C6': '#e377c2',  # pink
                    'C7': '#7f7f7f',  # gray
                    'C8': '#bcbd22',  # olive
                    'C9': '#17becf',  # cyan
                }
                if color in color_map:
                    return color_map[color]
                # Try to convert using matplotlib
                try:
                    rgb = mcolors.to_rgb(color)
                    return f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                except:
                    # Default to gray if conversion fails
                    return 'gray'
            
            # Plot dendrogram segments
            for xs, ys, color in zip(icoord, dcoord, color_list):
                plotly_color = convert_matplotlib_color(color)
                fig_dendro.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines',
                        line=dict(color=plotly_color, width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
            
            leaf_positions = [5 + 10*i for i in range(len(ivl))]
            fig_dendro.update_layout(
                title="Hierarchical Clustering Dendrogram",
                height=600,
                template="plotly_white",
                xaxis=dict(
                    tickmode='array',
                    tickvals=leaf_positions,
                    ticktext=ivl,
                    tickangle=90,
                    tickfont=dict(size=8)  # Smaller font for better fit
                ),
                xaxis_title="Stocks",
                yaxis_title="Distance",
                showlegend=False,
                margin=dict(b=100)  # Extra bottom margin for rotated labels
            )
            st.plotly_chart(fig_dendro, use_container_width=True)
        
        st.subheader("Cluster Distribution")
        cluster_counts = clusters_df.groupby(by="Cluster")['Ticker'].count().reset_index()
        cluster_counts.columns = ['Cluster', 'Number of Stocks']
        
        fig_counts = px.bar(
            x=cluster_counts['Cluster'].astype(str),
            y=cluster_counts['Number of Stocks'],
            labels={'x': 'Cluster', 'y': 'Number of Stocks'},
            title="Stocks per Cluster"
        )
        fig_counts.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_counts, use_container_width=True)
        st.dataframe(cluster_counts.set_index('Cluster'))
        
        # Silhouette Analysis
        st.subheader(f"Silhouette Analysis: {cluster_method}")
        st.write(f"""
        **Average Silhouette Width**: {metrics['silhouette']:.2f}
        
        Silhouette width ranges from -1 to 1:
        - **Positive values** (close to 1): Stock is well-matched to its cluster
        - **Values near 0**: Stock is on the boundary between clusters
        - **Negative values**: Stock may be assigned to the wrong cluster
        """)
        
        # Create silhouette plot
        silhouette_df = clusters_df.sort_values(['Cluster', 'Silhouette_Width'], ascending=[True, False])
        
        fig_sil = go.Figure()
        
        y_pos = 0
        colors = px.colors.qualitative.Set3
        for i, cluster in enumerate(sorted(silhouette_df['Cluster'].unique())):
            cluster_data = silhouette_df[silhouette_df['Cluster'] == cluster]
            cluster_data = cluster_data.sort_values('Silhouette_Width', ascending=True)
            
            y_positions = list(range(y_pos, y_pos + len(cluster_data)))
            y_pos += len(cluster_data) + 1  # Add gap between clusters
            
            fig_sil.add_trace(
                go.Bar(
                    x=cluster_data['Silhouette_Width'],
                    y=y_positions,
                    orientation='h',
                    name=f'Cluster {cluster}',
                    marker_color=colors[i % len(colors)],
                    text=cluster_data['Ticker'],
                    textposition='outside',
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Silhouette Width: %{x:.3f}<br>' +
                                  'Cluster: ' + str(cluster) + '<extra></extra>'
                )
            )
        
        # Add average line
        fig_sil.add_vline(
            x=metrics['silhouette'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {metrics['silhouette']:.2f}",
            annotation_position="top"
        )
        
        # Highlight negative values
        negative_stocks = silhouette_df[silhouette_df['Silhouette_Width'] < 0]
        if len(negative_stocks) > 0:
            st.warning(f"⚠️ {len(negative_stocks)} stock(s) have negative silhouette scores (poorly clustered): {', '.join(negative_stocks['Ticker'].tolist())}")
        
        fig_sil.update_layout(
            title=f"Figure 10: Silhouette Analysis: {cluster_method} (Average: {metrics['silhouette']:.2f})",
            xaxis_title="Silhouette width Si",
            yaxis_title="",
            template="plotly_white",
            height=max(400, len(silhouette_df) * 15),
            showlegend=True,
            yaxis=dict(showticklabels=False)
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        
        # Display stocks with negative silhouette
        if len(negative_stocks) > 0:
            st.subheader("Stocks with Negative Silhouette Scores")
            st.dataframe(negative_stocks[['Ticker', 'Cluster', 'Silhouette_Width']].sort_values('Silhouette_Width'))
        
        # PCA Visualization with 3 components
        st.subheader("Cluster Visualization (PCA with 3 Components)")
        st.write("**PCA (Principal Component Analysis)** projects the high-dimensional feature space onto 2D/3D for visualization. "
                "Using 3 components captures ~67% of variance. This helps visualize how well-separated the clusters are.")
        
        pca = PCA(n_components=0.65, random_state=42)
        pca.fit(features_scaled)
        features_3d = features_pca_reduced.values
        
        # 2D visualization (PC1 vs PC2)
        pca_df = pd.DataFrame({
            'PC1': features_3d[:, 0],
            'PC2': features_3d[:, 1],
            'PC3': features_3d[:, 2],
            'Cluster': clusters_df['Cluster'].astype(str),
            'Ticker': clusters_df['Ticker'],
            'Silhouette_Width': clusters_df['Silhouette_Width']
        })
        
        explained_var = pca.explained_variance_ratio_[:2].sum()
        
        fig_pca = go.Figure()
        
        for cluster in sorted(pca_df['Cluster'].unique()):
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            fig_pca.add_trace(
                go.Scatter(
                    x=cluster_data['PC1'],
                    y=cluster_data['PC2'],
                    mode='markers+text',
                    name=f'Cluster {cluster}',
                    text=cluster_data['Ticker'],
                    textposition='middle right',
                    textfont=dict(size=8),
                    marker=dict(size=10, opacity=0.7),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'PC1: %{x:.2f}<br>' +
                                  'PC2: %{y:.2f}<br>' +
                                  'Cluster: ' + str(cluster) + '<br>' +
                                  'Silhouette: %{customdata:.3f}<extra></extra>',
                    customdata=cluster_data['Silhouette_Width']
                )
            )
        
        fig_pca.update_layout(
            title=f"Figure 7.1: {cluster_method} (Sil: {metrics['silhouette']:.2f})",
            template="plotly_white",
            height=600,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            showlegend=True
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
        fig_3d = go.Figure()
        for cluster in sorted(pca_df['Cluster'].unique()):
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            fig_3d.add_trace(
                go.Scatter3d(
                    x=cluster_data['PC1'],
                    y=cluster_data['PC2'],
                    z=cluster_data['PC3'],
                    mode='markers+text',
                    name=f'Cluster {cluster}',
                    text=cluster_data['Ticker'],
                    textposition='middle right',
                    marker=dict(size=8, opacity=0.7),
                    hovertemplate='<b>%{text}</b><br>' +
                                'PC1: %{x:.2f}<br>' +
                                'PC2: %{y:.2f}<br>' +
                                'PC3: %{z:.2f}<br>' +
                                'Cluster: ' + str(cluster) + '<extra></extra>'
                )
            )
        fig_3d.update_layout(
            title="3D PCA Visualization",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
            ),
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.subheader("Cluster Characteristics")
        
        clusters_with_features = clusters_df.set_index('Ticker').join(feats, how='left')
        cluster_means = clusters_with_features.groupby('Cluster')[feats.columns].mean()
        
        # Z-score each feature across clusters (standardize)
        cluster_means_z = cluster_means.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        
        feature_groups = {
            "Risk/Return": ["mean_daily_return", "mean_stock_vs_market", "beta_global", "mean_sharpe_20d", "worst_drawdown"],
            "Momentum": ["mean_momentum_20d", "mean_close_vs_sma200", "mean_adx_14", "mean_price_pos_20"],
            "Volatility": ["mean_volatility_20d", "mean_atr_14", "mean_volatility_ratio", "mean_bb_width"],
            "Volume/Liquidity": ["mean_liquidity_20d", "mean_volume_ratio"],
            "Technical": ["mean_rsi_14", "mean_macd_hist", "mean_stoch_k"],
            "Distributional": ["return_skewness", "return_kurtosis"]
        }
        
        # Visualization choice
        viz_choice = st.radio(
            "Choose visualization:",
            ["Heatmap (Grouped by Feature Categories)", "Radar Charts (One per Cluster)"],
            horizontal=True,
            key="cluster_viz_choice"
        )
        
        if viz_choice == "Heatmap (Grouped by Feature Categories)":
            st.write("""
            **Standardized Cluster Means Heatmap:** This heatmap shows z-scored feature values for each cluster, 
            grouped by feature categories. Blue indicates below-average values, red indicates above-average values, 
            and white indicates average values.
            
            The heatmap highlights clear contrasts across clusters, particularly along momentum, beta, and volatility 
            dimensions, supporting the behavioral interpretation of cluster structure.
            """)
            
            # Create ordered feature list by groups with group labels
            ordered_features = []
            group_labels = []
            group_boundaries = []
            current_pos = 0
            
            for group_name, group_features in feature_groups.items():
                available_features = [f for f in group_features if f in cluster_means_z.columns]
                if available_features:
                    group_labels.append((group_name, current_pos))
                    ordered_features.extend(available_features)
                    group_boundaries.append((group_name, current_pos, current_pos + len(available_features)))
                    current_pos += len(available_features)
            
            remaining_features = [f for f in cluster_means_z.columns if f not in ordered_features]
            if remaining_features:
                group_labels.append(("Other", current_pos))
                ordered_features.extend(remaining_features)
                group_boundaries.append(("Other", current_pos, current_pos + len(remaining_features)))
            
            cluster_means_z_ordered = cluster_means_z[ordered_features]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
            z=cluster_means_z_ordered.values,
            x=cluster_means_z_ordered.columns,
            y=[f"Cluster {i}" for i in cluster_means_z_ordered.index],
            colorscale='RdBu',
            zmid=0,
            text=cluster_means_z_ordered.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 9},
                colorbar=dict(
                    title="Z-Score",
                    tickmode="linear",
                    tick0=-2,
                    dtick=1
                ),
                hovertemplate='Cluster: %{y}<br>Feature: %{x}<br>Z-Score: %{z:.2f}<extra></extra>'
            ))
            
            annotations = []
            for group_name, start_idx, end_idx in group_boundaries:
                if start_idx > 0:
                    fig_heatmap.add_vline(
                        x=start_idx - 0.5,
                        line_dash="dash",
                        line_color="gray",
                        line_width=2,
                        opacity=0.7
                    )
                mid_point = (start_idx + end_idx - 1) / 2
                annotations.append(dict(
                    x=mid_point,
                    y=-0.5,
                    text=f"<b>{group_name}</b>",
                    showarrow=False,
                    xref="x",
                    yref="paper",
                    font=dict(size=10, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                ))
            
                fig_heatmap.update_layout(
                title="Standardized Cluster Means Heatmap (Grouped by Feature Categories)",
                height=max(400, len(cluster_means_z_ordered) * 100),
                template="plotly_white",
                xaxis=dict(
                    tickangle=45,
                    tickfont=dict(size=9),
                    side="bottom"
                ),
                yaxis=dict(
                    tickfont=dict(size=11)
                ),
                margin=dict(l=100, r=50, t=80, b=180),
                annotations=annotations
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        else:
            st.write("""
            **Radar Charts:** Each cluster is represented by a radar chart showing z-scored feature values 
            grouped by feature categories. This allows for easy comparison of cluster profiles.
            """)
            
            n_clusters = len(cluster_means_z)
            cols = st.columns(min(2, n_clusters))
            
            for idx, cluster_num in enumerate(sorted(cluster_means_z.index)):
                col = cols[idx % len(cols)]
                
                with col:
                    radar_data = {}
                    for group_name, group_features in feature_groups.items():
                        available_features = [f for f in group_features if f in cluster_means_z.columns]
                        if available_features:
                            group_avg = cluster_means_z.loc[cluster_num, available_features].mean()
                            radar_data[group_name] = group_avg
                    
                    categories = list(radar_data.keys())
                    values = list(radar_data.values())
                    categories_plot = categories + [categories[0]]
                    values_plot = values + [values[0]]
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values_plot,
                        theta=categories_plot,
                        fill='toself',
                        name=f'Cluster {cluster_num}',
                        line=dict(color=px.colors.qualitative.Set3[cluster_num % len(px.colors.qualitative.Set3)])
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                showticklabels=False,
                                ticks=""
                            )),
                        showlegend=False,
                        title=f"Cluster {cluster_num} Profile",
                    height=400,
                    template="plotly_white"
                )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
        
        
        # Comparison Table for All Methods
        st.subheader("Comparison of All Clustering Methods")
        st.write("""
        This table compares all three clustering methods (KMeans, PAM, Hierarchical) using three evaluation metrics:
        - **Silhouette Score (SH)**: Higher is better (range: -1 to 1). Measures how similar objects are to their own cluster vs other clusters.
        - **Davies-Bouldin Score (DB)**: Lower is better. Measures the average similarity ratio between clusters.
        - **Calinski-Harabasz Score (CH)**: Higher is better. Ratio of between-cluster dispersion to within-cluster dispersion.
        """)
        
        comparison_data = []
        methods_to_compare = ["KMeans", "PAM", "Hierarchical"]
        
        for method in methods_to_compare:
            clusters_comp, metrics_comp, _, _, _ = run_clustering(
                method=method,
                n_clusters=n_cluster,
                features_scaled=features_scaled,
                n_pca_components=n_pca_components,
            )
            comparison_data.append({
                'Method': method,
                'Silhouette Score (SH)': metrics_comp['silhouette'],
                'Davies-Bouldin Score (DB)': metrics_comp['davies_bouldin'],
                'Calinski-Harabasz Score (CH)': metrics_comp['calinski_harabasz']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Silhouette Score (SH)'] = comparison_df['Silhouette Score (SH)'].round(4)
        comparison_df['Davies-Bouldin Score (DB)'] = comparison_df['Davies-Bouldin Score (DB)'].round(4)
        comparison_df['Calinski-Harabasz Score (CH)'] = comparison_df['Calinski-Harabasz Score (CH)'].round(2)
        
        # Display the table
        st.dataframe(comparison_df, use_container_width=True)
        
        best_sil = comparison_df.loc[comparison_df['Silhouette Score (SH)'].idxmax(), 'Method']
        best_db = comparison_df.loc[comparison_df['Davies-Bouldin Score (DB)'].idxmin(), 'Method']
        best_ch = comparison_df.loc[comparison_df['Calinski-Harabasz Score (CH)'].idxmax(), 'Method']
        
        st.write(f"**Best Methods:**")
        st.write(f"- **Silhouette Score**: {best_sil} ({comparison_df.loc[comparison_df['Method'] == best_sil, 'Silhouette Score (SH)'].values[0]:.4f})")
        st.write(f"- **Davies-Bouldin Score**: {best_db} ({comparison_df.loc[comparison_df['Method'] == best_db, 'Davies-Bouldin Score (DB)'].values[0]:.4f})")
        st.write(f"- **Calinski-Harabasz Score**: {best_ch} ({comparison_df.loc[comparison_df['Method'] == best_ch, 'Calinski-Harabasz Score (CH)'].values[0]:.2f})")
        
        fig_compare = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Silhouette Score<br>(Higher is Better)', 
                           'Davies-Bouldin Score<br>(Lower is Better)', 
                           'Calinski-Harabasz Score<br>(Higher is Better)'),
            vertical_spacing=0.15  # More space for titles
        )
        
        # Silhouette Score
        fig_compare.add_trace(
            go.Bar(x=comparison_df['Method'], y=comparison_df['Silhouette Score (SH)'],
                  name='Silhouette', marker_color='blue'),
            row=1, col=1
        )
        
        # Davies-Bouldin Score
        fig_compare.add_trace(
            go.Bar(x=comparison_df['Method'], y=comparison_df['Davies-Bouldin Score (DB)'],
                  name='Davies-Bouldin', marker_color='orange'),
            row=1, col=2
        )
        
        # Calinski-Harabasz Score
        fig_compare.add_trace(
            go.Bar(x=comparison_df['Method'], y=comparison_df['Calinski-Harabasz Score (CH)'],
                  name='Calinski-Harabasz', marker_color='green'),
            row=1, col=3
        )
        
        max_sil = comparison_df['Silhouette Score (SH)'].max()
        max_db = comparison_df['Davies-Bouldin Score (DB)'].max()
        max_ch = comparison_df['Calinski-Harabasz Score (CH)'].max()
        
        for i, method in enumerate(comparison_df['Method']):
            sil_val = comparison_df.loc[comparison_df['Method'] == method, 'Silhouette Score (SH)'].values[0]
            y_pos_sil = sil_val + max_sil * 0.03  # Small offset above bar
            fig_compare.add_annotation(
                x=method,
                y=y_pos_sil,
                text=f'{sil_val:.3f}',
                showarrow=False,
                font=dict(size=9, color='black'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1,
                row=1, col=1
            )
            # Davies-Bouldin Score
            db_val = comparison_df.loc[comparison_df['Method'] == method, 'Davies-Bouldin Score (DB)'].values[0]
            y_pos_db = db_val + max_db * 0.03
            fig_compare.add_annotation(
                x=method,
                y=y_pos_db,
                text=f'{db_val:.3f}',
                showarrow=False,
                font=dict(size=9, color='black'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1,
                row=1, col=2
            )
            # Calinski-Harabasz Score
            ch_val = comparison_df.loc[comparison_df['Method'] == method, 'Calinski-Harabasz Score (CH)'].values[0]
            y_pos_ch = ch_val + max_ch * 0.03
            fig_compare.add_annotation(
                x=method,
                y=y_pos_ch,
                text=f'{ch_val:.2f}',
                showarrow=False,
                font=dict(size=9, color='black'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1,
                row=1, col=3
            )
        
        fig_compare.update_layout(
            title="Clustering Method Comparison",
            height=500,  # Increased height to accommodate labels and titles
            template="plotly_white",
            showlegend=False,
            margin=dict(t=100, b=60)  # Extra top margin for subplot titles
        )
        
        fig_compare.update_yaxes(title_text="Score", row=1, col=1)
        fig_compare.update_yaxes(title_text="Score", row=1, col=2)
        fig_compare.update_yaxes(title_text="Score", row=1, col=3)
        fig_compare.update_xaxes(tickangle=0, row=1, col=1)
        fig_compare.update_xaxes(tickangle=0, row=1, col=2)
        fig_compare.update_xaxes(tickangle=0, row=1, col=3)
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
if __name__ == "__main__":
    main()