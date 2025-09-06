import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- Helper function to load and cache data for Forecasting Dashboard ---
@st.cache_data
def load_data(file_path):
    """
    Loads data from a CSV file and caches it for improved performance.
    """
    try:
        # Construct the absolute path to the file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.join(script_dir, file_path)
        
        df = pd.read_csv(absolute_path)
        df['PERIOD_START_TIME'] = pd.to_datetime(df['PERIOD_START_TIME'])
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        st.warning("Please ensure the required CSV file is in the 'src/data/' directory relative to your script.")
        return None

# --- Helper function to process data for Anomaly Detection Dashboard ---
def process_data(df, label):
    """
    Processes a dataframe to detect and visualize anomalies.
    """
    st.subheader(f"Anomaly Detection for {label.upper()}")
    
    value_col = f"{label}_value"
    residual_col = f"{label}_residual"

    # Custom threshold
    std_dev = df[residual_col].std()
    multiplier = st.slider(f"Set threshold multiplier for {label.upper()}", 0.5, 5.0, 3.0, 0.1, key=f"{label}_slider")
    threshold = multiplier * std_dev
    
    df['is_anomaly'] = df[residual_col].abs() > threshold

    # Plot
    fig = px.line(df, x='timestamp', y=value_col, title=f"{label.upper()} Value with Anomalies")
    anomalies = df[df['is_anomaly']]
    fig.add_scatter(
        x=anomalies['timestamp'], 
        y=anomalies[value_col], 
        mode='markers', 
        marker=dict(color='red', size=8), 
        name='Anomalies'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.head())
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"📥 Download {label.upper()} CSV with Anomalies",
        csv,
        f"{label}_anomalies.csv",
        "text/csv"
    )

    return df

# --- Dashboard 1: Forecasting Dashboard ---
def forecasting_dashboard():
    st.header("Time Series Forecasting Dashboard")
    st.markdown("""
    This dashboard visualizes the performance of the **LSTM** and **XGBoost** models for time series forecasting.
    Use the interactive widgets to compare model predictions.
    """)

    final_df = load_data("../data/final_result.csv")

    if final_df is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("Forecasting Filters")
        
        # Variable selector
        variable_options = ['pl', 'pd', 'pdv']
        selected_variable = st.sidebar.selectbox("Select a Variable to Plot", variable_options, key="forecast_var_select")

        # Model selector
        model_options = ['Both Models', 'LSTM', 'XGBoost']
        selected_model = st.sidebar.selectbox("Select a Model for Comparison", model_options, key="forecast_model_select")

        # Dynamic column selection for plotting
        y_cols = [selected_variable]
        if selected_model == 'LSTM' or selected_model == 'Both Models':
            y_cols.append(f"{selected_variable}_lstm")
        if selected_model == 'XGBoost' or selected_model == 'Both Models':
            y_cols.append(f"{selected_variable}_xgb")

        fig = px.line(final_df, 
                      x='PERIOD_START_TIME', 
                      y=y_cols,
                      title=f'{selected_variable.upper()} Value: Actual vs. Predictions ({selected_model})',
                      labels={'value': 'Value', 'variable': 'Model'},
                      markers=True)
        
        new_names = {selected_variable: 'Actual'}
        if f"{selected_variable}_lstm" in y_cols:
            new_names[f"{selected_variable}_lstm"] = "LSTM Prediction"
        if f"{selected_variable}_xgb" in y_cols:
            new_names[f"{selected_variable}_xgb"] = "XGBoost Prediction"
        
        fig.for_each_trace(lambda t: t.update(name = new_names[t.name]))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Raw Data Table")
        st.dataframe(final_df.style.highlight_max(axis=0))

# --- Dashboard 2: Anomaly Detection Dashboard ---
def anomaly_dashboard():
    st.header("QoS Anomaly Detection Dashboard")
    st.markdown("""
    This dashboard visualizes the detection of anomalies in QoS metrics.
    """)

    # --- Implicitly provided data ---
    date_rng = pd.date_range(start='2023-01-01', end='2023-01-02', freq='H')
    n_points = len(date_rng)
    
    pd_values = np.sin(np.linspace(0, 10, n_points)) * 50 + 100
    pd_residuals = np.random.normal(0, 2, n_points)
    pd_residuals[15:17] += 25  
    pd_residuals[30:32] -= 20  
    pd_df = pd.DataFrame({'timestamp': date_rng, 'pd_value': pd_values, 'pd_residual': pd_residuals})
    
    pdv_values = np.cos(np.linspace(0, 10, n_points)) * 30 + 70
    pdv_residuals = np.random.normal(0, 1.5, n_points)
    pdv_residuals[5:7] += 15
    pdv_residuals[20:22] -= 10
    pdv_df = pd.DataFrame({'timestamp': date_rng, 'pdv_value': pdv_values, 'pdv_residual': pdv_residuals})
    
    pl_values = np.random.normal(50, 10, n_points)
    pl_residuals = np.random.normal(0, 3, n_points)
    pl_residuals[10:12] += 20
    pl_df = pd.DataFrame({'timestamp': date_rng, 'pl_value': pl_values, 'pl_residual': pl_residuals})

    tabs = st.tabs(["📦 PD", "📶 PDV", "📉 PL"])
    with tabs[0]:
        process_data(pd_df, 'pd')
    with tabs[1]:
        process_data(pdv_df, 'pdv')
    with tabs[2]:
        process_data(pl_df, 'pl')

    st.markdown("---")
    st.subheader("📈 Combined View")
    
    combined_df = pd.DataFrame({
        'timestamp': pd_df['timestamp'],
        'PD': pd_df['pd_value'],
        'PDV': pdv_df['pdv_value'],
        'PL': pl_df['pl_value']
    })
    
    melted_df = combined_df.melt(id_vars='timestamp', var_name='Metric', value_name='Value')
    fig_all = px.line(melted_df, x='timestamp', y='Value', color='Metric', title="PD, PDV, PL Over Time")
    st.plotly_chart(fig_all, use_container_width=True)

# --- Main app logic to select dashboard ---
st.set_page_config(page_title="Network Dashboards", layout="wide")
st.sidebar.title("Dashboard Selector")
dashboard_selection = st.sidebar.radio(
    "Choose a Dashboard",
    ('Forecasting Dashboard', 'Anomaly Detection Dashboard')
)

if dashboard_selection == 'Forecasting Dashboard':
    forecasting_dashboard()
else:
    anomaly_dashboard()
