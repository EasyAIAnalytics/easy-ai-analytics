import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from io import BytesIO
import time
import json
import os
from datetime import datetime, timedelta
from utils.ai_analytics import AIAnalytics

# Page configuration
st.set_page_config(
    page_title="AI-Powered Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title with styling
st.markdown('<h1 class="main-header">AI-Powered Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Harness the power of artificial intelligence to uncover deeper insights from your data</p>', unsafe_allow_html=True)

# Check if API keys are available
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    st.warning("AI features require either an OpenAI or Anthropic API key. Please provide one in the settings to enable these features.")
elif OPENAI_API_KEY:
    st.info("OpenAI API is configured. Note that API usage is subject to rate limits and quotas. If you encounter quota errors, you may need to check your OpenAI account billing status.")
elif ANTHROPIC_API_KEY:
    st.info("Anthropic API is configured. Note that API usage is subject to rate limits and quotas.")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using AI analytics.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

# Create a single tab for Anomaly Detection
# tab1, tab2, tab3 = st.tabs([
#     "Predictive Forecasting",
#     "AI-Generated Insights",
#     "Anomaly Detection"
# ])
tab, = st.tabs(["Anomaly Detection"])

with tab:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Anomaly Detection")
    st.markdown("""
    Automatically identify unusual patterns in your data that might indicate opportunities or issues.
    Anomaly detection helps spot outliers, unusual trends, or suspicious data points.
    """)
    
    # Get numeric columns
    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_columns:
        st.warning("Anomaly detection requires numeric columns in your data.")
    else:
        # Configure anomaly detection
        st.markdown("#### Configure Anomaly Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            anomaly_column = st.selectbox("Column to Analyze", numeric_columns)
            contamination = st.slider(
                "Expected Anomaly Percentage",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                format="%.2f",
                help="The expected percentage of data points that are anomalies."
            )
        
        with col2:
            # Date column for time series context (optional)
            date_columns = []
            for col in st.session_state.cleaned_data.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.cleaned_data[col]):
                        date_columns.append(col)
                    elif pd.api.types.is_object_dtype(st.session_state.cleaned_data[col]):
                        # Try to convert to datetime
                        if pd.to_datetime(st.session_state.cleaned_data[col], errors='coerce').notna().any():
                            date_columns.append(col)
                except:
                    pass
            
            if date_columns:
                date_column = st.selectbox("Date Column (Optional)", ["None"] + date_columns)
            else:
                date_column = "None"
            
            # Color by (optional)
            categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
            if categorical_columns:
                color_column = st.selectbox("Color By (Optional)", ["None"] + categorical_columns)
            else:
                color_column = "None"
        
        # Detection button
        if st.button("Detect Anomalies"):
            # Initialize AI analytics
            ai_analytics = AIAnalytics(st.session_state.cleaned_data)
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            status_text.text("Analyzing data for anomalies...")
            progress_bar.progress(30)
            
            try:
                # Detect anomalies
                anomalies = ai_analytics.detect_anomalies(
                    column=anomaly_column,
                    contamination=contamination
                )
                
                # Update progress
                progress_bar.progress(70)
                status_text.text("Creating visualizations...")
                
                # Count anomalies
                anomaly_count = anomalies.sum()
                normal_count = len(anomalies) - anomaly_count
                
                # Create a copy of the data with anomaly flag
                result_data = st.session_state.cleaned_data.copy()
                result_data['is_anomaly'] = anomalies
                
                # Update progress
                progress_bar.progress(90)
                status_text.text("Finalizing results...")
                
                # Create visualizations
                if date_column != "None":
                    # Time series visualization with anomalies highlighted
                    try:
                        # Ensure date column is datetime
                        if not pd.api.types.is_datetime64_any_dtype(result_data[date_column]):
                            result_data[date_column] = pd.to_datetime(result_data[date_column], errors='coerce')
                        
                        # Sort by date
                        result_data = result_data.sort_values(date_column)
                        
                        # Create time series plot
                        fig = go.Figure()
                        
                        # Add normal points
                        normal_data = result_data[~result_data['is_anomaly']]
                        fig.add_trace(go.Scatter(
                            x=normal_data[date_column],
                            y=normal_data[anomaly_column],
                            mode='lines+markers',
                            name='Normal',
                            marker=dict(
                                size=6,
                                color='blue',
                                opacity=0.7
                            ),
                            line=dict(
                                color='blue',
                                width=2
                            )
                        ))
                        
                        # Add anomalies
                        anomaly_data = result_data[result_data['is_anomaly']]
                        fig.add_trace(go.Scatter(
                            x=anomaly_data[date_column],
                            y=anomaly_data[anomaly_column],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(
                                size=10,
                                color='red',
                                symbol='x',
                                line=dict(
                                    width=2,
                                    color='black'
                                )
                            )
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f'Anomaly Detection for {anomaly_column} Over Time',
                            xaxis_title=date_column,
                            yaxis_title=anomaly_column,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=500
                        )
                    
                    except Exception as e:
                        # Fallback to histogram visualization
                        st.warning(f"Could not create time series plot: {str(e)}. Showing distribution instead.")
                        
                        # Create distribution plot
                        fig = go.Figure()
                        
                        # Create histogram for normal points
                        fig.add_trace(go.Histogram(
                            x=result_data[~result_data['is_anomaly']][anomaly_column],
                            name='Normal',
                            opacity=0.7,
                            marker_color='blue'
                        ))
                        
                        # Create histogram for anomalies
                        fig.add_trace(go.Histogram(
                            x=result_data[result_data['is_anomaly']][anomaly_column],
                            name='Anomaly',
                            opacity=0.7,
                            marker_color='red'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f'Distribution of {anomaly_column} with Anomalies Highlighted',
                            xaxis_title=anomaly_column,
                            yaxis_title='Count',
                            barmode='overlay',
                            height=500
                        )
                
                else:
                    # Create distribution plot
                    fig = go.Figure()
                    
                    # Create histogram for normal points
                    fig.add_trace(go.Histogram(
                        x=result_data[~result_data['is_anomaly']][anomaly_column],
                        name='Normal',
                        opacity=0.7,
                        marker_color='blue'
                    ))
                    
                    # Create histogram for anomalies
                    fig.add_trace(go.Histogram(
                        x=result_data[result_data['is_anomaly']][anomaly_column],
                        name='Anomaly',
                        opacity=0.7,
                        marker_color='red'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Distribution of {anomaly_column} with Anomalies Highlighted',
                        xaxis_title=anomaly_column,
                        yaxis_title='Count',
                        barmode='overlay',
                        height=500
                    )
                
                # Update progress
                progress_bar.progress(100)
                status_text.text("Anomaly detection complete!")
                
                # Clear progress indicators after a delay
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display the visualization
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                st.markdown("#### Anomaly Detection Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Data Points",
                        f"{len(anomalies)}"
                    )
                
                with col2:
                    st.metric(
                        "Anomalies Detected",
                        f"{anomaly_count}"
                    )
                
                with col3:
                    st.metric(
                        "Anomaly Percentage",
                        f"{(anomaly_count / len(anomalies) * 100):.2f}%"
                    )
                
                # Show anomaly details
                st.markdown("#### Anomaly Details")
                
                # Create anomaly statistics
                if anomaly_count > 0:
                    # Calculate statistics
                    normal_mean = result_data[~result_data['is_anomaly']][anomaly_column].mean()
                    normal_std = result_data[~result_data['is_anomaly']][anomaly_column].std()
                    anomaly_mean = result_data[result_data['is_anomaly']][anomaly_column].mean()
                    anomaly_std = result_data[result_data['is_anomaly']][anomaly_column].std()
                    
                    # Calculate z-score (standard deviations from mean)
                    if normal_std > 0:
                        z_score = (anomaly_mean - normal_mean) / normal_std
                    else:
                        z_score = 0
                    
                    # Display statistics comparison
                    comparison_data = pd.DataFrame({
                        'Statistic': ['Mean', 'Standard Deviation', 'Minimum', 'Maximum'],
                        'Normal Data': [
                            normal_mean,
                            normal_std,
                            result_data[~result_data['is_anomaly']][anomaly_column].min(),
                            result_data[~result_data['is_anomaly']][anomaly_column].max()
                        ],
                        'Anomalies': [
                            anomaly_mean,
                            anomaly_std,
                            result_data[result_data['is_anomaly']][anomaly_column].min(),
                            result_data[result_data['is_anomaly']][anomaly_column].max()
                        ],
                        'Difference': [
                            anomaly_mean - normal_mean,
                            anomaly_std - normal_std,
                            result_data[result_data['is_anomaly']][anomaly_column].min() - result_data[~result_data['is_anomaly']][anomaly_column].min(),
                            result_data[result_data['is_anomaly']][anomaly_column].max() - result_data[~result_data['is_anomaly']][anomaly_column].max()
                        ]
                    })
                    
                    # Format numbers
                    comparison_data = comparison_data.round(2)
                    
                    # Display comparison
                    st.dataframe(comparison_data, use_container_width=True)
                    
                    # Provide interpretation
                    st.markdown("#### Interpretation")
                    
                    if abs(z_score) > 3:
                        st.warning(f"The anomalies are significantly different from normal data (z-score: {z_score:.2f}). They deviate by more than 3 standard deviations from the mean.")
                    elif abs(z_score) > 2:
                        st.info(f"The anomalies are moderately different from normal data (z-score: {z_score:.2f}). They deviate by 2-3 standard deviations from the mean.")
                    else:
                        st.info(f"The anomalies are somewhat different from normal data (z-score: {z_score:.2f}). They deviate by less than 2 standard deviations from the mean.")
                    
                    # Show anomaly data
                    with st.expander("View Anomaly Data"):
                        if color_column != "None":
                            # Include color column in the display
                            display_cols = [date_column] if date_column != "None" else []
                            display_cols.extend([anomaly_column, color_column])
                            st.dataframe(result_data[result_data['is_anomaly']][display_cols])
                        else:
                            # Just show date and anomaly columns
                            display_cols = [date_column] if date_column != "None" else []
                            display_cols.append(anomaly_column)
                            st.dataframe(result_data[result_data['is_anomaly']][display_cols])
                
                else:
                    st.info("No anomalies were detected with the current settings. Try increasing the 'Expected Anomaly Percentage'.")
            
            except Exception as e:
                # Update progress
                progress_bar.progress(100)
                status_text.text("Error during anomaly detection!")
                
                # Clear progress indicators after a delay
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.error(f"Error detecting anomalies: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)