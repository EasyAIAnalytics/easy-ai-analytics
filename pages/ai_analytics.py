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

# Initialize session state variables if they don't exist
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using AI analytics.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

# Create tabs for different AI analytics features
tab1, tab2, tab3 = st.tabs([
    "Predictive Forecasting",
    "AI-Generated Insights",
    "Anomaly Detection"
])

with tab1:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Predictive Forecasting")
    st.markdown("""
    Use machine learning models to predict future trends in your numeric data.
    This helps anticipate future performance and make proactive decisions.
    """)
    
    # Check for date columns
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
    
    # Get numeric columns
    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not date_columns:
        st.warning("Predictive forecasting requires at least one date/time column in your data.")
    elif not numeric_columns:
        st.warning("Predictive forecasting requires at least one numeric column in your data.")
    else:
        # Configure forecasting
        st.markdown("#### Configure Time Series Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_column = st.selectbox("Date/Time Column", date_columns)
            value_column = st.selectbox("Value to Forecast", numeric_columns)
        
        with col2:
            forecast_periods = st.slider("Forecast Periods", min_value=7, max_value=365, value=30)
            forecast_method = st.selectbox("Forecasting Method", ["Prophet (recommended)", "Exponential Smoothing"])
        
        # Check if the date column is in datetime format
        date_data = st.session_state.cleaned_data[date_column]
        if not pd.api.types.is_datetime64_any_dtype(date_data):
            # Try to convert to datetime
            try:
                st.session_state.cleaned_data[date_column] = pd.to_datetime(date_data)
                st.info(f"Converting '{date_column}' to datetime format.")
            except Exception as e:
                st.error(f"Could not convert '{date_column}' to datetime format. Error: {str(e)}")
                st.stop()
        
        # Generate forecast
        if st.button("Generate Forecast"):
            # Initialize AI analytics
            ai_analytics = AIAnalytics(st.session_state.cleaned_data)
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            status_text.text("Preparing data for forecasting...")
            progress_bar.progress(10)
            
            try:
                # Select forecast method
                method = "prophet" if forecast_method == "Prophet (recommended)" else "exponential_smoothing"
                
                # Update progress
                status_text.text("Training forecasting model...")
                progress_bar.progress(30)
                
                # Generate forecast
                forecast_results = ai_analytics.predict_time_series(
                    date_column=date_column,
                    value_column=value_column,
                    periods=forecast_periods,
                    method=method
                )
                
                # Update progress
                status_text.text("Creating visualizations...")
                progress_bar.progress(70)
                
                if forecast_results:
                    # Extract forecast data
                    forecast_dates = forecast_results['dates']
                    forecast_values = forecast_results['predicted_values']
                    historical_dates = forecast_results['historical_dates']
                    historical_values = forecast_results['historical_values']
                    
                    # Create confidence intervals if available
                    has_confidence_intervals = 'upper_bound' in forecast_results and 'lower_bound' in forecast_results
                    
                    # Create forecast plot
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_dates,
                        y=historical_values,
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='blue')
                    ))
                    
                    # Add forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add confidence intervals if available
                    if has_confidence_intervals:
                        upper_bound = forecast_results['upper_bound']
                        lower_bound = forecast_results['lower_bound']
                        
                        # Add confidence interval
                        fig.add_trace(go.Scatter(
                            x=forecast_dates + forecast_dates[::-1],
                            y=upper_bound + lower_bound[::-1],
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.1)',
                            line=dict(color='rgba(255, 0, 0, 0)'),
                            name='95% Confidence Interval'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Forecast of {value_column} ({forecast_periods} periods)',
                        xaxis_title='Date',
                        yaxis_title=value_column,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500
                    )
                    
                    # Add vertical line separating historical data from forecast
                    fig.add_vline(
                        x=historical_dates[-1],
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Forecast Start",
                        annotation_position="top right"
                    )
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Forecast completed!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display the forecast plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Provide forecast insights
                    st.markdown("#### Forecast Insights")
                    
                    # Calculate forecast statistics
                    # Ensure values are numeric before calculations
                    last_historical_value = float(historical_values[-1])
                    last_forecast_value = float(forecast_values[-1])
                    forecast_change = last_forecast_value - last_historical_value
                    forecast_pct_change = (forecast_change / last_historical_value) * 100 if last_historical_value != 0 else 0
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current Value",
                            f"{last_historical_value:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            f"Forecasted Value ({forecast_periods} periods ahead)",
                            f"{last_forecast_value:.2f}",
                            f"{forecast_change:+.2f} ({forecast_pct_change:+.1f}%)",
                            delta_color="normal"
                        )
                    
                    with col3:
                        # Calculate average growth rate
                        if len(forecast_values) > 1:
                            # Ensure values are numeric before calculations
                            first_value = float(forecast_values[0])
                            last_value = float(forecast_values[-1])
                            if first_value > 0:  # Avoid division by zero or negative values
                                growth_rate = (last_value / first_value) ** (1 / len(forecast_values)) - 1
                                growth_pct = growth_rate * 100
                                st.metric(
                                    "Average Growth Rate",
                                    f"{growth_pct:+.2f}% per period"
                                )
                            else:
                                st.metric(
                                    "Average Growth Rate",
                                    "N/A (invalid data)"
                                )
                    
                    # Provide additional insights
                    st.markdown("#### Additional Analysis")
                    
                    # Identify trends
                    if forecast_pct_change > 10:
                        st.success(f"📈 Strong upward trend: {value_column} is projected to increase by {forecast_pct_change:.1f}% over the next {forecast_periods} periods.")
                    elif forecast_pct_change > 5:
                        st.success(f"📈 Moderate upward trend: {value_column} is projected to increase by {forecast_pct_change:.1f}% over the next {forecast_periods} periods.")
                    elif forecast_pct_change > 0:
                        st.info(f"📈 Slight upward trend: {value_column} is projected to increase by {forecast_pct_change:.1f}% over the next {forecast_periods} periods.")
                    elif forecast_pct_change > -5:
                        st.info(f"📉 Slight downward trend: {value_column} is projected to decrease by {abs(forecast_pct_change):.1f}% over the next {forecast_periods} periods.")
                    elif forecast_pct_change > -10:
                        st.warning(f"📉 Moderate downward trend: {value_column} is projected to decrease by {abs(forecast_pct_change):.1f}% over the next {forecast_periods} periods.")
                    else:
                        st.error(f"📉 Strong downward trend: {value_column} is projected to decrease by {abs(forecast_pct_change):.1f}% over the next {forecast_periods} periods.")
                    
                    # Check for seasonality (only for Prophet)
                    if method == "prophet" and 'seasonality_components' in forecast_results:
                        st.markdown("#### Seasonality Analysis")
                        
                        if 'yearly' in forecast_results['seasonality_components']:
                            st.markdown("##### Yearly Seasonality")
                            
                            yearly_fig = go.Figure()
                            
                            yearly_fig.add_trace(go.Scatter(
                                x=forecast_results['seasonality_components']['yearly']['x'],
                                y=forecast_results['seasonality_components']['yearly']['y'],
                                mode='lines',
                                line=dict(color='green')
                            ))
                            
                            yearly_fig.update_layout(
                                title='Yearly Seasonality Pattern',
                                xaxis_title='Month',
                                yaxis_title='Effect',
                                height=300
                            )
                            
                            st.plotly_chart(yearly_fig, use_container_width=True)
                        
                        if 'weekly' in forecast_results['seasonality_components']:
                            st.markdown("##### Weekly Seasonality")
                            
                            weekly_fig = go.Figure()
                            
                            weekly_fig.add_trace(go.Scatter(
                                x=forecast_results['seasonality_components']['weekly']['x'],
                                y=forecast_results['seasonality_components']['weekly']['y'],
                                mode='lines',
                                line=dict(color='purple')
                            ))
                            
                            weekly_fig.update_layout(
                                title='Weekly Seasonality Pattern',
                                xaxis_title='Day of Week',
                                yaxis_title='Effect',
                                height=300
                            )
                            
                            st.plotly_chart(weekly_fig, use_container_width=True)
                
                else:
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Forecast failed!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.error("Could not generate forecast. Please check your data and try again.")
            
            except Exception as e:
                # Update progress
                progress_bar.progress(100)
                status_text.text("Error during forecasting!")
                
                # Clear progress indicators after a delay
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.error(f"Error generating forecast: {str(e)}")
                st.info("Forecasting works best with evenly spaced time series data. Try a different date column, value column, or forecasting method.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### AI-Generated Insights")
    st.markdown("""
    Leverage artificial intelligence to automatically generate natural language explanations of complex data patterns.
    AI can identify patterns and insights that might be missed in manual analysis.
    """)
    
    # Check if API keys are available
    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        st.warning("AI-generated insights require either an OpenAI or Anthropic API key. Please provide one in the settings.")
    else:
        # AI insight generation options
        st.markdown("#### Configure AI Insight Generation")
        
        insight_type = st.selectbox(
            "Insight Type",
            ["General Data Overview", "Relationship Analysis", "Business Recommendations", "Custom Query"]
        )
        
        # Optional column focus
        st.markdown("#### Column Focus (Optional)")
        st.info("Select specific columns to focus the AI analysis on, or leave empty to analyze all columns.")
        
        # Get column lists
        all_columns = st.session_state.cleaned_data.columns.tolist()
        num_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
        
        # Multi-select for columns to focus on
        selected_columns = st.multiselect(
            "Focus Columns",
            all_columns,
            default=[]
        )
        
        # Custom query input
        if insight_type == "Custom Query":
            custom_query = st.text_area(
                "Your Question for the AI",
                "What are the key patterns and insights in this dataset?",
                height=100
            )
        
        # Depth of analysis
        analysis_depth = st.slider(
            "Analysis Depth",
            min_value=1,
            max_value=5,
            value=3,
            help="Controls the depth and detail of AI analysis. Higher values provide more detailed insights but take longer."
        )
        
        # Number of insights
        num_insights = st.slider(
            "Number of Insights",
            min_value=3,
            max_value=10,
            value=5
        )
        
        # Generate insights button
        if st.button("Generate AI Insights"):
            # Initialize AI analytics
            ai_analytics = AIAnalytics(st.session_state.cleaned_data)
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            status_text.text("Analyzing your data...")
            progress_bar.progress(30)
            
            try:
                # Filter data if columns selected
                if selected_columns:
                    analysis_data = st.session_state.cleaned_data[selected_columns].copy()
                    ai_analytics = AIAnalytics(analysis_data)
                
                # Generate insights with AI
                insights = ai_analytics.generate_insights_with_ai(max_insights=num_insights)
                
                # Update progress
                progress_bar.progress(90)
                status_text.text("Processing results...")
                
                if insights:
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display insights
                    st.markdown("#### AI-Generated Insights")
                    
                    for i, insight in enumerate(insights, 1):
                        # Create expanded card with nice formatting for each insight
                        with st.expander(f"Insight {i}", expanded=i==1):
                            st.markdown(f"<div style='padding: 10px; border-radius: 5px; background-color: #f7f7f7;'>{insight}</div>", unsafe_allow_html=True)
                    
                    # Provide additional context
                    st.markdown("#### How to Use These Insights")
                    st.info("""
                    These AI-generated insights are meant to guide your analysis and highlight patterns that may deserve further investigation. Consider the following:
                    
                    - Verify insights with data visualizations
                    - Use these insights to guide business decisions
                    - Remember that AI can identify patterns but may not understand business context
                    - Always validate important findings with domain expertise
                    """)
                    
                    # Save insights option
                    if st.button("Save These Insights to Database"):
                        # Check if we have a current dataset ID
                        if 'current_dataset_id' in st.session_state and st.session_state.current_dataset_id:
                            import database as db
                            saved_count = db.save_insights(st.session_state.current_dataset_id, insights)
                            
                            if saved_count > 0:
                                st.success(f"Successfully saved {saved_count} insights to the database!")
                            else:
                                st.error("Failed to save insights.")
                        else:
                            st.warning("Please save your dataset first before saving insights.")
                else:
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Analysis failed!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.error("Could not generate insights. The AI service may be unavailable or the data may not contain enough patterns for analysis.")
            
            except Exception as e:
                # Update progress
                progress_bar.progress(100)
                status_text.text("Error during analysis!")
                
                # Clear progress indicators after a delay
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.error(f"Error generating insights: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
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