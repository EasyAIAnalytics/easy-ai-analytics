import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import time
import database as db
from sklearn.cluster import KMeans
from utils.ai_analytics import AIAnalytics

# Page configuration
st.set_page_config(
    page_title="Business Features",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    if st.button("Logout", key="logout_btn_business_features"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# Page title with styling
st.markdown('<h1 class="main-header">Business Intelligence Features</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced tools for business analysis, KPI tracking, and financial modeling</p>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using business features.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

if 'kpi_targets' not in st.session_state:
    st.session_state.kpi_targets = {}
    
if 'benchmarks' not in st.session_state:
    st.session_state.benchmarks = {}

# Create tabs for different business features
tab1, tab2, tab3, tab4 = st.tabs([
    "KPI Monitoring", 
    "Competitor Benchmarking",
    "Financial Analysis",
    "Customer Segmentation"
])

with tab1:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### KPI Monitoring Dashboard")
    st.markdown("""
    Create and track Key Performance Indicators (KPIs) to monitor your business performance.
    Set targets and receive alerts when metrics fall outside acceptable ranges.
    """)
    
    # KPI configuration section
    st.markdown("#### Configure KPIs")
    
    # Get numeric columns for KPIs
    if st.session_state.cleaned_data is None:
        st.warning("Please upload data in the main dashboard before using this feature.")
        st.stop()
    
    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    date_columns = []
    
    # Try to identify date columns
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
    
    # KPI creation form
    with st.form("kpi_form"):
        st.markdown("##### Create New KPI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            kpi_name = st.text_input("KPI Name", "")
            kpi_metric = st.selectbox("KPI Metric", numeric_columns if numeric_columns else ["No numeric columns available"])
            kpi_aggregation = st.selectbox("Aggregation", ["Sum", "Average", "Min", "Max", "Count", "Median"])
        
        with col2:
            kpi_target = st.number_input("Target Value", value=0.0)
            kpi_warning_threshold = st.slider("Warning Threshold (%)", min_value=1, max_value=50, value=10)
            kpi_date_column = st.selectbox("Time Dimension (optional)", ["None"] + date_columns)
        
        submit_button = st.form_submit_button("Create KPI")
        
        if submit_button and kpi_name and kpi_metric and kpi_metric in numeric_columns:
            # Create KPI configuration
            kpi_config = {
                "name": kpi_name,
                "metric": kpi_metric,
                "aggregation": kpi_aggregation,
                "target": kpi_target,
                "warning_threshold": kpi_warning_threshold / 100,  # Convert to decimal
                "date_column": kpi_date_column if kpi_date_column != "None" else None,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to session state
            st.session_state.kpi_targets[kpi_name] = kpi_config
            st.success(f"KPI '{kpi_name}' created successfully!")
    
    # Display KPIs
    if st.session_state.kpi_targets:
        st.markdown("#### KPI Dashboard")
        
        # Create KPI cards in a grid layout
        cols = st.columns(3)
        
        for i, (kpi_name, kpi_config) in enumerate(st.session_state.kpi_targets.items()):
            with cols[i % 3]:
                # Calculate current KPI value
                data = st.session_state.cleaned_data.copy()
                metric_col = kpi_config["metric"]
                
                if kpi_config["aggregation"] == "Sum":
                    current_value = data[metric_col].sum()
                elif kpi_config["aggregation"] == "Average":
                    current_value = data[metric_col].mean()
                elif kpi_config["aggregation"] == "Min":
                    current_value = data[metric_col].min()
                elif kpi_config["aggregation"] == "Max":
                    current_value = data[metric_col].max()
                elif kpi_config["aggregation"] == "Count":
                    current_value = data[metric_col].count()
                else:  # Median
                    current_value = data[metric_col].median()
                
                # Calculate percentage of target
                target = kpi_config["target"]
                if target != 0:
                    percentage = (current_value / target) * 100
                else:
                    percentage = 0
                
                # Determine status
                warning_threshold = kpi_config["warning_threshold"]
                if abs(1 - (current_value / target)) <= warning_threshold:
                    status = "On Target"
                    color = "green"
                elif current_value < target:
                    status = "Below Target"
                    color = "red"
                else:
                    status = "Above Target"
                    color = "blue" if kpi_config["aggregation"] in ["Sum", "Average", "Max", "Count"] else "red"
                
                # Create KPI card
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); background: white; margin-bottom: 20px;">
                    <h4 style="margin-top: 0;">{kpi_name}</h4>
                    <p style="color: gray; font-size: 14px;">{kpi_config["aggregation"]} of {metric_col}</p>
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 24px; font-weight: bold;">{current_value:.2f}</span>
                        <span style="color: {color}; font-weight: bold;">{status}</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Target: {target:.2f}</span>
                        <div style="height: 5px; width: 100%; background-color: #eee; border-radius: 5px; margin-top: 5px;">
                            <div style="height: 100%; width: {min(percentage, 100)}%; background-color: {color}; border-radius: 5px;"></div>
                        </div>
                    </div>
                    <p style="margin-bottom: 0; font-size: 12px; color: gray;">{min(percentage, 100):.1f}% of target</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Time series chart if date column is available
                if kpi_config["date_column"] and kpi_config["date_column"] in data.columns:
                    date_col = kpi_config["date_column"]
                    
                    # Ensure date column is datetime
                    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                    
                    # Group by date and calculate aggregation
                    data = data.dropna(subset=[date_col, metric_col])
                    data = data.sort_values(date_col)
                    
                    # Group by date (year-month)
                    data['period'] = data[date_col].dt.to_period('M')
                    
                    if kpi_config["aggregation"] == "Sum":
                        time_series = data.groupby('period')[metric_col].sum()
                    elif kpi_config["aggregation"] == "Average":
                        time_series = data.groupby('period')[metric_col].mean()
                    elif kpi_config["aggregation"] == "Min":
                        time_series = data.groupby('period')[metric_col].min()
                    elif kpi_config["aggregation"] == "Max":
                        time_series = data.groupby('period')[metric_col].max()
                    elif kpi_config["aggregation"] == "Count":
                        time_series = data.groupby('period')[metric_col].count()
                    else:  # Median
                        time_series = data.groupby('period')[metric_col].median()
                    
                    # Convert period index to datetime for plotting
                    time_series.index = time_series.index.to_timestamp()
                    
                    # Create time series chart
                    fig = go.Figure()
                    
                    # Add KPI value line
                    fig.add_trace(go.Scatter(
                        x=time_series.index,
                        y=time_series.values,
                        mode='lines+markers',
                        name=kpi_name,
                        line=dict(color=color, width=3)
                    ))
                    
                    # Add target line
                    fig.add_trace(go.Scatter(
                        x=[time_series.index.min(), time_series.index.max()],
                        y=[target, target],
                        mode='lines',
                        name='Target',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{kpi_name} Trend",
                        xaxis_title="Date",
                        yaxis_title=metric_col,
                        height=300,
                        margin=dict(l=30, r=30, t=40, b=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No KPIs configured yet. Use the form above to create your first KPI.")
    
    # KPI management
    with st.expander("Manage KPIs"):
        if st.session_state.kpi_targets:
            kpi_to_delete = st.selectbox(
                "Select KPI to delete",
                list(st.session_state.kpi_targets.keys())
            )
            
            if st.button("Delete Selected KPI"):
                if kpi_to_delete in st.session_state.kpi_targets:
                    del st.session_state.kpi_targets[kpi_to_delete]
                    st.success(f"KPI '{kpi_to_delete}' deleted successfully!")
                    st.rerun()
        else:
            st.info("No KPIs to manage.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Competitor Benchmarking")
    st.markdown("""
    Compare your performance metrics against industry benchmarks and competitors.
    Upload competitor data or use industry benchmark values to see how you stack up.
    """)
    
    # Benchmark configuration
    st.markdown("#### Benchmark Configuration")
    
    benchmark_method = st.radio(
        "Benchmark Method",
        ["Enter Industry Standards", "Upload Competitor Data"],
        horizontal=True
    )
    
    if benchmark_method == "Enter Industry Standards":
        # Manual benchmark entry
        with st.form("benchmark_form"):
            st.markdown("##### Create New Benchmark")
            
            col1, col2 = st.columns(2)
            
            with col1:
                benchmark_name = st.text_input("Benchmark Name", "")
                benchmark_metric = st.selectbox("Metric", numeric_columns if numeric_columns else ["No numeric columns available"])
            
            with col2:
                benchmark_value = st.number_input("Industry Average/Standard", value=0.0)
                benchmark_description = st.text_input("Description", "")
            
            submit_button = st.form_submit_button("Add Benchmark")
            
            if submit_button and benchmark_name and benchmark_metric and benchmark_metric in numeric_columns:
                # Create benchmark configuration
                benchmark_config = {
                    "name": benchmark_name,
                    "metric": benchmark_metric,
                    "value": benchmark_value,
                    "description": benchmark_description,
                    "type": "industry_standard",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add to session state
                st.session_state.benchmarks[benchmark_name] = benchmark_config
                st.success(f"Benchmark '{benchmark_name}' added successfully!")
    
    else:  # Upload Competitor Data
        st.markdown("##### Upload Competitor Data CSV")
        st.info("Upload a CSV file with competitor data. The file should have the same structure as your main dataset.")
        
        uploaded_file = st.file_uploader("Choose a competitor CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read competitor CSV
                competitor_data = pd.read_csv(uploaded_file)
                
                # Add form for creating benchmark from competitor data
                with st.form("competitor_benchmark_form"):
                    st.markdown("##### Create Competitor Benchmark")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        benchmark_name = st.text_input("Competitor Benchmark Name", "")
                        benchmark_metric = st.selectbox(
                            "Metric", 
                            [col for col in numeric_columns if col in competitor_data.columns]
                        )
                        benchmark_aggregation = st.selectbox(
                            "Aggregation", 
                            ["Average", "Sum", "Min", "Max", "Median"]
                        )
                    
                    with col2:
                        benchmark_description = st.text_input("Competitor Description", "")
                        st.markdown(f"**Preview:** {len(competitor_data)} rows, {len(competitor_data.columns)} columns")
                        st.markdown(f"**Available Metrics:** {', '.join([col for col in numeric_columns if col in competitor_data.columns])}")
                    
                    submit_button = st.form_submit_button("Add Competitor Benchmark")
                    
                    if submit_button and benchmark_name and benchmark_metric and benchmark_metric in competitor_data.columns:
                        # Calculate benchmark value
                        if benchmark_aggregation == "Average":
                            benchmark_value = competitor_data[benchmark_metric].mean()
                        elif benchmark_aggregation == "Sum":
                            benchmark_value = competitor_data[benchmark_metric].sum()
                        elif benchmark_aggregation == "Min":
                            benchmark_value = competitor_data[benchmark_metric].min()
                        elif benchmark_aggregation == "Max":
                            benchmark_value = competitor_data[benchmark_metric].max()
                        else:  # Median
                            benchmark_value = competitor_data[benchmark_metric].median()
                        
                        # Create benchmark configuration
                        benchmark_config = {
                            "name": benchmark_name,
                            "metric": benchmark_metric,
                            "value": benchmark_value,
                            "description": benchmark_description,
                            "aggregation": benchmark_aggregation,
                            "type": "competitor",
                            "data_rows": len(competitor_data),
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Add to session state
                        st.session_state.benchmarks[benchmark_name] = benchmark_config
                        st.success(f"Competitor benchmark '{benchmark_name}' added successfully!")
                
                # Preview competitor data
                with st.expander("Preview Competitor Data"):
                    st.dataframe(competitor_data.head(5))
            
            except Exception as e:
                st.error(f"Error processing competitor data: {e}")
    
    # Display benchmarks
    if st.session_state.benchmarks:
        st.markdown("#### Benchmark Comparison")
        
        # Select metrics to display
        available_metrics = set()
        for _, benchmark in st.session_state.benchmarks.items():
            available_metrics.add(benchmark["metric"])
        
        selected_metric = st.selectbox(
            "Select Metric to Compare",
            list(available_metrics)
        )
        
        if selected_metric:
            # Calculate your value
            your_data = st.session_state.cleaned_data.copy()
            your_value = your_data[selected_metric].mean()
            
            # Get relevant benchmarks
            relevant_benchmarks = {
                name: config for name, config in st.session_state.benchmarks.items()
                if config["metric"] == selected_metric
            }
            
            if relevant_benchmarks:
                # Create comparison chart
                fig = go.Figure()
                
                # Add your value
                fig.add_trace(go.Bar(
                    x=["Your Data"],
                    y=[your_value],
                    name="Your Data",
                    marker_color='royalblue'
                ))
                
                # Add benchmark values
                for name, config in relevant_benchmarks.items():
                    fig.add_trace(go.Bar(
                        x=[name],
                        y=[config["value"]],
                        name=name,
                        marker_color='lightgray' if config["type"] == "industry_standard" else 'crimson'
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Comparison of {selected_metric}",
                    xaxis_title="Data Source",
                    yaxis_title=selected_metric,
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display percentage differences
                st.markdown("#### Performance Gap Analysis")
                
                comparison_data = []
                for name, config in relevant_benchmarks.items():
                    benchmark_value = config["value"]
                    difference = your_value - benchmark_value
                    percentage = (difference / benchmark_value * 100) if benchmark_value != 0 else 0
                    
                    comparison_data.append({
                        "Benchmark": name,
                        "Your Value": round(your_value, 2),
                        "Benchmark Value": round(benchmark_value, 2),
                        "Difference": round(difference, 2),
                        "% Difference": round(percentage, 2),
                        "Status": "Higher" if difference > 0 else "Lower" if difference < 0 else "Equal"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Instead of using applymap with styling which can cause errors, let's use a simpler approach
                # Add emoji indicators to make status more visible
                comparison_df['Visual Status'] = comparison_df['Status'].apply(
                    lambda x: "✅ Higher" if x == "Higher" else "❌ Lower" if x == "Lower" else "➖ Same"
                )
                
                # Display the dataframe without styling
                st.dataframe(comparison_df, use_container_width=True)
                
                # Recommendations based on benchmark comparison
                st.markdown("#### Insights & Recommendations")
                
                recommendations = []
                for item in comparison_data:
                    benchmark = item["Benchmark"]
                    metric = selected_metric
                    status = item["Status"]
                    percentage = item["% Difference"]
                    
                    if status == "Higher" and percentage > 10:
                        recommendations.append(f"✅ Your {metric} is {percentage:.1f}% higher than {benchmark}. This suggests strong performance in this area.")
                    elif status == "Higher" and percentage <= 10:
                        recommendations.append(f"✓ Your {metric} is slightly higher ({percentage:.1f}%) than {benchmark}. You're performing well but the advantage is marginal.")
                    elif status == "Lower" and abs(percentage) > 20:
                        recommendations.append(f"❗ Your {metric} is {abs(percentage):.1f}% lower than {benchmark}. This represents a significant gap that requires attention.")
                    elif status == "Lower":
                        recommendations.append(f"⚠️ Your {metric} is {abs(percentage):.1f}% lower than {benchmark}. Consider strategies to improve this metric.")
                    else:
                        recommendations.append(f"Your {metric} is equal to {benchmark}.")
                
                for recommendation in recommendations:
                    st.markdown(recommendation)
            else:
                st.info(f"No benchmarks available for the selected metric: {selected_metric}")
    else:
        st.info("No benchmarks configured yet. Use the form above to create your first benchmark.")
    
    # Benchmark management
    with st.expander("Manage Benchmarks"):
        if st.session_state.benchmarks:
            benchmark_to_delete = st.selectbox(
                "Select benchmark to delete",
                list(st.session_state.benchmarks.keys())
            )
            
            if st.button("Delete Selected Benchmark"):
                if benchmark_to_delete in st.session_state.benchmarks:
                    del st.session_state.benchmarks[benchmark_to_delete]
                    st.success(f"Benchmark '{benchmark_to_delete}' deleted successfully!")
                    st.rerun()
        else:
            st.info("No benchmarks to manage.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Financial Analysis Tools")
    st.markdown("""
    Perform financial calculations including Net Present Value (NPV), Return on Investment (ROI),
    and other key financial metrics for business decision-making.
    """)
    
    # Create subtabs for different financial analyses
    fin_tab1, fin_tab2, fin_tab3 = st.tabs([
        "Investment Analysis",
        "Financial Ratios",
        "Break-even Analysis"
    ])
    
    with fin_tab1:
        st.markdown("#### Investment Analysis Calculator")
        st.markdown("Calculate Net Present Value (NPV), Internal Rate of Return (IRR), and Return on Investment (ROI).")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_investment = st.number_input("Initial Investment", min_value=0.0, value=10000.0, step=1000.0)
            
            # Create dynamic form for cash flows
            num_periods = st.number_input("Number of Cash Flow Periods", min_value=1, max_value=10, value=5)
            
            cash_flows = []
            for i in range(num_periods):
                cash_flow = st.number_input(f"Cash Flow - Year {i+1}", value=2000.0 + (i * 500), step=100.0)
                cash_flows.append(cash_flow)
        
        with col2:
            discount_rate = st.slider("Discount Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5)
            
            # Calculate financial metrics
            if st.button("Calculate Financial Metrics"):
                # Calculate Net Present Value (NPV)
                npv = -initial_investment
                for i, cf in enumerate(cash_flows):
                    npv += cf / ((1 + discount_rate/100) ** (i+1))
                
                # Calculate IRR
                # IRR is the discount rate that makes NPV = 0
                # We'll use a simple approximation method
                try:
                    irr = np.irr([-initial_investment] + cash_flows) * 100
                except:
                    irr = None
                
                # Calculate ROI
                total_cf = sum(cash_flows)
                roi = ((total_cf - initial_investment) / initial_investment) * 100
                
                # Calculate Payback Period
                cumulative_cf = 0
                payback_period = num_periods
                for i, cf in enumerate(cash_flows):
                    cumulative_cf += cf
                    if cumulative_cf >= initial_investment:
                        payback_period = i + 1 - (cumulative_cf - initial_investment) / cf
                        break
                
                # Display results
                st.markdown("#### Results")
                
                # Create metrics display
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("Net Present Value (NPV)", f"${npv:.2f}")
                    st.metric("Return on Investment (ROI)", f"{roi:.2f}%")
                
                with metric_col2:
                    st.metric("Internal Rate of Return (IRR)", f"{irr:.2f}%" if irr is not None else "N/A")
                    st.metric("Payback Period", f"{payback_period:.2f} years")
                
                # Interpretation
                st.markdown("#### Interpretation")
                
                if npv > 0:
                    st.success("This investment has a positive NPV, suggesting it would add value.")
                elif npv < 0:
                    st.error("This investment has a negative NPV, suggesting it would decrease value.")
                else:
                    st.info("This investment has a zero NPV, suggesting it would break even in present value terms.")
                
                # Visualize cash flows
                st.markdown("#### Cash Flow Visualization")
                
                # Prepare data for visualization
                periods = list(range(num_periods + 1))
                all_flows = [-initial_investment] + cash_flows
                cumulative = [all_flows[0]]
                
                for cf in cash_flows:
                    cumulative.append(cumulative[-1] + cf)
                
                # Create figure
                fig = go.Figure()
                
                # Add bars for individual cash flows
                fig.add_trace(go.Bar(
                    x=periods,
                    y=all_flows,
                    name="Cash Flows",
                    marker_color=['red' if flow < 0 else 'green' for flow in all_flows]
                ))
                
                # Add line for cumulative cash flow
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=cumulative,
                    mode='lines+markers',
                    name='Cumulative Cash Flow',
                    line=dict(color='blue', width=2)
                ))
                
                # Add break-even point
                if any(cum >= 0 for cum in cumulative) and cumulative[0] < 0:
                    # Find where cumulative first becomes positive
                    for i, cum in enumerate(cumulative):
                        if cum >= 0:
                            break_even_period = i
                            break
                    
                    fig.add_trace(go.Scatter(
                        x=[break_even_period],
                        y=[cumulative[break_even_period]],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='star'),
                        name='Break-even Point',
                        hoverinfo='text',
                        text=f'Break-even at period {break_even_period}'
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Cash Flow Over Time",
                    xaxis_title="Period",
                    yaxis_title="Cash Flow ($)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sensitivity analysis
                st.markdown("#### Sensitivity Analysis")
                st.markdown("See how changes in the discount rate affect NPV.")
                
                # Calculate NPV for different discount rates
                discount_rates = np.arange(0, 30.1, 2)
                npvs = []
                
                for rate in discount_rates:
                    npv_i = -initial_investment
                    for i, cf in enumerate(cash_flows):
                        npv_i += cf / ((1 + rate/100) ** (i+1))
                    npvs.append(npv_i)
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=discount_rates,
                    y=npvs,
                    mode='lines+markers',
                    name='NPV',
                    line=dict(color='green', width=2)
                ))
                
                # Add line at y=0
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Break-even NPV",
                    annotation_position="bottom right"
                )
                
                # Add line at current discount rate
                fig.add_vline(
                    x=discount_rate,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text="Current Discount Rate",
                    annotation_position="top left"
                )
                
                # Update layout
                fig.update_layout(
                    title="NPV Sensitivity to Discount Rate",
                    xaxis_title="Discount Rate (%)",
                    yaxis_title="Net Present Value ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with fin_tab2:
        st.markdown("#### Financial Ratio Calculator")
        st.markdown("Calculate key financial ratios to assess business performance.")
        
        # Input form for financial data
        with st.form("financial_ratios_form"):
            st.markdown("##### Enter Financial Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                revenue = st.number_input("Revenue ($)", min_value=0.0, value=1000000.0, step=10000.0)
                cost_of_goods_sold = st.number_input("Cost of Goods Sold ($)", min_value=0.0, value=600000.0, step=10000.0)
                operating_expenses = st.number_input("Operating Expenses ($)", min_value=0.0, value=200000.0, step=10000.0)
                net_income = st.number_input("Net Income ($)", min_value=-1000000.0, value=150000.0, step=10000.0)
            
            with col2:
                total_assets = st.number_input("Total Assets ($)", min_value=0.0, value=800000.0, step=10000.0)
                current_assets = st.number_input("Current Assets ($)", min_value=0.0, value=300000.0, step=10000.0)
                total_liabilities = st.number_input("Total Liabilities ($)", min_value=0.0, value=400000.0, step=10000.0)
                current_liabilities = st.number_input("Current Liabilities ($)", min_value=0.0, value=150000.0, step=10000.0)
            
            calculate_button = st.form_submit_button("Calculate Ratios")
            
            if calculate_button:
                # Calculate financial ratios
                gross_profit = revenue - cost_of_goods_sold
                gross_margin = (gross_profit / revenue) * 100 if revenue != 0 else 0
                
                operating_profit = gross_profit - operating_expenses
                operating_margin = (operating_profit / revenue) * 100 if revenue != 0 else 0
                
                net_profit_margin = (net_income / revenue) * 100 if revenue != 0 else 0
                
                return_on_assets = (net_income / total_assets) * 100 if total_assets != 0 else 0
                
                equity = total_assets - total_liabilities
                return_on_equity = (net_income / equity) * 100 if equity != 0 else 0
                
                current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
                
                quick_ratio = (current_assets - 0) / current_liabilities if current_liabilities != 0 else 0  # Simplified (no inventory)
                
                debt_to_equity = total_liabilities / equity if equity != 0 else 0
                
                asset_turnover = revenue / total_assets if total_assets != 0 else 0
                
                # Display results
                st.markdown("#### Financial Ratios")
                
                # Create columns for different ratio categories
                ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
                
                with ratio_col1:
                    st.markdown("##### Profitability Ratios")
                    st.metric("Gross Margin", f"{gross_margin:.2f}%")
                    st.metric("Operating Margin", f"{operating_margin:.2f}%")
                    st.metric("Net Profit Margin", f"{net_profit_margin:.2f}%")
                
                with ratio_col2:
                    st.markdown("##### Efficiency Ratios")
                    st.metric("Return on Assets (ROA)", f"{return_on_assets:.2f}%")
                    st.metric("Return on Equity (ROE)", f"{return_on_equity:.2f}%")
                    st.metric("Asset Turnover", f"{asset_turnover:.2f}x")
                
                with ratio_col3:
                    st.markdown("##### Liquidity & Solvency Ratios")
                    st.metric("Current Ratio", f"{current_ratio:.2f}x")
                    st.metric("Quick Ratio", f"{quick_ratio:.2f}x")
                    st.metric("Debt to Equity", f"{debt_to_equity:.2f}x")
                
                # Create visualizations
                st.markdown("#### Ratio Visualizations")
                
                # Create profitability chart
                profitability_data = pd.DataFrame({
                    'Ratio': ['Gross Margin', 'Operating Margin', 'Net Profit Margin'],
                    'Value': [gross_margin, operating_margin, net_profit_margin]
                })
                
                fig1 = px.bar(
                    profitability_data,
                    x='Ratio',
                    y='Value',
                    title='Profitability Ratios (%)',
                    color='Ratio',
                    text_auto='.1f'
                )
                
                fig1.update_layout(yaxis_title='Percentage (%)')
                
                # Create efficiency chart
                efficiency_data = pd.DataFrame({
                    'Ratio': ['Return on Assets', 'Return on Equity'],
                    'Value': [return_on_assets, return_on_equity]
                })
                
                fig2 = px.bar(
                    efficiency_data,
                    x='Ratio',
                    y='Value',
                    title='Efficiency Ratios (%)',
                    color='Ratio',
                    text_auto='.1f'
                )
                
                fig2.update_layout(yaxis_title='Percentage (%)')
                
                # Display charts in columns
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                with chart_col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Create liquidity and solvency chart
                liquidity_data = pd.DataFrame({
                    'Ratio': ['Current Ratio', 'Quick Ratio', 'Debt to Equity'],
                    'Value': [current_ratio, quick_ratio, debt_to_equity]
                })
                
                fig3 = px.bar(
                    liquidity_data,
                    x='Ratio',
                    y='Value',
                    title='Liquidity & Solvency Ratios',
                    color='Ratio',
                    text_auto='.2f'
                )
                
                fig3.update_layout(yaxis_title='Ratio Value (x)')
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Financial health assessment
                st.markdown("#### Financial Health Assessment")
                
                # Set thresholds for ratio interpretation
                assessment = []
                
                # Profitability assessment
                if gross_margin > 40:
                    assessment.append("✅ **Strong gross margin**: Your gross margin of {:.1f}% is excellent, indicating effective pricing and cost management.".format(gross_margin))
                elif gross_margin > 20:
                    assessment.append("✓ **Good gross margin**: Your gross margin of {:.1f}% is solid, though there may be room for improvement.".format(gross_margin))
                else:
                    assessment.append("⚠️ **Low gross margin**: Your gross margin of {:.1f}% is below average, suggesting potential pricing issues or high production costs.".format(gross_margin))
                
                if net_profit_margin > 20:
                    assessment.append("✅ **Excellent profitability**: Your net profit margin of {:.1f}% indicates very strong overall profitability.".format(net_profit_margin))
                elif net_profit_margin > 10:
                    assessment.append("✓ **Good profitability**: Your net profit margin of {:.1f}% shows healthy overall profitability.".format(net_profit_margin))
                elif net_profit_margin > 5:
                    assessment.append("⚠️ **Fair profitability**: Your net profit margin of {:.1f}% is acceptable but could be improved.".format(net_profit_margin))
                else:
                    assessment.append("❗ **Low profitability**: Your net profit margin of {:.1f}% indicates profitability challenges.".format(net_profit_margin))
                
                # Efficiency assessment
                if return_on_assets > 10:
                    assessment.append("✅ **Strong ROA**: Your return on assets of {:.1f}% shows excellent asset utilization.".format(return_on_assets))
                elif return_on_assets > 5:
                    assessment.append("✓ **Good ROA**: Your return on assets of {:.1f}% indicates good asset utilization.".format(return_on_assets))
                else:
                    assessment.append("⚠️ **Low ROA**: Your return on assets of {:.1f}% suggests inefficient asset utilization.".format(return_on_assets))
                
                if return_on_equity > 20:
                    assessment.append("✅ **Excellent ROE**: Your return on equity of {:.1f}% indicates very strong returns for shareholders.".format(return_on_equity))
                elif return_on_equity > 15:
                    assessment.append("✓ **Good ROE**: Your return on equity of {:.1f}% shows healthy returns for shareholders.".format(return_on_equity))
                else:
                    assessment.append("⚠️ **Low ROE**: Your return on equity of {:.1f}% suggests suboptimal returns for shareholders.".format(return_on_equity))
                
                # Liquidity assessment
                if current_ratio >= 2:
                    assessment.append("✅ **Strong liquidity**: Your current ratio of {:.2f}x indicates excellent short-term financial health.".format(current_ratio))
                elif current_ratio >= 1:
                    assessment.append("✓ **Adequate liquidity**: Your current ratio of {:.2f}x shows you can meet short-term obligations.".format(current_ratio))
                else:
                    assessment.append("❗ **Liquidity concern**: Your current ratio of {:.2f}x suggests potential difficulty meeting short-term obligations.".format(current_ratio))
                
                # Solvency assessment
                if debt_to_equity < 0.5:
                    assessment.append("✅ **Low leverage**: Your debt-to-equity ratio of {:.2f}x indicates conservative financial leverage.".format(debt_to_equity))
                elif debt_to_equity < 1.5:
                    assessment.append("✓ **Moderate leverage**: Your debt-to-equity ratio of {:.2f}x shows reasonable financial leverage.".format(debt_to_equity))
                else:
                    assessment.append("⚠️ **High leverage**: Your debt-to-equity ratio of {:.2f}x indicates high financial risk.".format(debt_to_equity))
                
                # Display assessment
                for item in assessment:
                    st.markdown(item)
    
    with fin_tab3:
        st.markdown("#### Break-even Analysis")
        st.markdown("Calculate how many units you need to sell to cover your costs.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fixed_costs = st.number_input("Fixed Costs ($)", min_value=0.0, value=50000.0, step=1000.0)
            variable_cost_per_unit = st.number_input("Variable Cost per Unit ($)", min_value=0.0, value=15.0, step=1.0)
            selling_price_per_unit = st.number_input("Selling Price per Unit ($)", min_value=0.1, value=25.0, step=1.0)
        
        with col2:
            # Current sales (optional)
            current_units_sold = st.number_input("Current Units Sold (optional)", min_value=0, value=5000, step=100)
            st.markdown("Leave at 0 if this is a new product.")
            
            # Target profit (optional)
            target_profit = st.number_input("Target Profit ($, optional)", min_value=0.0, value=20000.0, step=1000.0)
            st.markdown("Enter how much profit you're aiming for.")
        
        # Calculate break-even point
        if st.button("Calculate Break-even Point"):
            # Calculate contribution margin per unit
            contribution_margin = selling_price_per_unit - variable_cost_per_unit
            
            if contribution_margin <= 0:
                st.error("Contribution margin must be positive. Your selling price needs to be higher than your variable cost.")
            else:
                # Calculate break-even point in units
                break_even_units = fixed_costs / contribution_margin
                
                # Calculate break-even point in dollars
                break_even_dollars = break_even_units * selling_price_per_unit
                
                # Calculate profit at current sales level
                current_profit = (current_units_sold * contribution_margin) - fixed_costs
                
                # Calculate units needed for target profit
                target_profit_units = (fixed_costs + target_profit) / contribution_margin
                
                # Display results
                st.markdown("#### Break-even Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Break-even Units", f"{break_even_units:.0f} units")
                
                with col2:
                    st.metric("Break-even Revenue", f"${break_even_dollars:,.2f}")
                
                with col3:
                    st.metric("Contribution Margin", f"${contribution_margin:.2f} per unit")
                
                # Display current status and target
                st.markdown("#### Current Status & Targets")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if current_units_sold > 0:
                        profit_status = "Profit" if current_profit > 0 else "Loss"
                        profit_color = "green" if current_profit > 0 else "red"
                        
                        st.metric(
                            "Current Profit/Loss",
                            f"${current_profit:,.2f}",
                            delta=f"{profit_status}: {abs(current_profit/fixed_costs*100):.1f}% of fixed costs",
                            delta_color="normal"  # Using 'normal' instead of custom colors
                        )
                        
                        # Percentage of break-even
                        pct_of_breakeven = (current_units_sold / break_even_units) * 100
                        st.metric("% of Break-even", f"{pct_of_breakeven:.1f}%")
                    else:
                        st.info("Enter current units sold to see profit status.")
                
                with col2:
                    if target_profit > 0:
                        st.metric("Units for Target Profit", f"{target_profit_units:.0f} units")
                        st.metric("Target Profit Revenue", f"${target_profit_units * selling_price_per_unit:,.2f}")
                    else:
                        st.info("Enter target profit to see required units.")
                
                # Create break-even chart
                st.markdown("#### Break-even Chart")
                
                # Generate data points
                max_units = max(break_even_units * 2, current_units_sold * 1.5, target_profit_units * 1.2)
                units = np.linspace(0, max_units, 100)
                fixed_cost_line = np.full_like(units, fixed_costs)
                variable_costs = units * variable_cost_per_unit
                total_costs = fixed_costs + variable_costs
                revenue = units * selling_price_per_unit
                profit = revenue - total_costs
                
                # Create figure
                fig = go.Figure()
                
                # Add revenue line
                fig.add_trace(go.Scatter(
                    x=units,
                    y=revenue,
                    mode='lines',
                    name='Revenue',
                    line=dict(color='green', width=2)
                ))
                
                # Add total cost line
                fig.add_trace(go.Scatter(
                    x=units,
                    y=total_costs,
                    mode='lines',
                    name='Total Costs',
                    line=dict(color='red', width=2)
                ))
                
                # Add fixed cost line
                fig.add_trace(go.Scatter(
                    x=units,
                    y=fixed_cost_line,
                    mode='lines',
                    name='Fixed Costs',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                # Add break-even point
                fig.add_trace(go.Scatter(
                    x=[break_even_units],
                    y=[break_even_dollars],
                    mode='markers',
                    marker=dict(size=12, color='blue', symbol='star'),
                    name='Break-even Point',
                    hoverinfo='text',
                    text=f'Break-even at {break_even_units:.0f} units (${break_even_dollars:,.2f})'
                ))
                
                # Add current position if provided
                if current_units_sold > 0:
                    current_revenue = current_units_sold * selling_price_per_unit
                    current_total_cost = fixed_costs + (current_units_sold * variable_cost_per_unit)
                    
                    fig.add_trace(go.Scatter(
                        x=[current_units_sold],
                        y=[current_revenue],
                        mode='markers',
                        marker=dict(size=12, color='purple'),
                        name='Current Position',
                        hoverinfo='text',
                        text=f'Current: {current_units_sold} units (${current_revenue:,.2f})'
                    ))
                
                # Add target profit point if provided
                if target_profit > 0:
                    target_revenue = target_profit_units * selling_price_per_unit
                    
                    fig.add_trace(go.Scatter(
                        x=[target_profit_units],
                        y=[target_revenue],
                        mode='markers',
                        marker=dict(size=12, color='green'),
                        name='Target Profit Point',
                        hoverinfo='text',
                        text=f'Target: {target_profit_units:.0f} units (${target_revenue:,.2f})'
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Break-even Analysis",
                    xaxis_title="Units Sold",
                    yaxis_title="Amount ($)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Profit chart
                st.markdown("#### Profit Analysis")
                
                # Create figure
                fig = go.Figure()
                
                # Add profit line
                fig.add_trace(go.Scatter(
                    x=units,
                    y=profit,
                    mode='lines',
                    name='Profit',
                    line=dict(color='blue', width=3)
                ))
                
                # Add zero line
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray"
                )
                
                # Add break-even point
                fig.add_trace(go.Scatter(
                    x=[break_even_units],
                    y=[0],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star'),
                    name='Break-even Point',
                    hoverinfo='text',
                    text=f'Break-even at {break_even_units:.0f} units'
                ))
                
                # Add current position if provided
                if current_units_sold > 0:
                    fig.add_trace(go.Scatter(
                        x=[current_units_sold],
                        y=[current_profit],
                        mode='markers',
                        marker=dict(size=12, color='purple'),
                        name='Current Profit',
                        hoverinfo='text',
                        text=f'Current Profit: ${current_profit:,.2f}'
                    ))
                
                # Add target profit point if provided
                if target_profit > 0:
                    fig.add_trace(go.Scatter(
                        x=[target_profit_units],
                        y=[target_profit],
                        mode='markers',
                        marker=dict(size=12, color='green'),
                        name='Target Profit',
                        hoverinfo='text',
                        text=f'Target Profit: ${target_profit:,.2f}'
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Profit by Units Sold",
                    xaxis_title="Units Sold",
                    yaxis_title="Profit ($)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Margin of safety analysis
                if current_units_sold > break_even_units:
                    margin_of_safety_units = current_units_sold - break_even_units
                    margin_of_safety_revenue = margin_of_safety_units * selling_price_per_unit
                    margin_of_safety_percent = (margin_of_safety_units / current_units_sold) * 100
                    
                    st.markdown("#### Margin of Safety Analysis")
                    st.markdown(f"""
                    Your current sales level provides a margin of safety of **{margin_of_safety_units:.0f} units** 
                    (${margin_of_safety_revenue:,.2f}), which is **{margin_of_safety_percent:.1f}%** of your current sales volume.
                    
                    This means your sales could drop by {margin_of_safety_percent:.1f}% before reaching the break-even point.
                    """)
                    
                    if margin_of_safety_percent < 10:
                        st.warning("Your margin of safety is quite low. Consider strategies to increase sales, raise prices, or reduce costs.")
                    elif margin_of_safety_percent > 30:
                        st.success("You have a healthy margin of safety, indicating good resilience to sales fluctuations.")
                
                # Break-even insights
                st.markdown("#### Break-even Insights")
                
                insights = []
                
                # Price sensitivity
                price_increase_1pct = (fixed_costs / (contribution_margin * 1.01)) - break_even_units
                insights.append(f"A 1% price increase would reduce your break-even point by {abs(price_increase_1pct):.0f} units.")
                
                # Fixed cost sensitivity
                fixed_cost_decrease_10pct = (fixed_costs * 0.9 / contribution_margin) - break_even_units
                insights.append(f"A 10% reduction in fixed costs would reduce your break-even point by {abs(fixed_cost_decrease_10pct):.0f} units.")
                
                # Variable cost sensitivity
                new_cm = selling_price_per_unit - (variable_cost_per_unit * 0.9)
                var_cost_decrease_10pct = (fixed_costs / new_cm) - break_even_units
                insights.append(f"A 10% reduction in variable costs would reduce your break-even point by {abs(var_cost_decrease_10pct):.0f} units.")
                
                # Display insights
                for insight in insights:
                    st.markdown(f"- {insight}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Customer Segmentation")
    st.markdown("""
    Use clustering algorithms to segment your customers based on behavior, demographics, or other attributes.
    This helps target marketing efforts and personalize customer experiences.
    """)
    
    # Check if data is available
    if st.session_state.cleaned_data is not None:
        # Get numeric columns for clustering
        numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.warning("Customer segmentation requires at least 2 numeric columns for clustering.")
        else:
            # Select features for clustering
            st.markdown("#### Select Features for Customer Segmentation")
            
            selected_features = st.multiselect(
                "Select numeric features for segmentation:",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )
            
            if len(selected_features) < 2:
                st.warning("Please select at least 2 features for clustering.")
            else:
                # Select number of clusters
                num_clusters = st.slider(
                    "Number of Customer Segments:",
                    min_value=2,
                    max_value=10,
                    value=3
                )
                
                # Optional categorical features for profiling
                profiling_feature = st.selectbox(
                    "Select a categorical feature for segment profiling (optional):",
                    ["None"] + categorical_columns
                )
                
                # Run clustering
                if st.button("Run Customer Segmentation"):
                    # Get clean data for selected features
                    cluster_data = st.session_state.cleaned_data[selected_features].dropna()
                    
                    if len(cluster_data) < num_clusters * 5:
                        st.error(f"Insufficient data for clustering with {num_clusters} segments. Need at least {num_clusters * 5} complete rows.")
                    else:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("Standardizing data...")
                        progress_bar.progress(10)
                        
                        # Standardize the data
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(cluster_data)
                        
                        # Update progress
                        status_text.text("Running K-means clustering...")
                        progress_bar.progress(30)
                        
                        # Run K-means clustering
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(scaled_data)
                        
                        # Update progress
                        status_text.text("Analyzing results...")
                        progress_bar.progress(60)
                        
                        # Add cluster labels to original data
                        cluster_data['Cluster'] = clusters
                        
                        # If we have a profiling feature, add it
                        if profiling_feature != "None":
                            # Get the profiling feature values for rows in cluster_data
                            profile_values = st.session_state.cleaned_data.loc[cluster_data.index, profiling_feature]
                            cluster_data[profiling_feature] = profile_values
                        
                        # Calculate cluster statistics
                        cluster_stats = []
                        
                        for i in range(num_clusters):
                            cluster_i_data = cluster_data[cluster_data['Cluster'] == i]
                            stats = {
                                'Cluster': i,
                                'Size': len(cluster_i_data),
                                'Percentage': 100 * len(cluster_i_data) / len(cluster_data)
                            }
                            
                            # Calculate statistics for each feature
                            for feature in selected_features:
                                stats[f'{feature}_mean'] = cluster_i_data[feature].mean()
                                stats[f'{feature}_median'] = cluster_i_data[feature].median()
                                stats[f'{feature}_std'] = cluster_i_data[feature].std()
                            
                            # Profiling feature distribution if available
                            if profiling_feature != "None":
                                profile_dist = cluster_i_data[profiling_feature].value_counts().to_dict()
                                stats[f'{profiling_feature}_distribution'] = profile_dist
                            
                            cluster_stats.append(stats)
                        
                        # Update progress
                        status_text.text("Creating visualizations...")
                        progress_bar.progress(80)
                        
                        # Prepare for plotting
                        if len(selected_features) >= 2:
                            # Use the first two features for 2D plot
                            plot_x = selected_features[0]
                            plot_y = selected_features[1]
                            
                            # Create scatter plot
                            fig = px.scatter(
                                cluster_data, 
                                x=plot_x, 
                                y=plot_y,
                                color='Cluster',
                                color_continuous_scale=px.colors.qualitative.G10,
                                hover_data=selected_features,
                                title=f"Customer Segments by {plot_x} and {plot_y}"
                            )
                            
                            # Add cluster centers
                            cluster_centers = pd.DataFrame(
                                scaler.inverse_transform(kmeans.cluster_centers_),
                                columns=selected_features
                            )
                            
                            for i in range(num_clusters):
                                fig.add_trace(go.Scatter(
                                    x=[cluster_centers.iloc[i][plot_x]],
                                    y=[cluster_centers.iloc[i][plot_y]],
                                    mode='markers',
                                    marker=dict(
                                        color='black',
                                        symbol='x',
                                        size=15,
                                        line=dict(
                                            color='white',
                                            width=1
                                        )
                                    ),
                                    name=f'Cluster {i} Center'
                                ))
                            
                            # Update layout
                            fig.update_layout(
                                height=600,
                                legend_title="Cluster",
                                coloraxis_showscale=False
                            )
                        
                        # Create 3D plot if we have 3+ features
                        fig_3d = None
                        if len(selected_features) >= 3:
                            plot_z = selected_features[2]
                            
                            fig_3d = px.scatter_3d(
                                cluster_data,
                                x=plot_x,
                                y=plot_y,
                                z=plot_z,
                                color='Cluster',
                                color_continuous_scale=px.colors.qualitative.G10,
                                hover_data=selected_features,
                                title=f"3D Customer Segments by {plot_x}, {plot_y}, and {plot_z}"
                            )
                            
                            # Add cluster centers to 3D plot
                            for i in range(num_clusters):
                                fig_3d.add_trace(go.Scatter3d(
                                    x=[cluster_centers.iloc[i][plot_x]],
                                    y=[cluster_centers.iloc[i][plot_y]],
                                    z=[cluster_centers.iloc[i][plot_z]],
                                    mode='markers',
                                    marker=dict(
                                        color='black',
                                        symbol='x',
                                        size=8,
                                        line=dict(
                                            color='white',
                                            width=1
                                        )
                                    ),
                                    name=f'Cluster {i} Center'
                                ))
                            
                            # Update layout
                            fig_3d.update_layout(
                                height=700,
                                legend_title="Cluster",
                                coloraxis_showscale=False
                            )
                        
                        # Create parallel coordinates plot for multiple features
                        dimensions = []
                        for feature in selected_features:
                            dimensions.append(
                                dict(
                                    range=[cluster_data[feature].min(), cluster_data[feature].max()],
                                    label=feature,
                                    values=cluster_data[feature].values
                                )
                            )
                        
                        fig_parallel = go.Figure(data=
                            go.Parcoords(
                                line=dict(
                                    color=cluster_data['Cluster'],
                                    colorscale='Jet',
                                    showscale=True,
                                    cmin=0,
                                    cmax=num_clusters-1
                                ),
                                dimensions=dimensions
                            )
                        )
                        
                        fig_parallel.update_layout(
                            title="Parallel Coordinates Plot of Customer Segments",
                            height=500
                        )
                        
                        # Update progress
                        status_text.text("Complete!")
                        progress_bar.progress(100)
                        
                        # Clear progress bar and status text
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        st.markdown("#### Customer Segmentation Results")
                        st.markdown(f"Successfully segmented customers into {num_clusters} groups.")
                        
                        # Display cluster sizes
                        cluster_sizes = pd.DataFrame([
                            {
                                'Cluster': i,
                                'Size': stats['Size'],
                                'Percentage': f"{stats['Percentage']:.1f}%"
                            }
                            for i, stats in enumerate(cluster_stats)
                        ])
                        
                        st.markdown("##### Segment Sizes")
                        st.dataframe(cluster_sizes)
                        
                        # Create a pie chart of cluster sizes
                        fig_pie = px.pie(
                            cluster_sizes,
                            values='Size',
                            names='Cluster',
                            title='Customer Segment Distribution'
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Display scatter plot
                        st.markdown("##### Customer Segment Visualization")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display 3D scatter plot if available
                        if fig_3d is not None:
                            st.markdown("##### 3D Customer Segment Visualization")
                            st.markdown("*Tip: Click and drag to rotate the 3D plot. Double-click to reset the view.*")
                            st.plotly_chart(fig_3d, use_container_width=True)
                        
                        # Display parallel coordinates plot
                        st.markdown("##### Feature Comparison Across Segments")
                        st.markdown("*This plot shows how feature values differ across segments. Each line represents a customer.*")
                        st.plotly_chart(fig_parallel, use_container_width=True)
                        
                        # Display segment profiles
                        st.markdown("#### Segment Profiles")
                        
                        # Create a DataFrame with cluster means for all features
                        profile_data = []
                        
                        for i, stats in enumerate(cluster_stats):
                            profile = {'Cluster': f"Segment {i}"}
                            
                            for feature in selected_features:
                                profile[feature] = stats[f'{feature}_mean']
                            
                            profile_data.append(profile)
                        
                        profile_df = pd.DataFrame(profile_data)
                        
                        # Display profile dataframe
                        st.dataframe(profile_df.round(2), use_container_width=True)
                        
                        # Radar chart for segment profiles
                        st.markdown("##### Segment Profile Comparison")
                        
                        # Create radar chart data
                        fig_radar = go.Figure()
                        
                        # Normalize features for radar chart
                        max_values = cluster_data[selected_features].max()
                        min_values = cluster_data[selected_features].min()
                        
                        for i, profile in profile_df.iterrows():
                            radar_values = []
                            
                            for feature in selected_features:
                                # Normalize to 0-1 scale
                                if max_values[feature] > min_values[feature]:
                                    normalized = (profile[feature] - min_values[feature]) / (max_values[feature] - min_values[feature])
                                else:
                                    normalized = 0.5
                                radar_values.append(normalized)
                            
                            # Add trace
                            fig_radar.add_trace(go.Scatterpolar(
                                r=radar_values,
                                theta=selected_features,
                                fill='toself',
                                name=f'Segment {i}'
                            ))
                        
                        # Update layout
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Segment Profile Comparison",
                            height=500
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Feature importance for each segment
                        st.markdown("##### Key Distinguishing Features")
                        
                        # Calculate feature importance based on distance from the overall mean
                        overall_means = cluster_data[selected_features].mean()
                        
                        for i, stats in enumerate(cluster_stats):
                            st.markdown(f"**Segment {i}** ({stats['Size']} customers, {stats['Percentage']:.1f}%)")
                            
                            # Calculate importance scores
                            importance = {}
                            for feature in selected_features:
                                segment_mean = stats[f'{feature}_mean']
                                overall_mean = overall_means[feature]
                                std = cluster_data[feature].std()
                                
                                if std > 0:
                                    # Z-score distance from overall mean
                                    importance[feature] = abs(segment_mean - overall_mean) / std
                                else:
                                    importance[feature] = 0
                            
                            # Sort features by importance
                            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                            
                            # Create description
                            description = []
                            for feature, score in sorted_features[:3]:  # Top 3 features
                                segment_mean = stats[f'{feature}_mean']
                                overall_mean = overall_means[feature]
                                
                                if segment_mean > overall_mean:
                                    comparison = "higher"
                                else:
                                    comparison = "lower"
                                
                                # Calculate percentage difference
                                if overall_mean != 0:
                                    pct_diff = abs((segment_mean - overall_mean) / overall_mean) * 100
                                    description.append(f"**{feature}** is {pct_diff:.1f}% {comparison} than average ({segment_mean:.2f} vs {overall_mean:.2f} overall)")
                                else:
                                    description.append(f"**{feature}** is {comparison} than average ({segment_mean:.2f} vs {overall_mean:.2f} overall)")
                            
                            # Display description
                            for item in description:
                                st.markdown(f"- {item}")
                            
                            # Show profiling feature distribution if available
                            if profiling_feature != "None" and f'{profiling_feature}_distribution' in stats:
                                st.markdown(f"**{profiling_feature} Distribution:**")
                                
                                dist = stats[f'{profiling_feature}_distribution']
                                total = sum(dist.values())
                                
                                # Sort by frequency
                                sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                                
                                # Take top 5
                                top_dist = sorted_dist[:5]
                                
                                for category, count in top_dist:
                                    percentage = (count / total) * 100
                                    st.markdown(f"- {category}: {percentage:.1f}% ({count} customers)")
                            
                            st.markdown("---")
                        
                        # Generate segment names with AI if possible
                        try:
                            if 'ai_analytics' not in locals():
                                ai_analytics = AIAnalytics(st.session_state.cleaned_data)
                            
                            # Only attempt if OpenAI API key is available
                            if ai_analytics.has_openai or ai_analytics.has_anthropic:
                                st.markdown("#### AI-Generated Segment Names and Descriptions")
                                
                                # Generate segment descriptions
                                segment_descriptions = []
                                
                                for i, stats in enumerate(cluster_stats):
                                    # Prepare feature information
                                    feature_info = []
                                    for feature in selected_features:
                                        segment_mean = stats[f'{feature}_mean']
                                        overall_mean = overall_means[feature]
                                        
                                        if segment_mean > overall_mean:
                                            comparison = "higher"
                                        else:
                                            comparison = "lower"
                                        
                                        # Calculate percentage difference
                                        if overall_mean != 0:
                                            pct_diff = abs((segment_mean - overall_mean) / overall_mean) * 100
                                            feature_info.append(f"{feature} is {pct_diff:.1f}% {comparison} than average ({segment_mean:.2f} vs {overall_mean:.2f} overall)")
                                        else:
                                            feature_info.append(f"{feature} is {comparison} than average ({segment_mean:.2f} vs {overall_mean:.2f} overall)")
                                    
                                    # Combine information
                                    segment_info = {
                                        'segment_id': i,
                                        'size': stats['Size'],
                                        'percentage': stats['Percentage'],
                                        'features': feature_info
                                    }
                                    
                                    segment_descriptions.append(segment_info)
                                
                                # Generate names and descriptions using AI
                                ai_results = ai_analytics.generate_insights_with_ai(max_insights=num_clusters)
                                
                                if ai_results and len(ai_results) == num_clusters:
                                    for i, desc in enumerate(ai_results):
                                        st.markdown(f"**Segment {i}**: {desc}")
                                else:
                                    st.info("AI-generated segment names are not available. Check your OpenAI API key or try again later.")
                        except Exception as e:
                            st.info(f"AI-generated segment names are not available: {str(e)}")
                        
                        # Export options
                        st.markdown("#### Export Options")
                        
                        # Add cluster labels to the original data
                        export_data = st.session_state.cleaned_data.copy()
                        export_data['Customer_Segment'] = None
                        
                        # Map cluster labels to rows that were used in clustering
                        for idx, cluster_label in zip(cluster_data.index, cluster_data['Cluster']):
                            export_data.loc[idx, 'Customer_Segment'] = f"Segment {cluster_label}"
                        
                        # Convert to CSV
                        csv = export_data.to_csv(index=False)
                        
                        # Create download button
                        st.download_button(
                            label="Download Segmented Data as CSV",
                            data=csv,
                            file_name="customer_segments.csv",
                            mime="text/csv"
                        )
    else:
        st.warning("Please upload data in the main dashboard before performing customer segmentation.")
    
    st.markdown('</div>', unsafe_allow_html=True)