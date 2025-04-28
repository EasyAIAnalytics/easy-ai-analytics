import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer
from utils.pdf_generator import PDFGenerator

# Page configuration
st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Business Analytics & Data Analytics")

# Create assets directory if it doesn't exist
if not os.path.exists("assets"):
    os.makedirs("assets")

# Use the logo in the sidebar if it exists
if os.path.exists("assets/logo.svg"):
    with st.sidebar:
        st.image("assets/logo.svg", width=100)

with st.expander("How It Works (explained simply):", expanded=True):
    st.markdown("""
    1. User uploads a CSV file
    2. System shows rows, columns, missing values (bar chart)
    3. System shows column distributions (pie chart)
    4. User can clean missing data
    5. Auto-insights suggest what the data needs
    6. User fills a business requirement form
    7. User can download filled form as beautiful PDF
    """)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False

# Function to create sample data
def create_sample_data():
    # Create a sample dataset with various data types and some missing values
    np.random.seed(42)
    
    # Sample size
    n = 100
    
    # Create sales data
    data = {
        'Date': pd.date_range(start='2024-01-01', periods=n),
        'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'], n),
        'Category': np.random.choice(['Electronics', 'Office Supplies', 'Furniture', 'Apparel'], n),
        'Price': np.round(np.random.uniform(10, 500, n), 2),
        'Quantity': np.random.randint(1, 30, n),
        'Customer_ID': np.random.randint(1000, 9999, n),
        'Customer_Age': np.random.randint(18, 80, n),
        'Customer_Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Satisfaction': np.random.randint(1, 6, n),
        'Discount': np.round(np.random.uniform(0, 0.5, n), 2)
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate total sale amount
    df['Sale_Amount'] = df['Price'] * df['Quantity'] * (1 - df['Discount'])
    
    # Add some missing values
    for col in ['Price', 'Customer_Age', 'Satisfaction', 'Discount']:
        mask = np.random.choice([True, False], len(df), p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    return df

# Sidebar for file upload and operations
with st.sidebar:
    st.subheader("1. Get Your Data")
    
    # Two tabs for uploading or using sample data
    data_tab1, data_tab2 = st.tabs(["Upload CSV", "Use Sample Data"])
    
    with data_tab1:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                # Store in session state
                st.session_state.data = df
                st.session_state.cleaned_data = df.copy()
                st.session_state.sample_data_loaded = False
                st.success(f"Successfully loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
    
    with data_tab2:
        if st.button("Load Sample Data"):
            try:
                # Create sample data
                df = create_sample_data()
                # Store in session state
                st.session_state.data = df
                st.session_state.cleaned_data = df.copy()
                st.session_state.sample_data_loaded = True
                st.success(f"Successfully loaded sample data with {df.shape[0]} rows and {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error creating sample data: {e}")
    
    # Only show these options if data is loaded
    if st.session_state.data is not None:
        st.subheader("2. Data Operations")
        
        # Data cleaning options
        clean_option = st.selectbox(
            "Clean missing values",
            ["None", "Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode", "Fill with zero"]
        )
        
        if clean_option != "None":
            data_processor = DataProcessor(st.session_state.data)
            
            if clean_option == "Drop rows with missing values":
                st.session_state.cleaned_data = data_processor.drop_missing_rows()
            elif clean_option == "Fill with mean":
                st.session_state.cleaned_data = data_processor.fill_missing_values("mean")
            elif clean_option == "Fill with median":
                st.session_state.cleaned_data = data_processor.fill_missing_values("median")
            elif clean_option == "Fill with mode":
                st.session_state.cleaned_data = data_processor.fill_missing_values("mode")
            elif clean_option == "Fill with zero":
                st.session_state.cleaned_data = data_processor.fill_missing_values("zero")
            
            st.success("Data cleaning applied!")
            
        # Outlier detection
        st.subheader("3. Outlier Detection")
        
        if st.session_state.cleaned_data is not None:
            numeric_cols = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numeric_cols:
                outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)
                outlier_method = st.selectbox("Select outlier detection method", ["zscore", "iqr", "isolation_forest"])
                
                if st.button("Detect Outliers"):
                    try:
                        data_processor = DataProcessor(st.session_state.cleaned_data)
                        outliers = data_processor.detect_outliers(outlier_col, method=outlier_method)
                        st.info(f"Found {outliers.sum()} outliers in column '{outlier_col}' using {outlier_method} method")
                    except Exception as e:
                        st.error(f"Error detecting outliers: {e}")

# Main content area - Only display if data is uploaded
if st.session_state.data is not None:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Visualizations", "Insights", "Report Generation"])
    
    with tab1:
        st.header("Data Overview")
        
        # Data summary
        st.subheader("Data Sample")
        st.dataframe(st.session_state.cleaned_data.head(5))
        
        # Data statistics
        st.subheader("Data Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", st.session_state.cleaned_data.shape[0])
            st.metric("Total Columns", st.session_state.cleaned_data.shape[1])
        with col2:
            st.metric("Missing Values", st.session_state.cleaned_data.isna().sum().sum())
            st.metric("Duplicate Rows", st.session_state.cleaned_data.duplicated().sum())
        
        # Display column info
        st.subheader("Column Information")
        column_info = pd.DataFrame({
            'Column': st.session_state.cleaned_data.columns,
            'Data Type': st.session_state.cleaned_data.dtypes.values,
            'Missing Values': st.session_state.cleaned_data.isna().sum().values,
            'Unique Values': [st.session_state.cleaned_data[col].nunique() for col in st.session_state.cleaned_data.columns]
        })
        st.dataframe(column_info)
    
    with tab2:
        st.header("Data Visualizations")
        
        # Create visualization object
        visualizer = Visualizer(st.session_state.cleaned_data)
        
        # Create tabs for different visualization types
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Basic Distributions", 
            "Time Series", 
            "Correlations", 
            "Custom Charts"
        ])
        
        with viz_tab1:
            # Missing values visualization
            st.subheader("Missing Values by Column")
            missing_chart = visualizer.plot_missing_values()
            st.plotly_chart(missing_chart, use_container_width=True)
            
            # Column distribution for a selected column
            st.subheader("Column Distributions")
            numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if numeric_columns:
                    selected_num_col = st.selectbox("Select Numeric Column", numeric_columns)
                    num_dist_chart = visualizer.plot_numeric_distribution(selected_num_col)
                    st.plotly_chart(num_dist_chart, use_container_width=True)
            
            with col2:
                if categorical_columns:
                    selected_cat_col = st.selectbox("Select Categorical Column", categorical_columns)
                    cat_dist_chart = visualizer.plot_categorical_distribution(selected_cat_col)
                    st.plotly_chart(cat_dist_chart, use_container_width=True)
        
        with viz_tab2:
            # Time series analysis if Date column exists
            st.subheader("Time Series Analysis")
            
            date_columns = []
            for col in st.session_state.cleaned_data.columns:
                if pd.api.types.is_datetime64_any_dtype(st.session_state.cleaned_data[col]) or 'date' in col.lower():
                    date_columns.append(col)
            
            if date_columns:
                date_col = st.selectbox("Select Date Column", date_columns)
                
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(st.session_state.cleaned_data[date_col]):
                    try:
                        # Try to convert to datetime
                        date_series = pd.to_datetime(st.session_state.cleaned_data[date_col])
                        st.session_state.cleaned_data[date_col] = date_series
                        st.success(f"Converted {date_col} to datetime format")
                    except:
                        st.warning(f"Could not convert {date_col} to datetime format")
                
                # Select a metric to plot over time
                if numeric_columns:
                    metric_col = st.selectbox("Select Metric to Plot Over Time", numeric_columns)
                    
                    # Create time series chart
                    time_data = st.session_state.cleaned_data.groupby(date_col)[metric_col].mean().reset_index()
                    fig = px.line(
                        time_data, 
                        x=date_col, 
                        y=metric_col,
                        title=f"{metric_col} Over Time",
                        labels={metric_col: metric_col, date_col: date_col},
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show monthly trend if we have enough data
                    if len(time_data) > 10:
                        st.subheader("Monthly Trend")
                        try:
                            time_data['month'] = pd.to_datetime(time_data[date_col]).dt.month_name()
                            monthly_data = time_data.groupby('month')[metric_col].mean().reset_index()
                            fig = px.bar(
                                monthly_data,
                                x='month',
                                y=metric_col,
                                title=f"Average {metric_col} by Month",
                                color=metric_col,
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.info("Not enough data for monthly trend analysis")
            else:
                st.info("No date columns detected in your data. To use time series analysis, your data should include a date column.")
        
        with viz_tab3:
            # Correlation matrix for numeric columns
            st.subheader("Correlation Analysis")
            
            if len(numeric_columns) > 1:
                corr_chart = visualizer.plot_correlation_matrix()
                st.plotly_chart(corr_chart, use_container_width=True)
                
                # Scatter plot between two variables
                st.subheader("Relationship Between Variables")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Select X-axis Variable", numeric_columns)
                
                with col2:
                    remaining_cols = [col for col in numeric_columns if col != x_col]
                    y_col = st.selectbox("Select Y-axis Variable", remaining_cols)
                
                # Optional color by categorical variable
                color_by = None
                if categorical_columns:
                    color_col = st.selectbox("Color Points By (Optional)", ["None"] + categorical_columns)
                    if color_col != "None":
                        color_by = color_col
                
                # Create scatter plot
                scatter_fig = visualizer.plot_scatter(x_col, y_col, color_column=color_by)
                st.plotly_chart(scatter_fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns for correlation analysis.")
        
        with viz_tab4:
            st.subheader("Custom Visualization")
            
            chart_type = st.selectbox("Select Chart Type", [
                "Bar Chart", 
                "Line Chart", 
                "Scatter Plot",
                "Pie Chart",
                "Box Plot",
                "Heatmap"
            ])
            
            if chart_type == "Bar Chart":
                if categorical_columns:
                    x_axis = st.selectbox("Select X-axis (Categories)", categorical_columns)
                    if numeric_columns:
                        y_axis = st.selectbox("Select Y-axis (Values)", numeric_columns)
                        agg_func = st.selectbox("Aggregation Function", ["Sum", "Mean", "Count", "Max", "Min"])
                        
                        # Prepare data
                        agg_map = {
                            "Sum": "sum",
                            "Mean": "mean",
                            "Count": "count",
                            "Max": "max",
                            "Min": "min"
                        }
                        
                        grouped_data = st.session_state.cleaned_data.groupby(x_axis)[y_axis].agg(agg_map[agg_func]).reset_index()
                        
                        # Create bar chart
                        fig = px.bar(
                            grouped_data,
                            x=x_axis,
                            y=y_axis,
                            title=f"{agg_func} of {y_axis} by {x_axis}",
                            color=y_axis,
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Need at least one numeric column for bar chart values.")
                else:
                    st.info("Need at least one categorical column for bar chart categories.")
            
            elif chart_type == "Line Chart":
                if numeric_columns:
                    y_columns = st.multiselect("Select Y-axis Variables", numeric_columns)
                    if y_columns:
                        index_col = st.selectbox("Select X-axis (Index)", st.session_state.cleaned_data.columns.tolist())
                        
                        # Create line chart
                        fig = px.line(
                            st.session_state.cleaned_data,
                            x=index_col,
                            y=y_columns,
                            title=f"Line Chart of {', '.join(y_columns)} by {index_col}",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least one Y-axis variable.")
                else:
                    st.info("Need at least one numeric column for line chart values.")
            
            elif chart_type == "Scatter Plot":
                if len(numeric_columns) >= 2:
                    x_axis = st.selectbox("Select X-axis", numeric_columns)
                    y_axis = st.selectbox("Select Y-axis", [col for col in numeric_columns if col != x_axis])
                    
                    # Optional size and color
                    size_col = st.selectbox("Size Points By (Optional)", ["None"] + numeric_columns)
                    color_col = st.selectbox("Color Points By (Optional)", ["None"] + st.session_state.cleaned_data.columns.tolist())
                    
                    size = None if size_col == "None" else st.session_state.cleaned_data[size_col]
                    color = None if color_col == "None" else st.session_state.cleaned_data[color_col]
                    
                    # Create scatter plot
                    fig = px.scatter(
                        st.session_state.cleaned_data,
                        x=x_axis,
                        y=y_axis,
                        size=size,
                        color=color_col if color_col != "None" else None,
                        title=f"Scatter Plot of {y_axis} vs {x_axis}",
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least two numeric columns for scatter plot.")
            
            elif chart_type == "Pie Chart":
                if categorical_columns:
                    category_col = st.selectbox("Select Category Column", categorical_columns)
                    
                    # Optional value column
                    if numeric_columns:
                        has_values = st.checkbox("Use Values", value=False)
                        value_col = None
                        if has_values:
                            value_col = st.selectbox("Select Value Column", numeric_columns)
                    
                    # Prepare data
                    if has_values and value_col:
                        pie_data = st.session_state.cleaned_data.groupby(category_col)[value_col].sum().reset_index()
                        values = value_col
                    else:
                        pie_data = st.session_state.cleaned_data[category_col].value_counts().reset_index()
                        pie_data.columns = [category_col, 'count']
                        values = 'count'
                    
                    # Create pie chart
                    fig = px.pie(
                        pie_data,
                        names=category_col,
                        values=values,
                        title=f"Distribution of {category_col}",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least one categorical column for pie chart.")
            
            elif chart_type == "Box Plot":
                if numeric_columns:
                    y_axis = st.selectbox("Select Numeric Variable", numeric_columns)
                    
                    # Optional grouping
                    group_by = None
                    if categorical_columns:
                        use_grouping = st.checkbox("Group by Category", value=True)
                        if use_grouping:
                            group_by = st.selectbox("Select Grouping Variable", categorical_columns)
                    
                    # Create box plot
                    fig = px.box(
                        st.session_state.cleaned_data,
                        y=y_axis,
                        x=group_by,
                        title=f"Box Plot of {y_axis}" + (f" grouped by {group_by}" if group_by else ""),
                        color=group_by,
                        notched=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least one numeric column for box plot.")
            
            elif chart_type == "Heatmap":
                if len(numeric_columns) > 1:
                    selected_cols = st.multiselect("Select Numeric Columns", numeric_columns)
                    
                    if len(selected_cols) > 1:
                        # Create correlation heatmap
                        corr_data = st.session_state.cleaned_data[selected_cols].corr()
                        fig = px.imshow(
                            corr_data,
                            text_auto='.2f',
                            title="Correlation Heatmap",
                            color_continuous_scale="RdBu_r",
                            zmin=-1,
                            zmax=1
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least two columns for the heatmap.")
                else:
                    st.info("Need at least two numeric columns for heatmap.")
    
    with tab3:
        st.header("Auto-Generated Insights")
        
        if st.button("Generate Insights"):
            data_processor = DataProcessor(st.session_state.cleaned_data)
            st.session_state.insights = data_processor.generate_insights()
            st.success("Insights generated successfully!")
        
        if st.session_state.insights:
            for i, insight in enumerate(st.session_state.insights):
                with st.expander(f"Insight {i+1}", expanded=True):
                    st.write(insight)
    
    with tab4:
        st.header("Business Requirement Form")
        
        # Business requirement form
        with st.form("business_requirement_form"):
            st.subheader("Fill out the form for your report")
            
            st.session_state.form_data['company_name'] = st.text_input("Company Name")
            st.session_state.form_data['project_title'] = st.text_input("Project Title")
            st.session_state.form_data['objectives'] = st.text_area("Business Objectives")
            st.session_state.form_data['key_metrics'] = st.text_area("Key Metrics to Track")
            st.session_state.form_data['target_audience'] = st.text_input("Target Audience")
            st.session_state.form_data['timeframe'] = st.text_input("Analysis Timeframe")
            
            submitted = st.form_submit_button("Submit Form")
            
            if submitted:
                st.success("Form submitted successfully!")
        
        # Generate PDF report
        if st.session_state.form_data.get('company_name') and st.session_state.insights:
            if st.button("Generate PDF Report"):
                try:
                    visualizer = Visualizer(st.session_state.cleaned_data)
                    
                    # Generate plots for the PDF
                    missing_fig = visualizer.plot_missing_values()
                    
                    if numeric_columns:
                        num_fig = visualizer.plot_numeric_distribution(numeric_columns[0])
                    else:
                        num_fig = None
                        
                    if categorical_columns:
                        cat_fig = visualizer.plot_categorical_distribution(categorical_columns[0])
                    else:
                        cat_fig = None
                    
                    # Generate PDF
                    pdf_generator = PDFGenerator(
                        st.session_state.form_data,
                        st.session_state.cleaned_data,
                        st.session_state.insights,
                        missing_fig,
                        num_fig,
                        cat_fig
                    )
                    pdf_bytes = pdf_generator.generate_pdf()
                    
                    # Create download button
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{st.session_state.form_data['company_name']}_analytics_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
        else:
            st.info("Please fill out the form and generate insights before creating a PDF report.")

# What you get section
if st.session_state.data is not None:
    st.header("You now have:")
    
    st.markdown("""
    - A professional business analyst dashboard
    - A functional data analyst dashboard
    - A beautiful UI (Tailwind CSS based)
    - Real file download (PDF)
    - Smart insight generation
    - Advanced chart customization
    - Time series analysis
    - Correlation analysis tools
    - Outlier detection capabilities
    """)

# If no data is uploaded, display a welcome message
else:
    st.info("👆 Upload a CSV file or use our sample data to get started with your data analysis!")
    
    # Display some example features
    st.header("Features Available:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Analysis")
        st.markdown("""
        - Data overview and statistics
        - Missing value detection
        - Correlation analysis
        - Data cleaning tools
        - Outlier detection
        """)
    
    with col2:
        st.subheader("Visualizations")
        st.markdown("""
        - Interactive charts
        - Distribution analysis
        - Time series analysis
        - Custom chart builder
        - Correlation heatmaps
        """)
    
    with col3:
        st.subheader("Business Intelligence")
        st.markdown("""
        - Automated insights generation
        - Professional PDF reports
        - Business requirements form
        - Data statistics dashboard
        - Custom metric tracking
        """)
