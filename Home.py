import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer
from utils.pdf_generator import PDFGenerator
from utils.docx_generator import DocxGenerator
import database as db

# Page configuration
st.set_page_config(
    page_title="Easy AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('assets/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# App title and description with custom styling
st.markdown('<h1 class="main-header">Easy AI Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform your data into actionable insights with our AI-powered analytics platform</p>', unsafe_allow_html=True)

# Create assets directory if it doesn't exist
if not os.path.exists('assets'):
    os.makedirs('assets')

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
    
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
    
if "file_name" not in st.session_state:
    st.session_state.file_name = None
    
if "insights" not in st.session_state:
    st.session_state.insights = []

# Create a function to check DB connection
if "db_connection_checked" not in st.session_state:
    st.session_state.db_connection_checked = False
    st.session_state.db_connected = False

def check_db_connection():
    """Check database connection and update session state"""
    if not st.session_state.db_connection_checked:
        try:
            connection_status = db.check_database_connection()
            st.session_state.db_connected = connection_status
            st.session_state.db_connection_checked = True
        except Exception as e:
            st.session_state.db_connected = False
            st.session_state.db_connection_checked = True
            st.error(f"Database connection error: {str(e)}")

# Check database connection
check_db_connection()

# Create a function to load sample data
def create_sample_data():
    """Create sample data for demonstration"""
    # Sales data sample
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    regions = ['North', 'South', 'East', 'West']
    
    # Create sample data
    np.random.seed(42)  # For reproducible results
    sales_data = []
    
    for _ in range(500):
        date = np.random.choice(dates)
        product = np.random.choice(products)
        region = np.random.choice(regions)
        units_sold = np.random.randint(1, 50)
        unit_price = np.random.uniform(10, 100)
        total_sales = units_sold * unit_price
        customer_satisfaction = np.random.uniform(3, 5)
        
        sales_data.append({
            'Date': date,
            'Product': product,
            'Region': region,
            'Units Sold': units_sold,
            'Unit Price': round(unit_price, 2),
            'Total Sales': round(total_sales, 2),
            'Customer Satisfaction': round(customer_satisfaction, 1)
        })
    
    return pd.DataFrame(sales_data)

# Main layout with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Explore",
    "📊 Visualize",
    "📝 Generate Report",
    "💾 Saved Data"
])

# Tab 1: Upload & Explore
with tab1:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Upload Your Data")
    st.markdown("Upload your CSV or Excel file to get started, or use our sample data for a demo.")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Sample Data"):
            sample_data = create_sample_data()
            st.session_state.data = sample_data
            st.session_state.cleaned_data = sample_data  # Keep cleaned_data in sync
            st.session_state.file_name = "sample_sales_data.csv"
            st.success("Sample data loaded successfully!")
            
    with col2:
        if st.session_state.db_connected and st.button("Load From Database"):
            st.session_state.show_db_datasets = True
    
    # Show database datasets if requested
    if st.session_state.db_connected and st.session_state.get("show_db_datasets", False):
        st.markdown("### Saved Datasets")
        datasets = db.get_all_datasets()
        
        if datasets:
            selected_dataset = st.selectbox(
                "Select a saved dataset",
                options=[f"{d[1]} (ID: {d[0]})" for d in datasets],
                format_func=lambda x: x.split(" (ID: ")[0]
            )
            
            if st.button("Load Selected Dataset"):
                dataset_id = int(selected_dataset.split("ID: ")[1].rstrip(")"))
                loaded_df = db.get_dataset(dataset_id)
                if loaded_df is not None:
                    st.session_state.data = loaded_df
                    st.session_state.cleaned_data = loaded_df  # Keep cleaned_data in sync
                    st.session_state.file_name = next((d[3] for d in datasets if d[0] == dataset_id), "unknown.csv")
                    st.success(f"Dataset '{selected_dataset.split(' (ID: ')[0]}' loaded successfully!")
                else:
                    st.error("Failed to load dataset from database.")
        else:
            st.info("No datasets saved in the database yet.")
    
    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Check file type and read accordingly
            file_extension = uploaded_file.name.split(".")[-1]
            
            if file_extension.lower() == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() in ["xlsx", "xls"]:
                data = pd.read_excel(uploaded_file)
                
            st.session_state.data = data
            st.session_state.cleaned_data = data  # Keep cleaned_data in sync
            st.session_state.file_name = uploaded_file.name
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display data details and preview if data is loaded
    if st.session_state.data is not None:
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Data Preview")
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{st.session_state.data.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{st.session_state.data.shape[1]:,}")
        with col3:
            st.metric("File Size", f"{st.session_state.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data preview
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Column info
        st.markdown("### Column Information")
        
        col_data = []
        for col in st.session_state.data.columns:
            col_type = str(st.session_state.data[col].dtype)
            missing = st.session_state.data[col].isna().sum()
            missing_pct = (missing / len(st.session_state.data)) * 100
            unique = st.session_state.data[col].nunique()
            
            col_data.append({
                "Column Name": col,
                "Data Type": col_type,
                "Missing Values": missing,
                "Missing %": f"{missing_pct:.2f}%",
                "Unique Values": unique
            })
        
        col_df = pd.DataFrame(col_data)
        st.dataframe(col_df, use_container_width=True)
        
        # Data cleaning options
        st.markdown("### Data Cleaning")
        
        # Initialize DataProcessor
        data_processor = DataProcessor(st.session_state.data)
        
        # Show missing values options
        missing_cols = st.session_state.data.columns[st.session_state.data.isna().any()].tolist()
        
        if missing_cols:
            st.markdown("#### Handle Missing Values")
            
            missing_method = st.selectbox(
                "Select method for handling missing values:",
                options=["None", "Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode", "Fill with zero", "Fill with a specific value"]
            )
            
            if missing_method == "Drop rows with missing values":
                if st.button("Drop Missing Rows"):
                    cleaned_data = data_processor.drop_missing_rows()
                    st.session_state.data = cleaned_data
                    st.session_state.cleaned_data = cleaned_data  # Keep cleaned_data in sync
                    st.success(f"Dropped rows with missing values. New shape: {cleaned_data.shape}")
                    st.rerun()
            
            elif missing_method in ["Fill with mean", "Fill with median", "Fill with mode", "Fill with zero"]:
                method_map = {
                    "Fill with mean": "mean",
                    "Fill with median": "median",
                    "Fill with mode": "mode",
                    "Fill with zero": "zero"
                }
                
                if st.button(f"Fill Missing Values ({missing_method})"):
                    cleaned_data = data_processor.fill_missing_values(method=method_map[missing_method])
                    st.session_state.data = cleaned_data
                    st.success(f"Filled missing values using {missing_method.lower()}")
                    st.rerun()
            
            elif missing_method == "Fill with a specific value":
                fill_value = st.text_input("Enter value to fill missing data with:", "0")
                
                if st.button("Fill Missing Values (Custom)"):
                    # Convert to appropriate type based on column
                    try:
                        cleaned_data = st.session_state.data.copy()
                        for col in missing_cols:
                            if cleaned_data[col].dtype.kind in 'ifc':  # For numeric columns
                                cleaned_data[col].fillna(float(fill_value), inplace=True)
                            else:  # For string/object columns
                                cleaned_data[col].fillna(str(fill_value), inplace=True)
                                
                        st.session_state.data = cleaned_data
                        st.success(f"Filled missing values with '{fill_value}'")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error filling values: {e}")
        
        # Show outlier detection options
        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            st.markdown("#### Detect Outliers")
            
            outlier_col = st.selectbox(
                "Select column to check for outliers:",
                options=numeric_cols
            )
            
            outlier_method = st.selectbox(
                "Select outlier detection method:",
                options=["Z-score", "IQR (Interquartile Range)", "Isolation Forest"]
            )
            
            method_map = {
                "Z-score": "zscore",
                "IQR (Interquartile Range)": "iqr",
                "Isolation Forest": "isolation_forest"
            }
            
            if st.button("Detect Outliers"):
                with st.spinner("Detecting outliers..."):
                    outliers = data_processor.detect_outliers(
                        column=outlier_col,
                        method=method_map[outlier_method]
                    )
                    
                    num_outliers = outliers.sum()
                    outlier_percentage = (num_outliers / len(outliers)) * 100
                    
                    st.info(f"Detected {num_outliers} outliers ({outlier_percentage:.2f}%) in column '{outlier_col}' using {outlier_method}")
                    
                    # Show a sample of the outliers
                    if num_outliers > 0:
                        st.markdown("##### Sample of detected outliers:")
                        st.dataframe(st.session_state.data[outliers].sample(min(5, num_outliers)), use_container_width=True)
                        
                        # Option to remove outliers
                        if st.button(f"Remove {num_outliers} outliers from '{outlier_col}'"):
                            cleaned_data = st.session_state.data[~outliers].reset_index(drop=True)
                            st.session_state.data = cleaned_data
                            st.success(f"Removed {num_outliers} outliers. New shape: {cleaned_data.shape}")
                            st.rerun()
        
        # Data type conversion
        st.markdown("#### Data Type Conversion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            convert_col = st.selectbox(
                "Select column to convert:",
                options=st.session_state.data.columns.tolist()
            )
        
        with col2:
            target_type = st.selectbox(
                "Convert to:",
                options=["string", "numeric", "datetime", "categorical"]
            )
        
        if st.button("Convert Data Type"):
            try:
                converted_data = st.session_state.data.copy()
                
                if target_type == "string":
                    converted_data[convert_col] = converted_data[convert_col].astype(str)
                elif target_type == "numeric":
                    converted_data[convert_col] = pd.to_numeric(converted_data[convert_col], errors='coerce')
                elif target_type == "datetime":
                    converted_data[convert_col] = pd.to_datetime(converted_data[convert_col], errors='coerce')
                elif target_type == "categorical":
                    converted_data[convert_col] = converted_data[convert_col].astype('category')
                
                st.session_state.data = converted_data
                st.success(f"Converted '{convert_col}' to {target_type}")
                st.rerun()
            except Exception as e:
                st.error(f"Error converting data type: {e}")
        
        # Save processed data
        if st.session_state.db_connected:
            st.markdown("### Save Dataset")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_name = st.text_input("Dataset Name", f"Processed {st.session_state.file_name}")
                dataset_description = st.text_area("Description", "Dataset processed with Easy AI Analytics")
                
                if st.button("Save to Database"):
                    try:
                        dataset_id = db.save_dataset(
                            name=dataset_name,
                            description=dataset_description,
                            df=st.session_state.data,
                            file_name=st.session_state.file_name
                        )
                        
                        if dataset_id:
                            st.success(f"Dataset saved successfully with ID: {dataset_id}")
                        else:
                            st.error("Failed to save dataset")
                    except Exception as e:
                        st.error(f"Error saving dataset: {e}")
            
            with col2:
                st.markdown("#### Download Processed Data")
                # CSV Download
                csv_data = st.session_state.data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name=f"processed_{st.session_state.file_name.split('.')[0]}.csv",
                    mime="text/csv"
                )
                
                # Excel Download
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    st.session_state.data.to_excel(writer, sheet_name='Data', index=False)
                    
                st.download_button(
                    label="Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"processed_{st.session_state.file_name.split('.')[0]}.xlsx",
                    mime="application/vnd.ms-excel"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Visualize
with tab2:
    if st.session_state.data is not None:
        # Initialize Visualizer
        visualizer = Visualizer(st.session_state.data)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Data Visualization")
        st.markdown("Select visualization type and configure chart parameters")
        
        # Create sections for different visualization types
        viz_type = st.selectbox(
            "Select Visualization Type",
            options=[
                "Missing Values Heatmap",
                "Distribution Analysis",
                "Correlation Matrix",
                "Scatter Plot",
                "Line Chart",
                "Bar Chart",
                "Pie Chart",
                "Box Plot",
                "Histogram"
            ]
        )
        
        if viz_type == "Missing Values Heatmap":
            st.markdown("#### Missing Values Heatmap")
            st.markdown("Visualize the pattern of missing values in your dataset")
            
            fig = visualizer.plot_missing_values()
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Distribution Analysis":
            st.markdown("#### Distribution Analysis")
            
            # Select column type
            column_type = st.radio(
                "Select column type",
                options=["Numeric", "Categorical"],
                horizontal=True
            )
            
            if column_type == "Numeric":
                numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    selected_col = st.selectbox(
                        "Select numeric column",
                        options=numeric_cols
                    )
                    
                    fig = visualizer.plot_numeric_distribution(selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    stats = st.session_state.data[selected_col].describe()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{stats['50%']:.2f}")
                    with col3:
                        st.metric("Min", f"{stats['min']:.2f}")
                    with col4:
                        st.metric("Max", f"{stats['max']:.2f}")
                else:
                    st.warning("No numeric columns found in the dataset")
            
            else:  # Categorical
                categorical_cols = st.session_state.data.select_dtypes(exclude=['number']).columns.tolist()
                
                if categorical_cols:
                    selected_col = st.selectbox(
                        "Select categorical column",
                        options=categorical_cols
                    )
                    
                    fig = visualizer.plot_categorical_distribution(selected_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show value counts
                    value_counts = st.session_state.data[selected_col].value_counts()
                    
                    st.markdown("#### Value Counts")
                    st.dataframe(value_counts.reset_index().rename(columns={"index": selected_col, selected_col: "Count"}))
                else:
                    st.warning("No categorical columns found in the dataset")
        
        elif viz_type == "Correlation Matrix":
            st.markdown("#### Correlation Matrix")
            st.markdown("Analyze the relationships between numeric variables")
            
            # Get numeric columns
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Option to select specific columns
                use_all_numeric = st.checkbox("Use all numeric columns", value=True)
                
                if not use_all_numeric:
                    selected_cols = st.multiselect(
                        "Select columns for correlation matrix",
                        options=numeric_cols,
                        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                    )
                    
                    if len(selected_cols) < 2:
                        st.warning("Please select at least 2 columns")
                    else:
                        fig = visualizer.plot_correlation_matrix()
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = visualizer.plot_correlation_matrix()
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Correlation matrix requires at least 2 numeric columns")
        
        elif viz_type == "Scatter Plot":
            st.markdown("#### Scatter Plot")
            st.markdown("Visualize the relationship between two numeric variables")
            
            # Get numeric columns
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            all_cols = st.session_state.data.columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox(
                        "Select X-axis column",
                        options=numeric_cols,
                        index=0
                    )
                
                with col2:
                    y_col = st.selectbox(
                        "Select Y-axis column",
                        options=numeric_cols,
                        index=min(1, len(numeric_cols) - 1)
                    )
                
                # Optional color column
                use_color = st.checkbox("Add color dimension", value=False)
                color_col = None
                
                if use_color:
                    color_col = st.selectbox(
                        "Select column for color",
                        options=all_cols
                    )
                
                fig = visualizer.plot_scatter(x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add trendline option
                if st.checkbox("Add trendline", value=False):
                    fig = px.scatter(
                        st.session_state.data,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and show correlation
                    correlation = st.session_state.data[[x_col, y_col]].corr().iloc[0, 1]
                    st.metric("Correlation Coefficient", f"{correlation:.4f}")
            else:
                st.warning("Scatter plot requires at least 2 numeric columns")
        
        elif viz_type == "Line Chart":
            st.markdown("#### Line Chart")
            st.markdown("Visualize trends over a sequence (usually time)")
            
            # Check if there's a datetime column
            datetime_cols = []
            for col in st.session_state.data.columns:
                try:
                    if pd.to_datetime(st.session_state.data[col], errors='coerce').notna().all():
                        datetime_cols.append(col)
                except:
                    pass
            
            # Get numeric columns
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            all_cols = st.session_state.data.columns.tolist()
            
            if datetime_cols:
                st.success(f"Found {len(datetime_cols)} date/time columns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox(
                        "Select X-axis (date) column",
                        options=datetime_cols,
                        index=0
                    )
                
                with col2:
                    if numeric_cols:
                        y_col = st.selectbox(
                            "Select Y-axis (value) column",
                            options=numeric_cols,
                            index=0
                        )
                        
                        # Optional group by
                        use_groups = st.checkbox("Group by category", value=False)
                        group_col = None
                        
                        if use_groups:
                            categorical_cols = [col for col in all_cols if col not in numeric_cols or col != x_col]
                            if categorical_cols:
                                group_col = st.selectbox(
                                    "Select column for grouping",
                                    options=categorical_cols
                                )
                            else:
                                st.warning("No categorical columns available for grouping")
                        
                        try:
                            # Convert to datetime
                            x_values = pd.to_datetime(st.session_state.data[x_col])
                            
                            # Create line chart
                            if use_groups and group_col:
                                fig = px.line(
                                    st.session_state.data,
                                    x=x_col,
                                    y=y_col,
                                    color=group_col,
                                    title=f"{y_col} by {x_col}, grouped by {group_col}"
                                )
                            else:
                                fig = px.line(
                                    st.session_state.data,
                                    x=x_col,
                                    y=y_col,
                                    title=f"{y_col} by {x_col}"
                                )
                            
                            # Add markers if requested
                            show_markers = st.checkbox("Show markers", value=False)
                            if show_markers:
                                fig.update_traces(mode="lines+markers")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating line chart: {e}")
                    else:
                        st.warning("No numeric columns found for Y-axis")
            else:
                st.warning("No date/time columns detected. Please convert a column to datetime format first.")
        
        elif viz_type == "Bar Chart":
            st.markdown("#### Bar Chart")
            st.markdown("Visualize and compare values across categories")
            
            all_cols = st.session_state.data.columns.tolist()
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            
            # Select columns for x and y
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "Select X-axis (category) column",
                    options=all_cols,
                    index=0
                )
            
            with col2:
                if numeric_cols:
                    y_col = st.selectbox(
                        "Select Y-axis (value) column",
                        options=numeric_cols,
                        index=0
                    )
                    
                    # Bar chart options
                    chart_options = st.multiselect(
                        "Chart options",
                        options=["Group by another category", "Sort bars", "Show as horizontal bars"],
                        default=[]
                    )
                    
                    group_col = None
                    if "Group by another category" in chart_options:
                        other_cols = [col for col in all_cols if col != x_col and col != y_col]
                        if other_cols:
                            group_col = st.selectbox(
                                "Select column for grouping",
                                options=other_cols
                            )
                    
                    # Create the chart
                    try:
                        # Basic parameters
                        params = {
                            "data_frame": st.session_state.data,
                            "x": x_col,
                            "y": y_col,
                            "title": f"{y_col} by {x_col}"
                        }
                        
                        # Add grouping if selected
                        if "Group by another category" in chart_options and group_col:
                            params["color"] = group_col
                            params["barmode"] = "group"
                            params["title"] = f"{y_col} by {x_col}, grouped by {group_col}"
                        
                        # Choose orientation
                        if "Show as horizontal bars" in chart_options:
                            # For horizontal bars, we swap x and y
                            params["x"] = y_col
                            params["y"] = x_col
                            params["orientation"] = "h"
                        
                        # Create chart
                        fig = px.bar(**params)
                        
                        # Apply sorting if requested
                        if "Sort bars" in chart_options:
                            sort_order = st.radio(
                                "Sort order",
                                options=["Ascending", "Descending"],
                                horizontal=True
                            )
                            
                            if "Show as horizontal bars" in chart_options:
                                # For horizontal bars, we sort by x since x is the value
                                fig = px.bar(
                                    **params,
                                    category_orders={
                                        x_col: st.session_state.data.groupby(x_col)[y_col].mean().sort_values(
                                            ascending=(sort_order == "Ascending")
                                        ).index
                                    }
                                )
                            else:
                                # For vertical bars
                                fig = px.bar(
                                    **params,
                                    category_orders={
                                        x_col: st.session_state.data.groupby(x_col)[y_col].mean().sort_values(
                                            ascending=(sort_order == "Ascending")
                                        ).index
                                    }
                                )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating bar chart: {e}")
                else:
                    st.warning("No numeric columns found for Y-axis")
        
        elif viz_type == "Pie Chart":
            st.markdown("#### Pie Chart")
            st.markdown("Visualize the composition of a whole into its parts")
            
            all_cols = st.session_state.data.columns.tolist()
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            
            # Select columns for names and values
            col1, col2 = st.columns(2)
            
            with col1:
                names_col = st.selectbox(
                    "Select category column (names)",
                    options=all_cols,
                    index=0
                )
            
            with col2:
                if numeric_cols:
                    values_col = st.selectbox(
                        "Select values column",
                        options=numeric_cols,
                        index=0
                    )
                    
                    # Pie chart options
                    donut = st.checkbox("Show as donut chart", value=False)
                    show_pct = st.checkbox("Show percentages", value=True)
                    
                    # Check if we should aggregate
                    has_values = len(st.session_state.data[names_col].unique()) < len(st.session_state.data)
                    if has_values:
                        st.info(f"Found multiple values per category. Will aggregate {value_col} by sum.")
                        aggregated_data = st.session_state.data.groupby(names_col)[values_col].sum().reset_index()
                    else:
                        aggregated_data = st.session_state.data
                    
                    try:
                        # Create pie chart
                        fig = px.pie(
                            aggregated_data,
                            names=names_col,
                            values=values_col,
                            title=f"Distribution of {values_col} by {names_col}"
                        )
                        
                        # Apply donut if requested
                        if donut:
                            fig.update_traces(hole=0.4)
                        
                        # Configure text display
                        if show_pct:
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                        else:
                            fig.update_traces(textposition='inside', textinfo='label')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating pie chart: {e}")
                else:
                    st.warning("No numeric columns found for values")
        
        elif viz_type == "Box Plot":
            st.markdown("#### Box Plot")
            st.markdown("Visualize the distribution of numeric data across categories")
            
            all_cols = st.session_state.data.columns.tolist()
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                # Select columns for x and y
                col1, col2 = st.columns(2)
                
                with col1:
                    y_col = st.selectbox(
                        "Select numeric column (Y-axis)",
                        options=numeric_cols,
                        index=0
                    )
                
                with col2:
                    use_category = st.checkbox("Group by category", value=True)
                    x_col = None
                    
                    if use_category:
                        other_cols = [col for col in all_cols if col != y_col]
                        if other_cols:
                            x_col = st.selectbox(
                                "Select category column (X-axis)",
                                options=other_cols,
                                index=0
                            )
                    
                    # Create box plot
                    try:
                        if use_category and x_col:
                            fig = px.box(
                                st.session_state.data,
                                x=x_col,
                                y=y_col,
                                title=f"Distribution of {y_col} by {x_col}"
                            )
                        else:
                            fig = px.box(
                                st.session_state.data,
                                y=y_col,
                                title=f"Distribution of {y_col}"
                            )
                        
                        # Option to show individual points
                        show_points = st.checkbox("Show individual data points", value=False)
                        if show_points:
                            fig.update_traces(boxpoints="all", jitter=0.3)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics
                        st.markdown("#### Summary Statistics")
                        
                        if use_category and x_col:
                            stats = st.session_state.data.groupby(x_col)[y_col].describe().reset_index()
                        else:
                            stats = st.session_state.data[y_col].describe().to_frame().T
                            
                        st.dataframe(stats, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating box plot: {e}")
            else:
                st.warning("No numeric columns found for box plot")
        
        elif viz_type == "Histogram":
            st.markdown("#### Histogram")
            st.markdown("Visualize the distribution of a numeric variable")
            
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            all_cols = st.session_state.data.columns.tolist()
            
            if numeric_cols:
                # Select column
                col = st.selectbox(
                    "Select numeric column",
                    options=numeric_cols,
                    index=0
                )
                
                # Histogram options
                col1, col2 = st.columns(2)
                
                with col1:
                    bin_size = st.slider(
                        "Number of bins",
                        min_value=5,
                        max_value=100,
                        value=20
                    )
                
                with col2:
                    use_color = st.checkbox("Color by category", value=False)
                    color_col = None
                    
                    if use_color:
                        categorical_cols = [c for c in all_cols if c != col and st.session_state.data[c].nunique() < 20]
                        if categorical_cols:
                            color_col = st.selectbox(
                                "Select category column for color",
                                options=categorical_cols
                            )
                
                # Create histogram
                try:
                    if use_color and color_col:
                        fig = px.histogram(
                            st.session_state.data,
                            x=col,
                            color=color_col,
                            nbins=bin_size,
                            title=f"Distribution of {col} by {color_col}",
                            marginal="box"
                        )
                    else:
                        fig = px.histogram(
                            st.session_state.data,
                            x=col,
                            nbins=bin_size,
                            title=f"Distribution of {col}",
                            marginal="box"
                        )
                    
                    # Show density curve option
                    show_density = st.checkbox("Show density curve", value=False)
                    if show_density and not use_color:
                        fig = px.histogram(
                            st.session_state.data,
                            x=col,
                            nbins=bin_size,
                            title=f"Distribution of {col}",
                            marginal="box",
                            histnorm="probability density"
                        )
                        
                        # Add KDE curve
                        from scipy import stats
                        import numpy as np
                        
                        kde = stats.gaussian_kde(st.session_state.data[col].dropna())
                        x_range = np.linspace(
                            st.session_state.data[col].min(),
                            st.session_state.data[col].max(),
                            1000
                        )
                        y_kde = kde(x_range)
                        
                        fig.add_scatter(
                            x=x_range,
                            y=y_kde,
                            mode="lines",
                            line=dict(color="red", width=2),
                            name="Density Curve"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.markdown("#### Summary Statistics")
                    
                    stats = st.session_state.data[col].describe()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{stats['50%']:.2f}")
                    with col3:
                        st.metric("Std Dev", f"{stats['std']:.2f}")
                    with col4:
                        # Calculate skewness
                        skewness = st.session_state.data[col].skew()
                        st.metric("Skewness", f"{skewness:.2f}")
                except Exception as e:
                    st.error(f"Error creating histogram: {e}")
            else:
                st.warning("No numeric columns found for histogram")
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload data in the Upload & Explore tab first")

# Tab 3: Generate Report
with tab3:
    if st.session_state.data is not None:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Generate Report")
        
        # Generate insights from the data
        if not st.session_state.insights:
            data_processor = DataProcessor(st.session_state.data)
            try:
                st.session_state.insights = data_processor.generate_insights()
            except Exception as e:
                st.session_state.insights = [
                    "Failed to generate automated insights. Please check the data format.",
                    f"Error: {str(e)}"
                ]
        
        # Show insights
        st.markdown("#### Key Insights")
        for i, insight in enumerate(st.session_state.insights[:5], 1):
            st.markdown(f"{i}. {insight}")
            
        # Report form
        st.markdown("#### Report Configuration")
        
        report_form = st.form(key="report_form")
        
        with report_form:
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Report Title", "Data Analysis Report")
                company_name = st.text_input("Company Name", "Your Company")
            
            with col2:
                report_type = st.selectbox(
                    "Report Type",
                    options=["PDF", "DOCX", "Both"]
                )
                include_viz = st.checkbox("Include Visualizations", value=True)
            
            description = st.text_area(
                "Report Description",
                "This report provides an analysis of the uploaded dataset."
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                include_missing = st.checkbox("Include Missing Values Analysis", value=True)
            
            with col2:
                include_dist = st.checkbox("Include Distribution Analysis", value=True)
                
                if include_dist:
                    # Get numeric columns for distribution analysis
                    numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = st.session_state.data.select_dtypes(exclude=['number']).columns.tolist()
                    
                    if numeric_cols:
                        numeric_col = st.selectbox(
                            "Select column for numeric distribution",
                            options=numeric_cols
                        )
                    
                    if categorical_cols:
                        categorical_col = st.selectbox(
                            "Select column for categorical distribution",
                            options=categorical_cols
                        )
            
            with col3:
                include_corr = st.checkbox("Include Correlation Analysis", value=True)
            
            generate_pressed = st.form_submit_button("Generate Report")
            
        if generate_pressed:
            # Create visualization figures if requested
            missing_fig = None
            numeric_fig = None
            categorical_fig = None
            
            with st.spinner("Generating visualizations..."):
                visualizer = Visualizer(st.session_state.data)
                
                if include_viz and include_missing:
                    missing_fig = visualizer.plot_missing_values()
                
                if include_viz and include_dist:
                    if 'numeric_col' in locals() and numeric_col:
                        numeric_fig = visualizer.plot_numeric_distribution(numeric_col)
                    
                    if 'categorical_col' in locals() and categorical_col:
                        categorical_fig = visualizer.plot_categorical_distribution(categorical_col)
            
            # Generate form data dictionary
            form_data = {
                "title": title,
                "company_name": company_name,
                "description": description,
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "include_missing": include_missing,
                "include_dist": include_dist,
                "include_corr": include_corr
            }
            
            with st.spinner("Generating report..."):
                try:
                    if report_type in ["PDF", "Both"]:
                        # Generate PDF
                        pdf_generator = PDFGenerator(
                            form_data=form_data,
                            data=st.session_state.data,
                            insights=st.session_state.insights,
                            missing_fig=missing_fig,
                            numeric_fig=numeric_fig,
                            categorical_fig=categorical_fig
                        )
                        
                        pdf_data = pdf_generator.generate_pdf()
                        
                        # Download button for PDF
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"{title.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    
                    if report_type in ["DOCX", "Both"]:
                        # Generate DOCX
                        docx_generator = DocxGenerator(
                            form_data=form_data,
                            data=st.session_state.data,
                            insights=st.session_state.insights,
                            missing_fig=missing_fig,
                            numeric_fig=numeric_fig,
                            categorical_fig=categorical_fig
                        )
                        
                        docx_data = docx_generator.generate_docx()
                        
                        # Download button for DOCX
                        st.download_button(
                            label="Download DOCX Report",
                            data=docx_data,
                            file_name=f"{title.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    
                    # Save report to database if connected
                    if st.session_state.db_connected and st.checkbox("Save Report to Database", value=False):
                        try:
                            pdf_bytes = pdf_data if report_type in ["PDF", "Both"] else None
                            docx_bytes = docx_data if report_type in ["DOCX", "Both"] else None
                            
                            report_id = db.save_report(
                                title=title,
                                company_name=company_name,
                                description=description,
                                pdf_data=pdf_bytes,
                                docx_data=docx_bytes,
                                report_format=report_type.lower()
                            )
                            
                            if report_id:
                                st.success(f"Report saved to database with ID: {report_id}")
                            else:
                                st.error("Failed to save report to database")
                        except Exception as e:
                            st.error(f"Error saving report to database: {e}")
                    
                    st.success("Report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    st.exception(e)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload data in the Upload & Explore tab first")

# Tab 4: Saved Data
with tab4:
    if st.session_state.db_connected:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Saved Datasets")
        
        # Refresh data
        if st.button("Refresh Datasets"):
            st.rerun()
        
        # Get datasets from database
        datasets = db.get_all_datasets()
        
        if datasets:
            # Display datasets in a table
            dataset_data = []
            for dataset_id, name, description, file_name, created_at in datasets:
                dataset_data.append({
                    "ID": dataset_id,
                    "Name": name,
                    "Description": description,
                    "Filename": file_name,
                    "Created At": created_at
                })
            
            dataset_df = pd.DataFrame(dataset_data)
            st.dataframe(dataset_df, use_container_width=True)
            
            # Select dataset to load
            selected_dataset_id = st.selectbox(
                "Select a dataset to load",
                options=dataset_df["ID"].tolist(),
                format_func=lambda x: f"{dataset_df[dataset_df['ID'] == x]['Name'].iloc[0]} (ID: {x})"
            )
            
            if st.button("Load Selected Dataset"):
                loaded_df = db.get_dataset(selected_dataset_id)
                if loaded_df is not None:
                    st.session_state.data = loaded_df
                    st.session_state.file_name = dataset_df[dataset_df["ID"] == selected_dataset_id]["Filename"].iloc[0]
                    st.success(f"Dataset loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load dataset from database.")
        else:
            st.info("No datasets found in the database.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Saved Reports")
        
        # Refresh reports
        if st.button("Refresh Reports"):
            st.rerun()
        
        # Get reports from database
        reports = db.get_all_reports()
        
        if reports:
            # Display reports in a table
            report_data = []
            for report_id, title, company_name, description, created_at in reports:
                report_data.append({
                    "ID": report_id,
                    "Title": title,
                    "Company": company_name,
                    "Description": description,
                    "Created At": created_at
                })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)
            
            # Select report to view
            col1, col2 = st.columns(2)
            
            with col1:
                selected_report_id = st.selectbox(
                    "Select a report to view",
                    options=report_df["ID"].tolist(),
                    format_func=lambda x: f"{report_df[report_df['ID'] == x]['Title'].iloc[0]} (ID: {x})"
                )
            
            with col2:
                report_format = st.radio(
                    "Select format",
                    options=["PDF", "DOCX"],
                    horizontal=True
                )
            
            if st.button("Download Selected Report"):
                try:
                    report_data = db.get_report(selected_report_id, format=report_format.lower())
                    if report_data:
                        title = report_df[report_df["ID"] == selected_report_id]["Title"].iloc[0]
                        
                        if report_format == "PDF":
                            file_ext = "pdf"
                            mime_type = "application/pdf"
                        else:
                            file_ext = "docx"
                            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        
                        # Download button
                        st.download_button(
                            label=f"Download {report_format} Report",
                            data=report_data,
                            file_name=f"{title.replace(' ', '_')}.{file_ext}",
                            mime=mime_type
                        )
                    else:
                        st.error(f"Report not available in {report_format} format.")
                except Exception as e:
                    st.error(f"Error downloading report: {e}")
        else:
            st.info("No reports found in the database.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Database connection is required to view saved data.")