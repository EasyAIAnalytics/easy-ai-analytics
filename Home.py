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
import plotly.io as pio
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import bcrypt

# Set a global Plotly template and colorway
pio.templates.default = "plotly_white"
CUSTOM_COLORWAY = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

# Load custom CSS for login/signup styling
css_path = os.path.join("templates", "login.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

def show_login_signup():
    st.set_page_config(page_title="Login / Signup", page_icon="🔐", layout="centered", initial_sidebar_state="collapsed")
    hide_sidebar = """
        <style>
        [data-testid="stSidebar"], .css-1d391kg {display: none !important;}
        .main {padding-top: 0rem !important;}
        </style>
    """
    st.markdown(hide_sidebar, unsafe_allow_html=True)

    if "show_signup" not in st.session_state:
        st.session_state["show_signup"] = False
    if "login_error" not in st.session_state:
        st.session_state["login_error"] = ""
    if "signup_error" not in st.session_state:
        st.session_state["signup_error"] = ""
    if "signup_success" not in st.session_state:
        st.session_state["signup_success"] = ""

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # st.image("assets/logo.svg", width=80)  # Removed as requested
        st.markdown("<h2 style='text-align:center;'>Easy AI Analytics</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center;'>Sign in to your account or create a new one</h4>", unsafe_allow_html=True)
        st.markdown("---")

        if not st.session_state["show_signup"]:
            # Login Form
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    user = db.get_user_by_username(username)
                    if not username or not password:
                        st.session_state["login_error"] = "Please enter both username and password."
                    elif user and user.get("password_hash") and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
                        st.session_state.logged_in = True
                        st.session_state.user_id = user["id"]
                        st.session_state.username = user["username"]
                        st.session_state["login_error"] = ""
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.session_state["login_error"] = "Invalid username or password."
                if st.session_state["login_error"]:
                    st.error(st.session_state["login_error"])
            st.markdown("<p style='text-align:center;'>Don't have an account? <a href='#' style='color:#636EFA;' onclick=\"window.parent.postMessage({toggleForm:true}, '*')\">Sign up</a></p>", unsafe_allow_html=True)
            if st.button("Create a new account", key="show_signup_btn"):
                st.session_state["show_signup"] = True
                st.session_state["login_error"] = ""
                st.rerun()
        else:
            # Signup Form
            with st.form("signup_form", clear_on_submit=False):
                username = st.text_input("Username", key="signup_username")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
                password2 = st.text_input("Confirm Password", type="password", key="signup_password2")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    if not username or not password or not email or not password2:
                        st.session_state["signup_error"] = "Please fill in all fields."
                        st.session_state["signup_success"] = ""
                    elif password != password2:
                        st.session_state["signup_error"] = "Passwords do not match."
                        st.session_state["signup_success"] = ""
                    elif db.get_user_by_username(username):
                        st.session_state["signup_error"] = "Username already exists."
                        st.session_state["signup_success"] = ""
                    elif db.get_user_by_email(email):
                        st.session_state["signup_error"] = "Email already registered."
                        st.session_state["signup_success"] = ""
                    else:
                        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                        db.create_user(username, email, hashed)
                        st.session_state["signup_error"] = ""
                        st.session_state["signup_success"] = "Account created! Please log in."
                        st.session_state["show_signup"] = False
                        st.rerun()
                if st.session_state["signup_error"]:
                    st.error(st.session_state["signup_error"])
                if st.session_state["signup_success"]:
                    st.success(st.session_state["signup_success"])
            st.markdown("<p style='text-align:center;'>Already have an account?</p>", unsafe_allow_html=True)
            if st.button("Back to Login", key="show_login_btn"):
                st.session_state["show_signup"] = False
                st.session_state["signup_error"] = ""
                st.session_state["signup_success"] = ""
                st.rerun()

def main():
    # Show login page if not logged in
    if not st.session_state.get("logged_in", False):
        show_login_signup()
        st.stop()

    # Add logout button to the sidebar after login
    with st.sidebar:
        if st.button("Logout", key="logout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Page configuration for main app
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

    # Main layout with tabs
    tab1, tab2, tab3 = st.tabs([
        "📤 Upload & Explore",
        "📊 Visualize",
        "📝 Generate Report"
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
                        st.session_state.cleaned_data = cleaned_data  # Keep cleaned_data in sync
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
                            st.session_state.cleaned_data = cleaned_data  # Keep cleaned_data in sync
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
                                st.session_state.cleaned_data = cleaned_data  # Keep cleaned_data in sync
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
                    st.session_state.cleaned_data = converted_data  # Keep cleaned_data in sync
                    st.success(f"Converted '{convert_col}' to {target_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error converting data type: {e}")
            
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
                img_bytes = fig.to_image(format="png")
                st.download_button("Download as PNG", data=img_bytes, file_name="missing_values_heatmap.png", mime="image/png")
                img_bytes_jpg = fig.to_image(format="jpg")
                st.download_button("Download as JPG", data=img_bytes_jpg, file_name="missing_values_heatmap.jpg", mime="image/jpeg")
            
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
                        img_bytes = fig.to_image(format="png")
                        st.download_button("Download as PNG", data=img_bytes, file_name="numeric_distribution.png", mime="image/png")
                        img_bytes_jpg = fig.to_image(format="jpg")
                        st.download_button("Download as JPG", data=img_bytes_jpg, file_name="numeric_distribution.jpg", mime="image/jpeg")
                        
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
                        img_bytes = fig.to_image(format="png")
                        st.download_button("Download as PNG", data=img_bytes, file_name="categorical_distribution.png", mime="image/png")
                        img_bytes_jpg = fig.to_image(format="jpg")
                        st.download_button("Download as JPG", data=img_bytes_jpg, file_name="categorical_distribution.jpg", mime="image/jpeg")
                        
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
                            img_bytes = fig.to_image(format="png")
                            st.download_button("Download as PNG", data=img_bytes, file_name="correlation_matrix.png", mime="image/png")
                            img_bytes_jpg = fig.to_image(format="jpg")
                            st.download_button("Download as JPG", data=img_bytes_jpg, file_name="correlation_matrix.jpg", mime="image/jpeg")
                    else:
                        fig = visualizer.plot_correlation_matrix()
                        st.plotly_chart(fig, use_container_width=True)
                        img_bytes = fig.to_image(format="png")
                        st.download_button("Download as PNG", data=img_bytes, file_name="correlation_matrix.png", mime="image/png")
                        img_bytes_jpg = fig.to_image(format="jpg")
                        st.download_button("Download as JPG", data=img_bytes_jpg, file_name="correlation_matrix.jpg", mime="image/jpeg")
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
                    img_bytes = fig.to_image(format="png")
                    st.download_button("Download as PNG", data=img_bytes, file_name="scatter_plot.png", mime="image/png")
                    img_bytes_jpg = fig.to_image(format="jpg")
                    st.download_button("Download as JPG", data=img_bytes_jpg, file_name="scatter_plot.jpg", mime="image/jpeg")
                    
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
                        img_bytes = fig.to_image(format="png")
                        st.download_button("Download as PNG", data=img_bytes, file_name="trendline.png", mime="image/png")
                        img_bytes_jpg = fig.to_image(format="jpg")
                        st.download_button("Download as JPG", data=img_bytes_jpg, file_name="trendline.jpg", mime="image/jpeg")
                        
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
                                img_bytes = fig.to_image(format="png")
                                st.download_button("Download as PNG", data=img_bytes, file_name="line_chart.png", mime="image/png")
                                img_bytes_jpg = fig.to_image(format="jpg")
                                st.download_button("Download as JPG", data=img_bytes_jpg, file_name="line_chart.jpg", mime="image/jpeg")
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
                            
                            fig.update_layout(font=dict(family="Arial, sans-serif", size=16), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=60, b=40))
                            st.plotly_chart(fig, use_container_width=True)
                            img_bytes = fig.to_image(format="png")
                            st.download_button("Download as PNG", data=img_bytes, file_name="bar_chart.png", mime="image/png")
                            img_bytes_jpg = fig.to_image(format="jpg")
                            st.download_button("Download as JPG", data=img_bytes_jpg, file_name="bar_chart.jpg", mime="image/jpeg")
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
                            st.info(f"Found multiple values per category. Will aggregate {values_col} by sum.")
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
                            
                            fig.update_layout(font=dict(family="Arial, sans-serif", size=16), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=60, b=40))
                            st.plotly_chart(fig, use_container_width=True)
                            img_bytes = fig.to_image(format="png")
                            st.download_button("Download as PNG", data=img_bytes, file_name="pie_chart.png", mime="image/png")
                            img_bytes_jpg = fig.to_image(format="jpg")
                            st.download_button("Download as JPG", data=img_bytes_jpg, file_name="pie_chart.jpg", mime="image/jpeg")
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
                            
                            fig.update_layout(font=dict(family="Arial, sans-serif", size=16), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=60, b=40))
                            st.plotly_chart(fig, use_container_width=True)
                            img_bytes = fig.to_image(format="png")
                            st.download_button("Download as PNG", data=img_bytes, file_name="box_plot.png", mime="image/png")
                            img_bytes_jpg = fig.to_image(format="jpg")
                            st.download_button("Download as JPG", data=img_bytes_jpg, file_name="box_plot.jpg", mime="image/jpeg")
                            
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
                        
                        fig.update_layout(font=dict(family="Arial, sans-serif", size=16), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=60, b=40))
                        st.plotly_chart(fig, use_container_width=True)
                        img_bytes = fig.to_image(format="png")
                        st.download_button("Download as PNG", data=img_bytes, file_name="histogram.png", mime="image/png")
                        img_bytes_jpg = fig.to_image(format="jpg")
                        st.download_button("Download as JPG", data=img_bytes_jpg, file_name="histogram.jpg", mime="image/jpeg")
                        
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
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Generate Custom Report")
        with st.form("report_form"):
            company_name = st.text_input("Company Name")
            report_title = st.text_input("Report Title")
            notes = st.text_area("Additional Notes (optional)")
            st.markdown("#### Select charts/diagrams to include in the report:")
            # For demo: let user select from available chart types
            chart_options = [
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
            selected_charts = st.multiselect("Charts/Diagrams", chart_options)
            submit_report = st.form_submit_button("Generate Report")
        if submit_report:
            if not company_name or not report_title or not selected_charts:
                st.error("Please fill in the company name, report title, and select at least one chart/diagram.")
            else:
                # Generate the selected charts as images
                visualizer = Visualizer(st.session_state.data)
                chart_images = []
                for chart in selected_charts:
                    if chart == "Missing Values Heatmap":
                        fig = visualizer.plot_missing_values()
                    elif chart == "Distribution Analysis":
                        # Use first numeric column for demo
                        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                        if numeric_cols:
                            fig = visualizer.plot_numeric_distribution(numeric_cols[0])
                    elif chart == "Correlation Matrix":
                        fig = visualizer.plot_correlation_matrix()
                    elif chart == "Scatter Plot":
                        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                        if len(numeric_cols) >= 2:
                            fig = visualizer.plot_scatter(numeric_cols[0], numeric_cols[1])
                    elif chart == "Line Chart":
                        # Use first datetime and numeric column for demo
                        datetime_cols = [col for col in st.session_state.data.columns if pd.api.types.is_datetime64_any_dtype(st.session_state.data[col])]
                        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                        if datetime_cols and numeric_cols:
                            fig = px.line(st.session_state.data, x=datetime_cols[0], y=numeric_cols[0])
                    elif chart == "Bar Chart":
                        all_cols = st.session_state.data.columns.tolist()
                        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                        if all_cols and numeric_cols:
                            fig = px.bar(st.session_state.data, x=all_cols[0], y=numeric_cols[0])
                    elif chart == "Pie Chart":
                        all_cols = st.session_state.data.columns.tolist()
                        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                        if all_cols and numeric_cols:
                            fig = px.pie(st.session_state.data, names=all_cols[0], values=numeric_cols[0])
                    elif chart == "Box Plot":
                        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                        if numeric_cols:
                            fig = px.box(st.session_state.data, y=numeric_cols[0])
                    elif chart == "Histogram":
                        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                        if numeric_cols:
                            fig = px.histogram(st.session_state.data, x=numeric_cols[0])
                    else:
                        continue
                    fig.update_layout(font=dict(family="Arial, sans-serif", size=16), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=40, r=40, t=60, b=40))
                    img_bytes = fig.to_image(format="png")
                    chart_images.append((chart, img_bytes))
                # Generate PDF report
                form_data = {
                    "company_name": company_name,
                    "project_title": report_title,
                    "objectives": notes,  # or add more fields as needed
                    "target_audience": "",
                    "timeframe": "",
                    "key_metrics": ""
                }
                pdf_gen = PDFGenerator(form_data, st.session_state.data, st.session_state.insights, chart_images=chart_images)
                pdf_bytes = pdf_gen.generate_pdf()
                st.success("Report generated!")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{report_title.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()