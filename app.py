import streamlit as st
import pandas as pd
import numpy as np
import io
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

# Sidebar for file upload and operations
with st.sidebar:
    st.subheader("1. Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            # Store in session state
            st.session_state.data = df
            st.session_state.cleaned_data = df.copy()
            st.success(f"Successfully loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    
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
        
        # Missing values visualization
        st.subheader("Missing Values by Column")
        missing_chart = visualizer.plot_missing_values()
        st.plotly_chart(missing_chart, use_container_width=True)
        
        # Column distribution for a selected column
        st.subheader("Column Distribution")
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
        
        # Correlation matrix for numeric columns
        if len(numeric_columns) > 1:
            st.subheader("Correlation Matrix")
            corr_chart = visualizer.plot_correlation_matrix()
            st.plotly_chart(corr_chart, use_container_width=True)
    
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
    """)

# If no data is uploaded, display a welcome message
else:
    st.info("👆 Upload a CSV file to get started with your data analysis!")
    
    # Display some example features
    st.header("Features Available After Upload:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Analysis")
        st.markdown("""
        - Data overview and statistics
        - Missing value detection
        - Correlation analysis
        - Data cleaning tools
        """)
    
    with col2:
        st.subheader("Visualizations & Reporting")
        st.markdown("""
        - Column distribution charts
        - Missing values visualization
        - Automated insights generation
        - Professional PDF report generation
        """)
