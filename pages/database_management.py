import streamlit as st
import pandas as pd
import io
import datetime
import database as db
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer
from utils.pdf_generator import PDFGenerator
from utils.docx_generator import DocxGenerator

# Page configuration
st.set_page_config(
    page_title="Database Management",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Database Management")

# Check database connection
db_status = db.check_database_connection()

if not db_status:
    st.error("⚠️ Database connection failed. Please check your database configuration.")
    st.stop()

# Create tabs for different database operations
tab1, tab2, tab3, tab4 = st.tabs([
    "Save/Load Datasets", 
    "Reports", 
    "Insights", 
    "Settings"
])

# Initialize session state variables if they don't exist
if 'db_datasets' not in st.session_state:
    st.session_state.db_datasets = []
if 'db_reports' not in st.session_state:
    st.session_state.db_reports = []
if 'db_insights' not in st.session_state:
    st.session_state.db_insights = []

# Helper function to refresh dataset list
def refresh_datasets():
    st.session_state.db_datasets = db.get_all_datasets()

# Helper function to refresh report list
def refresh_reports():
    st.session_state.db_reports = db.get_all_reports()

# Helper function to refresh insights
def refresh_insights():
    st.session_state.db_insights = db.get_insights()

# Tab 1: Save/Load Datasets
with tab1:
    st.header("Saved Datasets")
    
    # Refresh button
    if st.button("Refresh Datasets"):
        refresh_datasets()
    
    # Get datasets if not already loaded
    if not st.session_state.db_datasets:
        refresh_datasets()
    
    # Show datasets in a dataframe
    if st.session_state.db_datasets:
        # Convert to DataFrame for better display
        datasets_df = pd.DataFrame(
            st.session_state.db_datasets,
            columns=["ID", "Name", "Description", "File Name", "Created At"]
        )
        st.dataframe(datasets_df)
        
        # Load selected dataset
        selected_dataset_id = st.selectbox(
            "Select a dataset to load", 
            options=[d[0] for d in st.session_state.db_datasets],
            format_func=lambda x: next((d[1] for d in st.session_state.db_datasets if d[0] == x), "")
        )
        
        if st.button("Load Selected Dataset"):
            with st.spinner("Loading dataset..."):
                loaded_df = db.get_dataset(selected_dataset_id)
                if loaded_df is not None:
                    st.session_state.data = loaded_df
                    st.session_state.cleaned_data = loaded_df.copy()
                    st.success(f"Dataset loaded with {loaded_df.shape[0]} rows and {loaded_df.shape[1]} columns")
                    # Redirect to main page
                    st.info("Dataset loaded! Go to the main page to analyze it.")
                else:
                    st.error("Failed to load the dataset")
    else:
        st.info("No datasets saved in the database. Save a dataset first.")
    
    # Save current dataset section
    st.header("Save Current Dataset")
    
    if st.session_state.data is not None:
        dataset_name = st.text_input("Dataset Name")
        dataset_desc = st.text_area("Dataset Description")
        
        if st.button("Save Current Dataset to Database"):
            if dataset_name:
                with st.spinner("Saving dataset..."):
                    # Default filename if it was from sample data
                    filename = "sample_data.csv"
                    if not st.session_state.get('sample_data_loaded', False):
                        filename = "uploaded_data.csv"
                    
                    dataset_id = db.save_dataset(
                        name=dataset_name,
                        description=dataset_desc,
                        df=st.session_state.cleaned_data,
                        file_name=filename
                    )
                    
                    st.success(f"Dataset saved successfully with ID: {dataset_id}")
                    refresh_datasets()
            else:
                st.warning("Please provide a name for the dataset")
    else:
        st.info("No dataset loaded. Please upload or load a dataset first.")

# Tab 2: Reports Management
with tab2:
    st.header("Saved Reports")
    
    # Refresh button
    if st.button("Refresh Reports"):
        refresh_reports()
    
    # Get reports if not already loaded
    if not st.session_state.db_reports:
        refresh_reports()
    
    # Show reports in a dataframe
    if st.session_state.db_reports:
        # Convert to DataFrame for better display
        reports_df = pd.DataFrame(
            st.session_state.db_reports,
            columns=["ID", "Title", "Company", "Description", "Created At"]
        )
        st.dataframe(reports_df)
        
        # Download selected report
        selected_report_id = st.selectbox(
            "Select a report to download", 
            options=[r[0] for r in st.session_state.db_reports],
            format_func=lambda x: next((f"{r[1]} ({r[2]})" for r in st.session_state.db_reports if r[0] == x), "")
        )
        
        if selected_report_id:
            report_info = next((r for r in st.session_state.db_reports if r[0] == selected_report_id), None)
            
            # Create columns for different download formats
            dl_col1, dl_col2 = st.columns(2)
            
            # PDF Download
            with dl_col1:
                pdf_data = db.get_report(selected_report_id, format='pdf')
                if pdf_data:
                    pdf_filename = f"{report_info[2]}_{report_info[1]}.pdf" if report_info else "report.pdf"
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
            
            # DOCX Download
            with dl_col2:
                docx_data = db.get_report(selected_report_id, format='docx')
                if docx_data:
                    docx_filename = f"{report_info[2]}_{report_info[1]}.docx" if report_info else "report.docx"
                    
                    st.download_button(
                        label="Download DOCX Report",
                        data=docx_data,
                        file_name=docx_filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                else:
                    st.info("No DOCX version available for this report")
    else:
        st.info("No reports saved in the database. Generate and save a report first.")
    
    # Save current report section
    st.header("Save Current Report")
    
    if st.session_state.get('form_data', {}).get('company_name') and st.session_state.get('insights', []):
        if st.button("Generate and Save New Report"):
            try:
                with st.spinner("Generating and saving report..."):
                    if st.session_state.cleaned_data is not None:
                        visualizer = Visualizer(st.session_state.cleaned_data)
                        
                        # Generate plots for the reports
                        missing_fig = visualizer.plot_missing_values()
                        
                        numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
                        
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
                        
                        # Generate DOCX with Excel formulas
                        docx_generator = DocxGenerator(
                            st.session_state.form_data,
                            st.session_state.cleaned_data,
                            st.session_state.insights,
                            missing_fig,
                            num_fig,
                            cat_fig
                        )
                        docx_bytes = docx_generator.generate_docx()
                        
                        # Save both formats to database
                        report_id = db.save_report(
                            title=st.session_state.form_data.get('project_title', 'Analytics Report'),
                            company_name=st.session_state.form_data.get('company_name', ''),
                            description=st.session_state.form_data.get('objectives', ''),
                            pdf_data=pdf_bytes,
                            docx_data=docx_bytes,
                            report_format='both'
                        )
                        
                        st.success(f"Report saved successfully with ID: {report_id}")
                        refresh_reports()
                    else:
                        st.error("No data available for report generation")
            except Exception as e:
                st.error(f"Error generating and saving report: {e}")
    else:
        st.info("Please fill out the business requirement form and generate insights before saving a report.")

# Tab 3: Insights Management
with tab3:
    st.header("Saved Insights")
    
    # Refresh button
    if st.button("Refresh Insights"):
        refresh_insights()
    
    # Get insights if not already loaded
    if not st.session_state.db_insights:
        refresh_insights()
    
    # Filter options
    insight_types = ["All Types", "general", "missing_values", "correlation", "distribution", "recommendation"]
    selected_type = st.selectbox("Filter by Insight Type", options=insight_types)
    
    # Filter insights
    filtered_insights = st.session_state.db_insights
    if selected_type != "All Types" and filtered_insights:
        filtered_insights = [i for i in filtered_insights if i.insight_type == selected_type]
    
    # Show insights
    if filtered_insights:
        for i, insight in enumerate(filtered_insights):
            with st.expander(f"Insight {i+1} ({insight.insight_type})"):
                st.write(insight.insight_text)
                st.caption(f"Dataset ID: {insight.dataset_id} | Created: {insight.created_at}")
    else:
        st.info("No insights saved in the database matching the selected filter.")
    
    # Save current insights section
    st.header("Save Current Insights")
    
    if st.session_state.data is not None and st.session_state.get('insights', []):
        
        # Get datasets to associate insights with
        available_datasets = st.session_state.db_datasets
        
        if available_datasets:
            selected_dataset_id = st.selectbox(
                "Select a dataset to associate insights with", 
                options=[d[0] for d in available_datasets],
                format_func=lambda x: next((d[1] for d in available_datasets if d[0] == x), "")
            )
            
            if st.button("Save Current Insights"):
                with st.spinner("Saving insights..."):
                    count = db.save_insights(
                        dataset_id=selected_dataset_id,
                        insights_list=st.session_state.insights
                    )
                    
                    st.success(f"Saved {count} insights to the database")
                    refresh_insights()
        else:
            st.warning("Please save the dataset first before saving insights")
    else:
        st.info("No insights generated. Please upload a dataset and generate insights first.")

# Tab 4: Settings
with tab4:
    st.header("Application Settings")
    
    # Get current settings
    default_chart_type = db.get_or_create_setting("default_chart_type", "bar")
    theme_color = db.get_or_create_setting("theme_color", "#1E88E5")
    max_rows_preview = db.get_or_create_setting("max_rows_preview", "5")
    auto_insights = db.get_or_create_setting("auto_generate_insights", "false")
    
    # Settings form
    with st.form("settings_form"):
        new_chart_type = st.selectbox(
            "Default Chart Type", 
            options=["bar", "line", "scatter", "pie", "box"],
            index=["bar", "line", "scatter", "pie", "box"].index(default_chart_type)
        )
        
        new_theme_color = st.color_picker("Theme Color", theme_color)
        
        new_max_rows = st.number_input(
            "Maximum Rows in Preview", 
            min_value=1, 
            max_value=100, 
            value=int(max_rows_preview)
        )
        
        new_auto_insights = st.checkbox(
            "Auto-generate Insights on Data Load", 
            value=auto_insights.lower() == "true"
        )
        
        save_settings = st.form_submit_button("Save Settings")
        
        if save_settings:
            db.update_setting("default_chart_type", new_chart_type)
            db.update_setting("theme_color", new_theme_color)
            db.update_setting("max_rows_preview", str(new_max_rows))
            db.update_setting("auto_generate_insights", str(new_auto_insights).lower())
            
            st.success("Settings updated successfully!")
    
    # Database metrics
    st.header("Database Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        datasets_count = len(st.session_state.db_datasets)
        st.metric("Saved Datasets", datasets_count)
    
    with col2:
        reports_count = len(st.session_state.db_reports)
        st.metric("Saved Reports", reports_count)
    
    with col3:
        insights_count = len(st.session_state.db_insights)
        st.metric("Saved Insights", insights_count)
    
    # Database connection status
    st.subheader("Connection Status")
    
    if db_status:
        st.success("✅ Connected to database")
    else:
        st.error("❌ Database connection failed")