import os
import io
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Text, TIMESTAMP, LargeBinary, ForeignKey, text, JSON, select, insert, update, delete
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import time

# Get database connection from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define tables
saved_datasets = Table(
    'saved_datasets', 
    metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(255), nullable=False),
    Column('description', Text),
    Column('file_data', LargeBinary, nullable=False),
    Column('file_name', String(255), nullable=False),
    Column('created_at', TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)

saved_reports = Table(
    'saved_reports', 
    metadata,
    Column('id', Integer, primary_key=True),
    Column('title', String(255), nullable=False),
    Column('company_name', String(255)),
    Column('description', Text),
    Column('pdf_data', LargeBinary, nullable=False),
    Column('docx_data', LargeBinary),  # DOCX report data (optional)
    Column('report_format', String(10), default='pdf'),  # Format of the report: 'pdf' or 'docx'
    Column('created_at', TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)

analysis_insights = Table(
    'analysis_insights', 
    metadata,
    Column('id', Integer, primary_key=True),
    Column('dataset_id', Integer, ForeignKey('saved_datasets.id', ondelete='CASCADE')),
    Column('insight_text', Text, nullable=False),
    Column('insight_type', String(50)),
    Column('created_at', TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)

user_settings = Table(
    'user_settings', 
    metadata,
    Column('id', Integer, primary_key=True),
    Column('setting_name', String(100), nullable=False, unique=True),
    Column('setting_value', Text),
    Column('updated_at', TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)

processing_metrics = Table(
    'processing_metrics', 
    metadata,
    Column('id', Integer, primary_key=True),
    Column('dataset_id', Integer, ForeignKey('saved_datasets.id', ondelete='CASCADE')),
    Column('process_type', String(50), nullable=False),
    Column('process_details', JSONB),
    Column('execution_time', Float),
    Column('created_at', TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)

# Database function for saving dataset
def save_dataset(name, description, df, file_name):
    """
    Save dataset to database
    
    Args:
        name (str): Name of the dataset
        description (str): Description of the dataset
        df (pd.DataFrame): Dataframe to save
        file_name (str): Original file name
        
    Returns:
        int: ID of the saved dataset
    """
    # Convert dataframe to CSV bytes
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    # Insert into database
    with engine.connect() as connection:
        stmt = insert(saved_datasets).values(
            name=name,
            description=description,
            file_data=csv_data,
            file_name=file_name
        ).returning(saved_datasets.c.id)
        
        result = connection.execute(stmt)
        connection.commit()
        return result.scalar_one()

def get_all_datasets():
    """
    Get all saved datasets
    
    Returns:
        list: List of dataset records (id, name, description, file_name, created_at)
    """
    with engine.connect() as connection:
        stmt = select(
            saved_datasets.c.id,
            saved_datasets.c.name, 
            saved_datasets.c.description,
            saved_datasets.c.file_name,
            saved_datasets.c.created_at
        )
        result = connection.execute(stmt)
        return result.fetchall()

def get_dataset(dataset_id):
    """
    Get dataset by ID
    
    Args:
        dataset_id (int): ID of the dataset
        
    Returns:
        pd.DataFrame: Dataframe of the dataset
    """
    with engine.connect() as connection:
        stmt = select(saved_datasets).where(saved_datasets.c.id == dataset_id)
        result = connection.execute(stmt)
        dataset = result.fetchone()
        
        if dataset:
            csv_data = dataset.file_data
            return pd.read_csv(io.BytesIO(csv_data))
        return None

def save_report(title, company_name, description, pdf_data, docx_data=None, report_format='pdf'):
    """
    Save generated report to database in PDF and/or DOCX format
    
    Args:
        title (str): Title of the report
        company_name (str): Company name
        description (str): Description of the report
        pdf_data (bytes): PDF file data
        docx_data (bytes, optional): DOCX file data
        report_format (str, optional): Format of the report ('pdf', 'docx', or 'both')
        
    Returns:
        int: ID of the saved report
    """
    with engine.connect() as connection:
        values = {
            'title': title,
            'company_name': company_name,
            'description': description,
            'pdf_data': pdf_data,
            'report_format': report_format
        }
        
        if docx_data:
            values['docx_data'] = docx_data
            
        stmt = insert(saved_reports).values(**values).returning(saved_reports.c.id)
        
        result = connection.execute(stmt)
        connection.commit()
        return result.scalar_one()

def get_all_reports():
    """
    Get all saved reports
    
    Returns:
        list: List of report records (id, title, company_name, description, created_at)
    """
    with engine.connect() as connection:
        stmt = select(
            saved_reports.c.id,
            saved_reports.c.title, 
            saved_reports.c.company_name,
            saved_reports.c.description,
            saved_reports.c.created_at
        )
        result = connection.execute(stmt)
        return result.fetchall()

def get_report(report_id, format='pdf'):
    """
    Get report by ID
    
    Args:
        report_id (int): ID of the report
        format (str, optional): Format to retrieve ('pdf' or 'docx')
        
    Returns:
        bytes: Report data in requested format
    """
    with engine.connect() as connection:
        if format == 'docx':
            stmt = select(saved_reports.c.docx_data).where(saved_reports.c.id == report_id)
        else:  # Default to PDF
            stmt = select(saved_reports.c.pdf_data).where(saved_reports.c.id == report_id)
        
        result = connection.execute(stmt)
        report = result.fetchone()
        
        if report:
            # Return the first column (either pdf_data or docx_data)
            return report[0]
        return None

def save_insights(dataset_id, insights_list):
    """
    Save insights for a dataset
    
    Args:
        dataset_id (int): ID of the dataset
        insights_list (list): List of insight strings
        
    Returns:
        int: Number of insights saved
    """
    with engine.connect() as connection:
        for idx, insight in enumerate(insights_list):
            # Determine insight type based on content
            insight_type = "general"
            if "missing" in insight.lower():
                insight_type = "missing_values"
            elif "correlation" in insight.lower():
                insight_type = "correlation"
            elif "skew" in insight.lower():
                insight_type = "distribution"
            elif "recommendation" in insight.lower():
                insight_type = "recommendation"
                
            stmt = insert(analysis_insights).values(
                dataset_id=dataset_id,
                insight_text=insight,
                insight_type=insight_type
            )
            connection.execute(stmt)
        
        connection.commit()
        return len(insights_list)

def get_insights(dataset_id=None, insight_type=None):
    """
    Get insights from database
    
    Args:
        dataset_id (int, optional): Filter by dataset ID
        insight_type (str, optional): Filter by insight type
        
    Returns:
        list: List of insight records
    """
    with engine.connect() as connection:
        stmt = select(analysis_insights)
        
        if dataset_id:
            stmt = stmt.where(analysis_insights.c.dataset_id == dataset_id)
        
        if insight_type:
            stmt = stmt.where(analysis_insights.c.insight_type == insight_type)
            
        result = connection.execute(stmt)
        return result.fetchall()

def save_process_metrics(dataset_id, process_type, process_details, execution_time):
    """
    Save processing metrics for performance monitoring
    
    Args:
        dataset_id (int): ID of the dataset
        process_type (str): Type of process (data_cleaning, analysis, visualization, etc.)
        process_details (dict): Details of the process
        execution_time (float): Execution time in seconds
        
    Returns:
        int: ID of the saved metric
    """
    with engine.connect() as connection:
        stmt = insert(processing_metrics).values(
            dataset_id=dataset_id,
            process_type=process_type,
            process_details=process_details,
            execution_time=execution_time
        ).returning(processing_metrics.c.id)
        
        result = connection.execute(stmt)
        connection.commit()
        return result.scalar_one()

def get_or_create_setting(setting_name, default_value=None):
    """
    Get setting or create if not exists
    
    Args:
        setting_name (str): Name of the setting
        default_value (str, optional): Default value if setting doesn't exist
        
    Returns:
        str: Setting value
    """
    with engine.connect() as connection:
        # Try to get existing setting
        stmt = select(user_settings.c.setting_value).where(user_settings.c.setting_name == setting_name)
        result = connection.execute(stmt)
        setting = result.fetchone()
        
        if setting:
            return setting.setting_value
        
        # If setting doesn't exist, create it with default value
        if default_value is not None:
            stmt = insert(user_settings).values(
                setting_name=setting_name,
                setting_value=str(default_value)
            )
            connection.execute(stmt)
            connection.commit()
            return default_value
        
        return None

def update_setting(setting_name, setting_value):
    """
    Update setting value
    
    Args:
        setting_name (str): Name of the setting
        setting_value (str): New value
        
    Returns:
        bool: Success status
    """
    with engine.connect() as connection:
        stmt = update(user_settings).where(
            user_settings.c.setting_name == setting_name
        ).values(
            setting_value=str(setting_value),
            updated_at=datetime.now()
        )
        
        result = connection.execute(stmt)
        connection.commit()
        return result.rowcount > 0

# Performance measurement decorator
def measure_execution_time(dataset_id, process_type):
    """
    Decorator to measure execution time and save to database
    
    Args:
        dataset_id (int): ID of the dataset
        process_type (str): Type of process
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Extract function args for details
            process_details = {
                "function": func.__name__,
                "arguments": str(args),
                "keyword_arguments": str(kwargs)
            }
            
            # Save metrics
            try:
                save_process_metrics(
                    dataset_id=dataset_id,
                    process_type=process_type,
                    process_details=process_details,
                    execution_time=execution_time
                )
            except Exception as e:
                print(f"Failed to save metrics: {e}")
                
            return result
        return wrapper
    return decorator

# Check database connection
def check_database_connection():
    """
    Check if database connection is working
    
    Returns:
        bool: True if connection is working
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False