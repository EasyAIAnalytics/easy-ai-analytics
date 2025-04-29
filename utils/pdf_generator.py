import io
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import plotly.io as pio
import tempfile
import os
import datetime

class PDFGenerator:
    """
    Class for generating PDF reports
    """
    
    def __init__(self, form_data, data, insights, missing_fig=None, numeric_fig=None, categorical_fig=None):
        """
        Initialize the PDFGenerator
        
        Args:
            form_data (dict): Form data submitted by user
            data (pd.DataFrame): The dataset being analyzed
            insights (list): List of insights generated from the data
            missing_fig (plotly.graph_objects.Figure, optional): Missing values chart
            numeric_fig (plotly.graph_objects.Figure, optional): Numeric distribution chart
            categorical_fig (plotly.graph_objects.Figure, optional): Categorical distribution chart
        """
        self.form_data = form_data
        self.data = data
        self.insights = insights
        self.missing_fig = missing_fig
        self.numeric_fig = numeric_fig
        self.categorical_fig = categorical_fig
        
    def _save_figure_as_image(self, fig):
        """
        Save a Plotly figure as an image
        
        Args:
            fig (plotly.graph_objects.Figure): The figure to save
        
        Returns:
            str: Path to the saved image
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            pio.write_image(fig, temp.name, format='png', width=600, height=400)
            return temp.name
    
    def generate_pdf(self):
        """
        Generate a PDF report
        
        Returns:
            bytes: PDF file as bytes
        """
        # Create a buffer to receive the PDF data
        buffer = io.BytesIO()
        
        # Create the PDF object
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create a list to store content
        content = []
        
        # Add logo
        logo_path = 'assets/logo.svg'
        if os.path.exists(logo_path):
            content.append(Image(logo_path, width=2*inch, height=2*inch))
            content.append(Spacer(1, 0.1*inch))
        
        # Add title
        title_style = styles['Title']
        content.append(Paragraph("Easy AI Analytics Report", title_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Add company and project info
        company_style = styles['Heading1']
        content.append(Paragraph(f"Company: {self.form_data.get('company_name', 'N/A')}", company_style))
        content.append(Paragraph(f"Project: {self.form_data.get('project_title', 'N/A')}", company_style))
        content.append(Spacer(1, 0.1*inch))
        
        # Add date
        date_style = styles['Normal']
        content.append(Paragraph(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", date_style))
        content.append(Spacer(1, 0.25*inch))
        
        # Add business objectives
        section_style = styles['Heading2']
        content.append(Paragraph("Business Objectives", section_style))
        content.append(Paragraph(self.form_data.get('objectives', 'N/A'), styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Add target audience
        content.append(Paragraph("Target Audience", section_style))
        content.append(Paragraph(self.form_data.get('target_audience', 'N/A'), styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Add timeframe
        content.append(Paragraph("Analysis Timeframe", section_style))
        content.append(Paragraph(self.form_data.get('timeframe', 'N/A'), styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Add dataset overview
        content.append(Paragraph("Dataset Overview", section_style))
        content.append(Paragraph(f"Rows: {self.data.shape[0]}", styles['Normal']))
        content.append(Paragraph(f"Columns: {self.data.shape[1]}", styles['Normal']))
        content.append(Paragraph(f"Missing Values: {self.data.isna().sum().sum()}", styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Add visualizations if available
        if any([self.missing_fig, self.numeric_fig, self.categorical_fig]):
            content.append(Paragraph("Data Visualizations", section_style))
            
            # Save figures as temporary images
            temp_images = []
            
            if self.missing_fig:
                try:
                    img_path = self._save_figure_as_image(self.missing_fig)
                    content.append(Paragraph("Missing Values Chart", styles['Heading3']))
                    content.append(Image(img_path, width=6*inch, height=4*inch))
                    content.append(Spacer(1, 0.2*inch))
                    temp_images.append(img_path)
                except Exception as e:
                    content.append(Paragraph(f"Error generating missing values chart: {e}", styles['Normal']))
            
            if self.numeric_fig:
                try:
                    img_path = self._save_figure_as_image(self.numeric_fig)
                    content.append(Paragraph("Numeric Distribution Chart", styles['Heading3']))
                    content.append(Image(img_path, width=6*inch, height=4*inch))
                    content.append(Spacer(1, 0.2*inch))
                    temp_images.append(img_path)
                except Exception as e:
                    content.append(Paragraph(f"Error generating numeric distribution chart: {e}", styles['Normal']))
            
            if self.categorical_fig:
                try:
                    img_path = self._save_figure_as_image(self.categorical_fig)
                    content.append(Paragraph("Categorical Distribution Chart", styles['Heading3']))
                    content.append(Image(img_path, width=6*inch, height=4*inch))
                    content.append(Spacer(1, 0.2*inch))
                    temp_images.append(img_path)
                except Exception as e:
                    content.append(Paragraph(f"Error generating categorical distribution chart: {e}", styles['Normal']))
        
        # Add key metrics
        content.append(Paragraph("Key Metrics to Track", section_style))
        content.append(Paragraph(self.form_data.get('key_metrics', 'N/A'), styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Add insights
        content.append(Paragraph("Data Insights", section_style))
        for insight in self.insights:
            content.append(Paragraph(insight, styles['Normal']))
            content.append(Spacer(1, 0.1*inch))
        
        # Add sample data table
        content.append(Paragraph("Data Sample", section_style))
        
        # Create the table data including headers
        table_data = [self.data.columns.tolist()]
        
        # Add up to 5 rows of data
        for i in range(min(5, len(self.data))):
            table_data.append([str(cell) for cell in self.data.iloc[i].values])
        
        # Create the table
        table = Table(table_data)
        
        # Add style to the table
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        table.setStyle(table_style)
        
        content.append(table)
        
        # Build the PDF document
        doc.build(content)
        
        # Clean up temporary image files
        if 'temp_images' in locals():
            for img_path in temp_images:
                try:
                    os.unlink(img_path)
                except:
                    pass
        
        # Get the value of the buffer and return it
        buffer.seek(0)
        return buffer.getvalue()
