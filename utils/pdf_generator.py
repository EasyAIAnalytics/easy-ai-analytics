import io
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.units import inch
import plotly.io as pio
import tempfile
import os
import datetime
from reportlab.lib.enums import TA_CENTER

class PDFGenerator:
    """
    Class for generating PDF reports
    """
    
    def __init__(self, form_data, data, insights, chart_images=None, missing_fig=None, numeric_fig=None, categorical_fig=None):
        """
        Initialize the PDFGenerator
        
        Args:
            form_data (dict): Form data submitted by user
            data (pd.DataFrame): The dataset being analyzed
            insights (list): List of insights generated from the data
            chart_images (list): List of (chart_name, image_bytes) tuples
            missing_fig (plotly.graph_objects.Figure, optional): Missing values chart
            numeric_fig (plotly.graph_objects.Figure, optional): Numeric distribution chart
            categorical_fig (plotly.graph_objects.Figure, optional): Categorical distribution chart
        """
        self.form_data = form_data
        self.data = data
        self.insights = insights if insights else []
        self.chart_images = chart_images if chart_images else []
        self.missing_fig = missing_fig
        self.numeric_fig = numeric_fig
        self.categorical_fig = categorical_fig
        self.temp_images = []
        
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
        
        # Add logo - using PNG instead of SVG for better compatibility
        try:
            # Try with SVG first
            logo_path = 'assets/logo.svg'
            if os.path.exists(logo_path):
                # Convert SVG to PNG using reportlab's rendering
                from reportlab.graphics import renderPM
                from svglib.svglib import svg2rlg
                
                # Create a temporary PNG file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                    temp_png = temp.name
                    try:
                        drawing = svg2rlg(logo_path)
                        renderPM.drawToFile(drawing, temp_png, fmt='PNG')
                        content.append(Image(temp_png, width=2*inch, height=2*inch))
                        self.temp_images.append(temp_png)  # Add to temp files for cleanup
                    except Exception as e:
                        # If SVG conversion fails, add a text header instead
                        content.append(Paragraph("Easy AI Analytics", styles['Title']))
            else:
                # If no logo file, add a text header
                content.append(Paragraph("Easy AI Analytics", styles['Title']))
        except Exception as e:
            # Fallback to text if any error occurs
            content.append(Paragraph("Easy AI Analytics", styles['Title']))
            
        content.append(Spacer(1, 0.1*inch))
        
        # Center the main title and add more spacing
        centered_title_style = ParagraphStyle(
            name='CenteredTitle', parent=styles['Title'], alignment=TA_CENTER, fontSize=22, spaceAfter=18
        )
        content.append(Paragraph("Easy AI Analytics Report", centered_title_style))
        content.append(Spacer(1, 0.35*inch))
        
        # Add a horizontal rule function
        def add_section_divider():
            content.append(Spacer(1, 0.1*inch))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            content.append(Spacer(1, 0.1*inch))
        
        # Add company and project info
        company_style = styles['Heading1']
        content.append(Paragraph(f"Company: {self.form_data.get('company_name', 'N/A')}", company_style))
        content.append(Paragraph(f"Project: {self.form_data.get('project_title', 'N/A')}", company_style))
        add_section_divider()
        
        # Add date
        date_style = styles['Normal']
        content.append(Paragraph(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", date_style))
        add_section_divider()
        
        # Add business objectives
        section_style = styles['Heading2']
        content.append(Paragraph("Business Objectives", section_style))
        content.append(Paragraph(self.form_data.get('objectives', 'N/A'), styles['Normal']))
        add_section_divider()
        
        # Add target audience
        content.append(Paragraph("Target Audience", section_style))
        content.append(Paragraph(self.form_data.get('target_audience', 'N/A'), styles['Normal']))
        add_section_divider()
        
        # Add timeframe
        content.append(Paragraph("Analysis Timeframe", section_style))
        content.append(Paragraph(self.form_data.get('timeframe', 'N/A'), styles['Normal']))
        add_section_divider()
        
        # Add dataset overview with error handling
        content.append(Paragraph("Dataset Overview", section_style))
        if self.data is not None and not self.data.empty:
            content.append(Paragraph(f"Rows: {self.data.shape[0]}", styles['Normal']))
            content.append(Paragraph(f"Columns: {self.data.shape[1]}", styles['Normal']))
            content.append(Paragraph(f"Missing Values: {self.data.isna().sum().sum()}", styles['Normal']))
        else:
            content.append(Paragraph("No data available for analysis", styles['Normal']))
        add_section_divider()
        
        # Add visualizations if available
        if any([self.missing_fig, self.numeric_fig, self.categorical_fig]):
            content.append(Paragraph("Data Visualizations", section_style))
            
            if self.missing_fig:
                try:
                    img_path = self._save_figure_as_image(self.missing_fig)
                    content.append(Paragraph("Missing Values Chart", styles['Heading3']))
                    content.append(Image(img_path, width=6*inch, height=4*inch))
                    content.append(Spacer(1, 0.2*inch))
                    self.temp_images.append(img_path)
                except Exception as e:
                    content.append(Paragraph(f"Error generating missing values chart: {e}", styles['Normal']))
            
            if self.numeric_fig:
                try:
                    img_path = self._save_figure_as_image(self.numeric_fig)
                    content.append(Paragraph("Numeric Distribution Chart", styles['Heading3']))
                    content.append(Image(img_path, width=6*inch, height=4*inch))
                    content.append(Spacer(1, 0.2*inch))
                    self.temp_images.append(img_path)
                except Exception as e:
                    content.append(Paragraph(f"Error generating numeric distribution chart: {e}", styles['Normal']))
            
            if self.categorical_fig:
                try:
                    img_path = self._save_figure_as_image(self.categorical_fig)
                    content.append(Paragraph("Categorical Distribution Chart", styles['Heading3']))
                    content.append(Image(img_path, width=6*inch, height=4*inch))
                    content.append(Spacer(1, 0.2*inch))
                    self.temp_images.append(img_path)
                except Exception as e:
                    content.append(Paragraph(f"Error generating categorical distribution chart: {e}", styles['Normal']))
        
        # Add visualizations from chart_images
        if self.chart_images:
            for chart_name, img_bytes in self.chart_images:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                    temp_img.write(img_bytes)
                    temp_img_path = temp_img.name
                    self.temp_images.append(temp_img_path)
                content.append(Paragraph(chart_name, styles['Heading3']))
                content.append(
                    Image(temp_img_path, width=6.5*inch, height=4.5*inch, hAlign='CENTER')
                )
                content.append(Spacer(1, 0.2*inch))
        
        # Add key metrics
        content.append(Paragraph("Key Metrics to Track", section_style))
        content.append(Paragraph(self.form_data.get('key_metrics', 'N/A'), styles['Normal']))
        add_section_divider()
        
        # Add insights
        content.append(Paragraph("Data Insights", section_style))
        for insight in self.insights:
            content.append(Paragraph(insight, styles['Normal']))
            content.append(Spacer(1, 0.1*inch))
        
        # Add sample data table
        content.append(Paragraph("Data Sample", section_style))
        
        # For the data sample, use a well-formatted table
        if self.data is not None and not self.data.empty:
            sample_df = self.data.head(5)
            table_data = [sample_df.columns.tolist()] + sample_df.values.tolist()
            table = Table(table_data, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#636EFA')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            content.append(table)
            content.append(Spacer(1, 0.2*inch))
        
        # Ensure self.temp_images is defined
        if not hasattr(self, 'temp_images'):
            self.temp_images = []
            
        # Build the PDF document
        try:
            doc.build(content)
        except Exception as e:
            # If document building fails, create a simple error document
            buffer = io.BytesIO()
            simple_doc = SimpleDocTemplate(buffer, pagesize=letter)
            error_content = []
            error_content.append(Paragraph("Error Generating Report", styles['Title']))
            error_content.append(Spacer(1, 0.25*inch))
            error_content.append(Paragraph(f"An error occurred: {str(e)}", styles['Normal']))
            simple_doc.build(error_content)
            
        # Clean up temporary image files
        for img_path in self.temp_images:
            try:
                os.unlink(img_path)
            except:
                pass
        
        # Get the value of the buffer and return it
        buffer.seek(0)
        return buffer.getvalue()
