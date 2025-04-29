import io
import os
import pandas as pd
import numpy as np
import datetime
import tempfile
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docxtpl import DocxTemplate, InlineImage
from docx.enum.dml import MSO_THEME_COLOR_INDEX

class DocxGenerator:
    """
    Class for generating DOCX reports with formula support
    """
    
    def __init__(self, form_data, data, insights, missing_fig=None, numeric_fig=None, categorical_fig=None):
        """
        Initialize the DocxGenerator
        
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
        self.insights = insights if insights else []
        self.missing_fig = missing_fig
        self.numeric_fig = numeric_fig
        self.categorical_fig = categorical_fig
        # Initialize temp_images as instance variable to avoid scoping issues
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
            fig.write_image(temp.name, format='png', width=800, height=500)
            return temp.name
            
    def _insert_formula(self, document, formula_text, paragraph=None, is_dax=False):
        """
        Insert a mathematical formula into the document
        
        Args:
            document (Document): The document object
            formula_text (str): The formula text in LaTeX format or DAX format
            paragraph (Paragraph, optional): Paragraph to add the formula to. If None, creates a new paragraph.
            is_dax (bool, optional): If True, treat formula_text as DAX formula instead of LaTeX
            
        Returns:
            Paragraph: The paragraph containing the formula
        """
        if paragraph is None:
            paragraph = document.add_paragraph()
        
        if is_dax:
            # For DAX formulas, we use a different style (code-like formatting)
            run = paragraph.add_run(formula_text)
            run.font.name = 'Consolas'
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 0, 139)  # Dark blue color for DAX code
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            
            # Add a light gray shading to the paragraph for DAX formulas
            shading_elm = OxmlElement('w:shd')
            shading_elm.set(qn('w:fill'), 'F2F2F2')  # Light gray background
            paragraph._element.get_or_add_pPr().append(shading_elm)
        else:
            # Simpler approach for math formulas - just display as formatted text
            # instead of using Office Math Markup Language which can cause XML issues
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(formula_text)
            run.font.italic = True
            run.font.name = 'Cambria Math'
            run.font.size = Pt(12)
            run.font.bold = False
            
            # Add a note that this is a mathematical formula
            note_run = paragraph.add_run()
            note_run.add_break()
            note_run.add_text("(Mathematical Formula)")
            note_run.font.size = Pt(8)
            note_run.font.italic = True
            note_run.font.color.rgb = RGBColor(100, 100, 100)
        
        return paragraph
        
    # Removed _insert_lookup_function method as this functionality has been moved to a dedicated page

    def generate_docx(self):
        """
        Generate a DOCX report with formula support
        
        Returns:
            bytes: DOCX file as bytes
        """
        # Create a new document
        doc = Document()
        
        # Set up document properties
        doc.core_properties.title = "Easy AI Analytics Report"
        doc.core_properties.author = "Easy AI Analytics"
        doc.core_properties.comments = f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d')}"
        
        # Add title
        title = doc.add_heading('Easy AI Analytics Report', level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add company and project info
        doc.add_heading(f"Company: {self.form_data.get('company_name', 'N/A')}", level=1)
        doc.add_heading(f"Project: {self.form_data.get('project_title', 'N/A')}", level=1)
        
        # Add date
        date_paragraph = doc.add_paragraph()
        date_paragraph.add_run(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d')}").bold = True
        
        # Add business objectives
        doc.add_heading("Business Objectives", level=2)
        doc.add_paragraph(self.form_data.get('objectives', 'N/A'))
        
        # Add target audience
        doc.add_heading("Target Audience", level=2)
        doc.add_paragraph(self.form_data.get('target_audience', 'N/A'))
        
        # Add timeframe
        doc.add_heading("Analysis Timeframe", level=2)
        doc.add_paragraph(self.form_data.get('timeframe', 'N/A'))
        
        # Add dataset overview with error handling
        doc.add_heading("Dataset Overview", level=2)
        if self.data is not None and not self.data.empty:
            p = doc.add_paragraph()
            p.add_run(f"Rows: ").bold = True
            p.add_run(f"{self.data.shape[0]}")
            
            p = doc.add_paragraph()
            p.add_run(f"Columns: ").bold = True
            p.add_run(f"{self.data.shape[1]}")
            
            p = doc.add_paragraph()
            p.add_run(f"Missing Values: ").bold = True
            p.add_run(f"{self.data.isna().sum().sum()}")
            
            # Add a sample formula for dataset statistics
            doc.add_heading("Statistical Formulas Used in Analysis", level=3)
            self._insert_formula(doc, "\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n}x_i")
            doc.add_paragraph("Formula for calculating the mean value of the dataset")
            
            self._insert_formula(doc, "s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})^2}")
            doc.add_paragraph("Formula for calculating the standard deviation")
            
            # Add correlation formula if data includes numeric columns
            if self.data.select_dtypes(include=['number']).shape[1] > 1:
                self._insert_formula(doc, "r_{xy} = \\frac{\\sum_{i=1}^{n}(x_i-\\bar{x})(y_i-\\bar{y})}{\\sqrt{\\sum_{i=1}^{n}(x_i-\\bar{x})^2\\sum_{i=1}^{n}(y_i-\\bar{y})^2}}")
                doc.add_paragraph("Formula for calculating the Pearson correlation coefficient between variables")
                
            # Note: Advanced Formulas & Lookups section has been moved to a dedicated page in the application
        else:
            doc.add_paragraph("No data available for analysis")
        
        # Add visualizations if available
        if any([self.missing_fig, self.numeric_fig, self.categorical_fig]):
            doc.add_heading("Data Visualizations", level=2)
            
            if self.missing_fig:
                try:
                    img_path = self._save_figure_as_image(self.missing_fig)
                    doc.add_heading("Missing Values Chart", level=3)
                    doc.add_picture(img_path, width=Inches(6))
                    self.temp_images.append(img_path)
                except Exception as e:
                    doc.add_paragraph(f"Error generating missing values chart: {e}")
            
            if self.numeric_fig:
                try:
                    img_path = self._save_figure_as_image(self.numeric_fig)
                    doc.add_heading("Numeric Distribution Chart", level=3)
                    doc.add_picture(img_path, width=Inches(6))
                    self.temp_images.append(img_path)
                except Exception as e:
                    doc.add_paragraph(f"Error generating numeric distribution chart: {e}")
            
            if self.categorical_fig:
                try:
                    img_path = self._save_figure_as_image(self.categorical_fig)
                    doc.add_heading("Categorical Distribution Chart", level=3)
                    doc.add_picture(img_path, width=Inches(6))
                    self.temp_images.append(img_path)
                except Exception as e:
                    doc.add_paragraph(f"Error generating categorical distribution chart: {e}")
        
        # Add key metrics
        doc.add_heading("Key Metrics to Track", level=2)
        doc.add_paragraph(self.form_data.get('key_metrics', 'N/A'))
        
        # Add insights
        doc.add_heading("Data Insights", level=2)
        for insight in self.insights:
            doc.add_paragraph(insight, style='List Bullet')
        
        # Add sample data table
        doc.add_heading("Data Sample", level=2)
        
        if self.data is not None and not self.data.empty:
            # Convert any problematic columns to string type
            safe_data = self.data.copy()
            for col in safe_data.columns:
                if pd.api.types.is_datetime64_any_dtype(safe_data[col]):
                    safe_data[col] = safe_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Create a table for the data
            # Add headers as first row
            table = doc.add_table(rows=1, cols=len(safe_data.columns))
            table.style = 'Table Grid'
            
            # Add header row
            header_cells = table.rows[0].cells
            for i, column_name in enumerate(safe_data.columns):
                header_cells[i].text = str(column_name)
                # Format header cells
                for paragraph in header_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
            
            # Add data rows (up to 5 rows)
            for i in range(min(5, len(safe_data))):
                row_cells = table.add_row().cells
                for j, value in enumerate(safe_data.iloc[i]):
                    if isinstance(value, (pd.Timestamp, np.datetime64)):
                        cell_text = str(pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S'))
                    else:
                        cell_text = str(value)
                    row_cells[j].text = cell_text
        else:
            doc.add_paragraph("No data available to display")
        
        # Save to bytes
        f = io.BytesIO()
        doc.save(f)
        
        # Clean up temporary image files
        for img_path in self.temp_images:
            try:
                os.unlink(img_path)
            except:
                pass
        
        # Get the value of the buffer and return it
        f.seek(0)
        return f.getvalue()