import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    """
    Class for creating visualizations from the data
    """
    
    def __init__(self, data):
        """
        Initialize the Visualizer with a pandas DataFrame
        
        Args:
            data (pd.DataFrame): The data to visualize
        """
        self.data = data
    
    def plot_missing_values(self):
        """
        Create a bar chart showing missing values by column
        
        Returns:
            plotly.graph_objects.Figure: A bar chart of missing values
        """
        missing = self.data.isna().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing Percentage'] = (missing['Missing Count'] / len(self.data) * 100).round(2)
        
        # Sort by missing count descending
        missing = missing.sort_values('Missing Count', ascending=False)
        
        # Only show columns with missing values
        missing = missing[missing['Missing Count'] > 0]
        
        if len(missing) == 0:
            # Create a figure with a message if no missing values
            fig = go.Figure()
            fig.add_annotation(
                text="No Missing Values",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        fig = px.bar(
            missing,
            x='Column',
            y='Missing Count',
            text='Missing Percentage',
            title='Missing Values by Column',
            labels={'Missing Count': 'Number of Missing Values', 'Column': 'Column Name'},
            color='Missing Percentage',
            color_continuous_scale='Reds',
            height=400
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title='Column',
            yaxis_title='Missing Count',
            coloraxis_colorbar=dict(title='Missing %')
        )
        
        return fig
    
    def plot_numeric_distribution(self, column):
        """
        Create a histogram for a numeric column
        
        Args:
            column (str): The column to visualize
        
        Returns:
            plotly.graph_objects.Figure: A histogram
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        fig = px.histogram(
            self.data,
            x=column,
            title=f'Distribution of {column}',
            labels={column: column},
            color_discrete_sequence=['#1E88E5'],
            height=400
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Count'
        )
        
        # Add a box plot on the second y-axis
        fig.add_trace(
            go.Box(
                x=self.data[column],
                name='Box Plot',
                marker_color='#1E88E5',
                boxmean=True,
                orientation='h',
                y0=0,
                yaxis='y2'
            )
        )
        
        # Update layout to include second y-axis
        fig.update_layout(
            yaxis2=dict(
                overlaying='y',
                side='right',
                showticklabels=False,
                range=[0, 0.1]
            )
        )
        
        return fig
    
    def plot_categorical_distribution(self, column):
        """
        Create a pie or bar chart for a categorical column
        
        Args:
            column (str): The column to visualize
        
        Returns:
            plotly.graph_objects.Figure: A pie chart or bar chart
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        value_counts = self.data[column].value_counts()
        
        # If there are too many categories, limit to top 10 and group others
        if len(value_counts) > 10:
            top_values = value_counts.head(9)
            others = pd.Series({'Others': value_counts[9:].sum()})
            value_counts = pd.concat([top_values, others])
        
        fig = px.pie(
            names=value_counts.index,
            values=value_counts.values,
            title=f'Distribution of {column}',
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=400
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def plot_correlation_matrix(self):
        """
        Create a heatmap of the correlation matrix for numeric columns
        
        Returns:
            plotly.graph_objects.Figure: A correlation heatmap
        """
        numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        
        if numeric_data.shape[1] < 2:
            # Create a figure with a message if not enough numeric columns
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough numeric columns for correlation",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        corr = numeric_data.corr()
        
        fig = px.imshow(
            corr,
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title='Correlation Matrix',
            height=500
        )
        
        # Add correlation values as text
        annotations = []
        for i, row in enumerate(corr.values):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{value:.2f}",
                        showarrow=False,
                        font=dict(
                            color='white' if abs(value) > 0.5 else 'black'
                        )
                    )
                )
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def plot_scatter(self, x_column, y_column, color_column=None):
        """
        Create a scatter plot of two numeric columns
        
        Args:
            x_column (str): The column for the x-axis
            y_column (str): The column for the y-axis
            color_column (str, optional): The column to use for coloring points
        
        Returns:
            plotly.graph_objects.Figure: A scatter plot
        """
        if x_column not in self.data.columns or y_column not in self.data.columns:
            raise ValueError(f"Columns '{x_column}' or '{y_column}' not found in data")
        
        if not pd.api.types.is_numeric_dtype(self.data[x_column]) or not pd.api.types.is_numeric_dtype(self.data[y_column]):
            raise ValueError(f"Columns '{x_column}' and '{y_column}' must be numeric")
        
        if color_column and color_column not in self.data.columns:
            raise ValueError(f"Column '{color_column}' not found in data")
        
        fig = px.scatter(
            self.data,
            x=x_column,
            y=y_column,
            color=color_column,
            title=f'{y_column} vs {x_column}',
            height=400,
            opacity=0.7
        )
        
        # Add trend line
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column
        )
        
        return fig
