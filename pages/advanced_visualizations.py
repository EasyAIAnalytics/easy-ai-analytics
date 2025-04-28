import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
import json
import os
import re
import uuid
import plotly.utils
from datetime import datetime, timedelta

# Helper function for creating charts based on configuration
def create_chart_from_config(data, config):
    chart_type = config.get("type")
    
    if chart_type == "Bar Chart":
        x_column = config.get("x_column")
        y_column = config.get("y_column")
        color_column = config.get("color_column")
        orientation = config.get("orientation", "Vertical")
        bar_mode = config.get("bar_mode", "group")
        
        if orientation == "Vertical":
            if color_column:
                fig = px.bar(
                    data,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    title=config.get("title", "Bar Chart"),
                    barmode=bar_mode
                )
            else:
                fig = px.bar(
                    data,
                    x=x_column,
                    y=y_column,
                    title=config.get("title", "Bar Chart"),
                    barmode=bar_mode
                )
        else:  # Horizontal
            if color_column:
                fig = px.bar(
                    data,
                    y=x_column,  # Swap x and y for horizontal
                    x=y_column,
                    color=color_column,
                    title=config.get("title", "Bar Chart"),
                    barmode=bar_mode,
                    orientation='h'
                )
            else:
                fig = px.bar(
                    data,
                    y=x_column,  # Swap x and y for horizontal
                    x=y_column,
                    title=config.get("title", "Bar Chart"),
                    barmode=bar_mode,
                    orientation='h'
                )
        
        return fig
    
    elif chart_type == "Line Chart":
        x_column = config.get("x_column")
        y_column = config.get("y_column")
        color_column = config.get("color_column")
        line_shape = config.get("line_shape", "linear")
        
        if color_column:
            fig = px.line(
                data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=config.get("title", "Line Chart"),
                line_shape=line_shape
            )
        else:
            fig = px.line(
                data,
                x=x_column,
                y=y_column,
                title=config.get("title", "Line Chart"),
                line_shape=line_shape
            )
        
        return fig
    
    elif chart_type == "Scatter Plot":
        x_column = config.get("x_column")
        y_column = config.get("y_column")
        color_column = config.get("color_column")
        size_column = config.get("size_column")
        
        if color_column and size_column:
            fig = px.scatter(
                data,
                x=x_column,
                y=y_column,
                color=color_column,
                size=size_column,
                title=config.get("title", "Scatter Plot")
            )
        elif color_column:
            fig = px.scatter(
                data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=config.get("title", "Scatter Plot")
            )
        elif size_column:
            fig = px.scatter(
                data,
                x=x_column,
                y=y_column,
                size=size_column,
                title=config.get("title", "Scatter Plot")
            )
        else:
            fig = px.scatter(
                data,
                x=x_column,
                y=y_column,
                title=config.get("title", "Scatter Plot")
            )
        
        return fig
    
    elif chart_type == "Pie Chart":
        value_column = config.get("value_column")
        names_column = config.get("names_column")
        is_donut = config.get("is_donut", False)
        
        fig = px.pie(
            data,
            values=value_column,
            names=names_column,
            title=config.get("title", "Pie Chart")
        )
        
        if is_donut:
            fig.update_traces(hole=0.4)
        
        return fig
    
    elif chart_type == "Heatmap":
        x_column = config.get("x_column")
        y_column = config.get("y_column")
        value_column = config.get("value_column")
        color_scale = config.get("color_scale", "viridis")
        
        # Create pivot table
        pivot_data = data.pivot_table(
            values=value_column,
            index=y_column,
            columns=x_column,
            aggfunc='mean'
        )
        
        fig = px.imshow(
            pivot_data,
            title=config.get("title", "Heatmap"),
            color_continuous_scale=color_scale
        )
        
        return fig
    
    elif chart_type == "Box Plot":
        x_column = config.get("x_column")
        y_column = config.get("y_column")
        color_column = config.get("color_column")
        
        if color_column:
            fig = px.box(
                data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=config.get("title", "Box Plot")
            )
        else:
            fig = px.box(
                data,
                x=x_column,
                y=y_column,
                title=config.get("title", "Box Plot")
            )
        
        return fig
    
    elif chart_type == "Histogram":
        hist_column = config.get("hist_column")
        num_bins = config.get("num_bins", 20)
        color_column = config.get("color_column")
        
        if color_column:
            fig = px.histogram(
                data,
                x=hist_column,
                color=color_column,
                nbins=num_bins,
                title=config.get("title", "Histogram")
            )
        else:
            fig = px.histogram(
                data,
                x=hist_column,
                nbins=num_bins,
                title=config.get("title", "Histogram")
            )
        
        return fig
    
    elif chart_type == "Area Chart":
        x_column = config.get("x_column")
        y_column = config.get("y_column")
        color_column = config.get("color_column")
        
        if color_column:
            fig = px.area(
                data,
                x=x_column,
                y=y_column,
                color=color_column,
                title=config.get("title", "Area Chart")
            )
        else:
            fig = px.area(
                data,
                x=x_column,
                y=y_column,
                title=config.get("title", "Area Chart")
            )
        
        return fig
    
    return None


# Function to handle plotly click events
def plotly_events(fig, click_event=True, select_event=False, hover_event=False):
    """
    Adds callback functionality to a plotly figure for click, select, or hover events.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The plotly figure to add events to
    click_event : bool, default True
        Whether to add click event
    select_event : bool, default False
        Whether to add select event
    hover_event : bool, default False
        Whether to add hover event
        
    Returns:
    --------
    list
        List of dictionaries containing event data or empty list if no events
    """
    # Generate a unique div ID for this chart
    div_id = f"chart_{uuid.uuid4().hex}"
    
    # Convert the figure to JSON
    fig_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    
    # Build the JavaScript code for event handling
    js_code = f"""
    <div id="{div_id}" class="plotly-chart"></div>
    <script>
        var figJson = {fig_json};
        var graphDiv = document.getElementById('{div_id}');
        
        Plotly.newPlot(graphDiv, figJson.data, figJson.layout);
        
        var eventData = [];
        
        if ({str(click_event).lower()}) {{
            graphDiv.on('plotly_click', function(data) {{
                eventData = data;
                document.dispatchEvent(new CustomEvent('onPlotlyEvent', {{ detail: JSON.stringify(data.points) }}));
            }});
        }};
        
        if ({str(select_event).lower()}) {{
            graphDiv.on('plotly_selected', function(data) {{
                if (data) {{
                    eventData = data;
                    document.dispatchEvent(new CustomEvent('onPlotlyEvent', {{ detail: JSON.stringify(data.points) }}));
                }}
            }});
        }};
        
        if ({str(hover_event).lower()}) {{
            graphDiv.on('plotly_hover', function(data) {{
                eventData = data;
                document.dispatchEvent(new CustomEvent('onPlotlyEvent', {{ detail: JSON.stringify(data.points) }}));
            }});
        }};
    </script>
    """
    
    # Create a placeholder for the chart
    chart_placeholder = st.empty()
    
    # Inject HTML/JS code and render the chart
    chart_placeholder.components.html(js_code, height=fig.layout.height or 600)
    
    # Create a placeholder for receiving events via streamlit-javascript-eval
    event_data = st.session_state.get('plotly_event_data', [])
    
    # Check if we have streamlit-javascript-eval installed
    try:
        from streamlit_javascript import st_javascript
        
        # JavaScript to listen for events and return data
        js_result = st_javascript("""
        async function fetchPlotlyEvents() {
            return new Promise((resolve) => {
                let eventData = [];
                document.addEventListener('onPlotlyEvent', function(e) {
                    eventData = JSON.parse(e.detail);
                    resolve(eventData);
                });
                
                // Timeout after 1 second if no event
                setTimeout(() => resolve([]), 1000);
            });
        }
        fetchPlotlyEvents();
        """)
        
        if js_result:
            event_data = js_result
            st.session_state['plotly_event_data'] = event_data
    except:
        # If streamlit-javascript-eval is not available, use a fallback
        st.info("Note: Full interactivity requires the 'streamlit-javascript-eval' package.")
    
    return event_data

# Page configuration
st.set_page_config(
    page_title="Advanced Visualizations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title with styling
st.markdown('<h1 class="main-header">Advanced Visualizations</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Create sophisticated interactive and animated visualizations to explore your data</p>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before creating advanced visualizations.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

if 'saved_visualizations' not in st.session_state:
    st.session_state.saved_visualizations = []

# Create tabs for different advanced visualization options
tab1, tab2, tab3, tab4 = st.tabs([
    "Interactive Dashboards",
    "Drill-Down Analysis",
    "3D Visualizations",
    "Animated Charts"
])

with tab1:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Interactive Dashboard Creator")
    st.markdown("""
    Create a customized dashboard with multiple visualizations. Arrange and save your favorite charts
    for quick access and comprehensive data analysis.
    """)
    
    # Dashboard configuration
    st.markdown("#### Dashboard Configuration")
    
    dashboard_name = st.text_input("Dashboard Name", "My Custom Dashboard", key="dashboard_name_input")
    
    # Dashboard layout
    st.markdown("#### Dashboard Layout")
    
    layout_options = ["1 x 1 (Single chart)", "1 x 2 (Two columns)", "2 x 1 (Two rows)", "2 x 2 (Four charts)"]
    selected_layout = st.selectbox("Select Dashboard Layout", layout_options, key="layout_select")
    
    # Parse layout dimensions
    layout_dims = [int(dim) for dim in re.findall(r'\d+', selected_layout.split('(')[0])]
    num_charts = layout_dims[0] * layout_dims[1]
    
    # Get data columns for visualization
    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
    
    # Try to identify date columns
    date_columns = []
    for col in st.session_state.cleaned_data.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(st.session_state.cleaned_data[col]):
                date_columns.append(col)
            elif pd.api.types.is_object_dtype(st.session_state.cleaned_data[col]):
                # Try to convert to datetime
                if pd.to_datetime(st.session_state.cleaned_data[col], errors='coerce').notna().any():
                    date_columns.append(col)
        except:
            pass
    
    # Chart configuration
    st.markdown("#### Chart Configuration")
    
    charts_config = []
    
    for i in range(num_charts):
        with st.expander(f"Chart {i+1} Configuration", expanded=i==0):
            # Chart type
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap", "Box Plot", "Histogram", "Area Chart"],
                key=f"chart_type_{i}"
            )
            
            # Title
            chart_title = st.text_input("Chart Title", f"Chart {i+1}", key=f"chart_title_{i}")
            
            # Chart specific options
            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Area Chart"]:
                # X-axis
                x_column = st.selectbox(
                    "X-axis Column", 
                    date_columns + categorical_columns + numeric_columns,
                    key=f"x_column_{i}"
                )
                
                # Y-axis
                y_column = st.selectbox(
                    "Y-axis Column", 
                    numeric_columns,
                    key=f"y_column_{i}"
                )
                
                # Color by (optional)
                color_column = st.selectbox(
                    "Color by (optional)",
                    ["None"] + categorical_columns,
                    key=f"color_column_{i}"
                )
                
                # Additional options for specific chart types
                if chart_type == "Bar Chart":
                    orientation = st.selectbox(
                        "Orientation", 
                        ["Vertical", "Horizontal"],
                        key=f"orientation_{i}"
                    )
                    
                    bar_mode = st.selectbox(
                        "Bar Mode", 
                        ["Group", "Stack"],
                        key=f"bar_mode_{i}"
                    )
                    
                    chart_config = {
                        "type": chart_type,
                        "title": chart_title,
                        "x_column": x_column,
                        "y_column": y_column,
                        "color_column": None if color_column == "None" else color_column,
                        "orientation": orientation,
                        "bar_mode": bar_mode.lower()
                    }
                
                elif chart_type == "Line Chart":
                    line_shape = st.selectbox(
                        "Line Shape", 
                        ["Linear", "Spline", "Step"],
                        key=f"line_shape_{i}"
                    )
                    
                    chart_config = {
                        "type": chart_type,
                        "title": chart_title,
                        "x_column": x_column,
                        "y_column": y_column,
                        "color_column": None if color_column == "None" else color_column,
                        "line_shape": line_shape.lower()
                    }
                
                elif chart_type == "Scatter Plot":
                    marker_size_column = st.selectbox(
                        "Marker Size Column (optional)",
                        ["None"] + numeric_columns,
                        key=f"marker_size_{i}"
                    )
                    
                    chart_config = {
                        "type": chart_type,
                        "title": chart_title,
                        "x_column": x_column,
                        "y_column": y_column,
                        "color_column": None if color_column == "None" else color_column,
                        "size_column": None if marker_size_column == "None" else marker_size_column
                    }
                
                elif chart_type == "Box Plot":
                    chart_config = {
                        "type": chart_type,
                        "title": chart_title,
                        "x_column": x_column,
                        "y_column": y_column,
                        "color_column": None if color_column == "None" else color_column
                    }
                
                elif chart_type == "Area Chart":
                    chart_config = {
                        "type": chart_type,
                        "title": chart_title,
                        "x_column": x_column,
                        "y_column": y_column,
                        "color_column": None if color_column == "None" else color_column
                    }
                
            elif chart_type == "Pie Chart":
                # Value column
                value_column = st.selectbox(
                    "Value Column", 
                    numeric_columns,
                    key=f"value_column_{i}"
                )
                
                # Names column
                names_column = st.selectbox(
                    "Names Column", 
                    categorical_columns,
                    key=f"names_column_{i}"
                )
                
                # Donut chart option
                is_donut = st.checkbox("Display as Donut Chart", key=f"is_donut_{i}")
                
                chart_config = {
                    "type": chart_type,
                    "title": chart_title,
                    "value_column": value_column,
                    "names_column": names_column,
                    "is_donut": is_donut
                }
            
            elif chart_type == "Heatmap":
                # Value column
                value_column = st.selectbox(
                    "Value Column", 
                    numeric_columns,
                    key=f"value_column_{i}"
                )
                
                # X-axis categorical column
                x_column = st.selectbox(
                    "X-axis Column", 
                    categorical_columns + date_columns,
                    key=f"x_column_{i}"
                )
                
                # Y-axis categorical column
                y_column = st.selectbox(
                    "Y-axis Column", 
                    categorical_columns,
                    key=f"y_column_{i}"
                )
                
                # Color scale
                color_scale = st.selectbox(
                    "Color Scale", 
                    ["Viridis", "Blues", "Reds", "Purples", "Cividis", "RdBu", "YlGnBu"],
                    key=f"color_scale_{i}"
                )
                
                chart_config = {
                    "type": chart_type,
                    "title": chart_title,
                    "x_column": x_column,
                    "y_column": y_column,
                    "value_column": value_column,
                    "color_scale": color_scale.lower()
                }
            
            elif chart_type == "Histogram":
                # Column for histogram
                hist_column = st.selectbox(
                    "Column for Histogram", 
                    numeric_columns,
                    key=f"hist_column_{i}"
                )
                
                # Number of bins
                num_bins = st.slider(
                    "Number of Bins", 
                    min_value=5, 
                    max_value=50, 
                    value=20,
                    key=f"num_bins_{i}"
                )
                
                # Color
                hist_color = st.selectbox(
                    "Color by (optional)",
                    ["None"] + categorical_columns,
                    key=f"hist_color_{i}"
                )
                
                chart_config = {
                    "type": chart_type,
                    "title": chart_title,
                    "hist_column": hist_column,
                    "num_bins": num_bins,
                    "color_column": None if hist_color == "None" else hist_color
                }
            
            # Add chart config to list
            charts_config.append(chart_config)
    
    # Create dashboard button
    if st.button("Create Dashboard", key="create_dashboard_button"):
        # Create a unique ID for the dashboard
        dashboard_id = f"dashboard_{int(time.time())}"
        
        # Create dashboard config
        dashboard_config = {
            "id": dashboard_id,
            "name": dashboard_name,
            "layout": selected_layout,
            "charts": charts_config,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to session state
        st.session_state.saved_visualizations.append(dashboard_config)
        
        st.success(f"Dashboard '{dashboard_name}' created successfully!")
    
    # Display dashboard preview
    st.markdown("#### Dashboard Preview")
    
    # Create plotly figure based on layout and chart configs
    if num_charts == 1:
        # Single chart
        chart_config = charts_config[0]
        fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Multi-chart layout
        if selected_layout == "1 x 2 (Two columns)":
            cols = st.columns(2)
            for i, col in enumerate(cols):
                if i < len(charts_config):
                    with col:
                        chart_config = charts_config[i]
                        fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        
        elif selected_layout == "2 x 1 (Two rows)":
            for i in range(2):
                if i < len(charts_config):
                    chart_config = charts_config[i]
                    fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        elif selected_layout == "2 x 2 (Four charts)":
            for row in range(2):
                cols = st.columns(2)
                for col_idx, col in enumerate(cols):
                    chart_idx = row * 2 + col_idx
                    if chart_idx < len(charts_config):
                        with col:
                            chart_config = charts_config[chart_idx]
                            fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
    
    # Saved dashboards section
    if st.session_state.saved_visualizations:
        st.markdown("#### Saved Dashboards")
        
        # Create a dropdown to select a dashboard
        dashboard_names = [dash["name"] for dash in st.session_state.saved_visualizations]
        selected_dashboard = st.selectbox("Select a Dashboard", dashboard_names, key="saved_dashboard_select")
        
        # Find the selected dashboard config
        selected_config = next((dash for dash in st.session_state.saved_visualizations if dash["name"] == selected_dashboard), None)
        
        if selected_config:
            st.markdown(f"**Created:** {selected_config['created_at']}")
            
            # Display the selected dashboard
            if len(selected_config["charts"]) == 1:
                # Single chart
                chart_config = selected_config["charts"][0]
                fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Multi-chart layout
                layout = selected_config["layout"]
                charts = selected_config["charts"]
                
                if layout == "1 x 2 (Two columns)":
                    cols = st.columns(2)
                    for i, col in enumerate(cols):
                        if i < len(charts):
                            with col:
                                chart_config = charts[i]
                                fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                
                elif layout == "2 x 1 (Two rows)":
                    for i in range(2):
                        if i < len(charts):
                            chart_config = charts[i]
                            fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                
                elif layout == "2 x 2 (Four charts)":
                    for row in range(2):
                        cols = st.columns(2)
                        for col_idx, col in enumerate(cols):
                            chart_idx = row * 2 + col_idx
                            if chart_idx < len(charts):
                                with col:
                                    chart_config = charts[chart_idx]
                                    fig = create_chart_from_config(st.session_state.cleaned_data, chart_config)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
            
            # Delete dashboard option
            if st.button(f"Delete Dashboard: {selected_dashboard}", key="delete_dashboard_button"):
                st.session_state.saved_visualizations = [
                    dash for dash in st.session_state.saved_visualizations 
                    if dash["name"] != selected_dashboard
                ]
                st.success(f"Dashboard '{selected_dashboard}' deleted successfully!")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Drill-Down Analysis")
    st.markdown("""
    Create interactive visualizations that allow you to click on elements to explore deeper into your data.
    This helps uncover insights at different levels of detail.
    """)
    
    # Configure initial view
    st.markdown("#### Configure Drill-Down Visualization")
    
    # Get data columns for visualization
    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
    
    # Hierarchy levels for drill-down
    st.markdown("##### Define Hierarchy for Drill-Down")
    st.info("Select categorical columns in order from highest to lowest level of detail. For example: Region > Country > City.")
    
    # Maximum 3 hierarchy levels
    if len(categorical_columns) >= 1:
        level1 = st.selectbox("Level 1 (Highest)", categorical_columns, key="level1_select")
        
        # Filter remaining columns for level 2
        level2_options = [col for col in categorical_columns if col != level1]
        level2 = st.selectbox("Level 2 (Optional)", ["None"] + level2_options, key="level2_select") if level2_options else "None"
        
        # Filter remaining columns for level 3
        if level2 != "None":
            level3_options = [col for col in categorical_columns if col not in [level1, level2]]
            level3 = st.selectbox("Level 3 (Optional)", ["None"] + level3_options, key="level3_select") if level3_options else "None"
        else:
            level3 = "None"
        
        # Measure column
        measure = st.selectbox("Measure (Numeric Value)", numeric_columns, key="measure_select")
        
        # Aggregation method
        aggregation = st.selectbox("Aggregation Method", ["Sum", "Average", "Count", "Min", "Max"], key="agg_method_select")
        
        # Chart type
        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Treemap", "Sunburst"], key="drill_chart_type")
        
        # Initialize drill-down state in session state if not exists
        if 'drill_down_state' not in st.session_state:
            st.session_state.drill_down_state = {
                "current_level": 0,  # 0 = level 1, 1 = level 2, etc.
                "filters": {}  # Stores selected values for each level
            }
        
        # Create the drill-down visualization
        if st.button("Generate Drill-Down Visualization", key="generate_drilldown_button"):
            # Reset drill-down state
            st.session_state.drill_down_state = {
                "current_level": 0,
                "filters": {}
            }
            
            st.success("Drill-down visualization created! Click on a category to explore details.")
        
        # Collect hierarchy levels
        hierarchy = [level1]
        if level2 != "None":
            hierarchy.append(level2)
        if level3 != "None":
            hierarchy.append(level3)
        
        # Display current drill-down state
        current_level = st.session_state.drill_down_state["current_level"]
        current_filters = st.session_state.drill_down_state["filters"]
        
        # Show current filters
        if current_filters:
            st.markdown("##### Current Filters")
            for level, value in current_filters.items():
                st.markdown(f"**{level}**: {value}")
        
        # Create title based on current state
        if current_level == 0:
            title = f"{measure} by {hierarchy[0]}"
        else:
            title = f"{measure} by {hierarchy[current_level]}"
            for i in range(current_level):
                filter_value = current_filters.get(hierarchy[i], "All")
                title += f" for {hierarchy[i]} = {filter_value}"
        
        # Prepare data based on current state
        data = st.session_state.cleaned_data.copy()
        
        # Apply filters
        for level, value in current_filters.items():
            data = data[data[level] == value]
        
        # Check if we have data after filtering
        if len(data) == 0:
            st.warning("No data matches the current filters.")
        else:
            # Current level to display
            display_level = hierarchy[current_level] if current_level < len(hierarchy) else None
            
            if display_level:
                # Aggregate data
                if aggregation == "Sum":
                    agg_data = data.groupby(display_level)[measure].sum().reset_index()
                elif aggregation == "Average":
                    agg_data = data.groupby(display_level)[measure].mean().reset_index()
                elif aggregation == "Count":
                    agg_data = data.groupby(display_level)[measure].count().reset_index()
                elif aggregation == "Min":
                    agg_data = data.groupby(display_level)[measure].min().reset_index()
                else:  # Max
                    agg_data = data.groupby(display_level)[measure].max().reset_index()
                
                # Sort data by measure value
                agg_data = agg_data.sort_values(measure, ascending=False)
                
                # Create visualization based on chart type
                if chart_type == "Bar Chart":
                    fig = px.bar(
                        agg_data,
                        x=display_level,
                        y=measure,
                        title=title,
                        color=measure
                    )
                elif chart_type == "Treemap":
                    fig = px.treemap(
                        agg_data,
                        path=[display_level],
                        values=measure,
                        title=title,
                        color=measure
                    )
                else:  # Sunburst
                    fig = px.sunburst(
                        agg_data,
                        path=[display_level],
                        values=measure,
                        title=title,
                        color=measure
                    )
                
                # Add click event note
                if current_level < len(hierarchy) - 1:
                    st.info("👆 Click on a category to drill down to the next level.")
                
                # Show the visualization
                selected_point = plotly_events(fig, click_event=True)
                
                # Handle clicks for drill-down
                if selected_point:
                    clicked_value = None
                    
                    if chart_type == "Bar Chart":
                        # For bar chart, get the x value
                        clicked_idx = selected_point[0]["pointIndex"]
                        clicked_value = agg_data.iloc[clicked_idx][display_level]
                    elif chart_type in ["Treemap", "Sunburst"]:
                        # For treemap/sunburst, get value from label
                        clicked_idx = selected_point[0]["pointIndex"]
                        clicked_value = agg_data.iloc[clicked_idx][display_level]
                    
                    if clicked_value:
                        # Update filters
                        st.session_state.drill_down_state["filters"][display_level] = clicked_value
                        
                        # Move to next level if available
                        if current_level < len(hierarchy) - 1:
                            st.session_state.drill_down_state["current_level"] += 1
                        
                        # Rerun to update the display
                        st.rerun()
                
                # Add button to go back up a level
                if current_level > 0:
                    if st.button("⬆️ Go Back Up a Level", key="go_back_level"):
                        # Remove the current level filter
                        prev_level = hierarchy[current_level - 1]
                        if prev_level in st.session_state.drill_down_state["filters"]:
                            del st.session_state.drill_down_state["filters"][prev_level]
                        
                        # Decrease current level
                        st.session_state.drill_down_state["current_level"] -= 1
                        
                        # Rerun to update the display
                        st.rerun()
            else:
                st.info("You've reached the lowest level of detail.")
                
                # Show detail view for lowest level
                st.dataframe(data)
    else:
        st.warning("Drill-down analysis requires at least one categorical column in your data.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### 3D Visualizations")
    st.markdown("""
    Create interactive 3D visualizations to explore relationships between three variables simultaneously.
    These visualizations offer a more comprehensive view of complex data relationships.
    """)
    
    # 3D visualization options
    st.markdown("#### Configure 3D Visualization")
    
    # Get numeric columns for 3D visualization
    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_columns) < 3:
        st.warning(f"3D visualizations require at least 3 numeric columns. Your data has only {len(numeric_columns)} numeric columns.")
    else:
        # 3D visualization type
        viz_type = st.selectbox(
            "Visualization Type",
            ["3D Scatter", "3D Surface", "3D Line"],
            key="viz_type_select"
        )
        
        # Select columns for visualization
        if viz_type == "3D Scatter":
            x_column = st.selectbox("X-axis Column", numeric_columns, index=0, key="3d_x_col")
            y_column = st.selectbox("Y-axis Column", numeric_columns, index=min(1, len(numeric_columns)-1), key="3d_y_col")
            z_column = st.selectbox("Z-axis Column", numeric_columns, index=min(2, len(numeric_columns)-1), key="3d_z_col")
            
            color_by = st.selectbox("Color By (optional)", ["None"] + categorical_columns + numeric_columns, key="3d_color")
            size_by = st.selectbox("Size By (optional)", ["None"] + numeric_columns, key="3d_size")
            
            # Generate visualization
            if st.button("Generate 3D Scatter Plot", key="gen_3d_scatter"):
                # Create the 3D scatter plot
                fig = go.Figure()
                
                # Prepare data
                data = st.session_state.cleaned_data.dropna(subset=[x_column, y_column, z_column])
                
                if color_by != "None":
                    # Color by a column
                    if color_by in categorical_columns:
                        # Categorical column - use discrete colors
                        for category in data[color_by].unique():
                            subset = data[data[color_by] == category]
                            
                            # Add trace for each category
                            size = None
                            if size_by != "None":
                                # Normalize size to a reasonable range
                                size = subset[size_by] / subset[size_by].max() * 30 + 5
                            
                            fig.add_trace(go.Scatter3d(
                                x=subset[x_column],
                                y=subset[y_column],
                                z=subset[z_column],
                                mode='markers',
                                marker=dict(
                                    size=10 if size_by == "None" else size,
                                    opacity=0.7
                                ),
                                name=str(category)
                            ))
                    else:
                        # Numeric column - use color scale
                        size = None
                        if size_by != "None":
                            # Normalize size to a reasonable range
                            size = data[size_by] / data[size_by].max() * 30 + 5
                        
                        fig.add_trace(go.Scatter3d(
                            x=data[x_column],
                            y=data[y_column],
                            z=data[z_column],
                            mode='markers',
                            marker=dict(
                                size=10 if size_by == "None" else size,
                                color=data[color_by],
                                colorscale='Viridis',
                                opacity=0.7,
                                colorbar=dict(title=color_by)
                            ),
                            text=data.index,
                            hovertemplate=
                                f"{x_column}: %{{x}}<br>" +
                                f"{y_column}: %{{y}}<br>" +
                                f"{z_column}: %{{z}}<br>" +
                                f"{color_by}: %{{marker.color}}<br>" +
                                "<extra></extra>"
                        ))
                else:
                    # No color differentiation
                    size = None
                    if size_by != "None":
                        # Normalize size to a reasonable range
                        size = data[size_by] / data[size_by].max() * 30 + 5
                    
                    fig.add_trace(go.Scatter3d(
                        x=data[x_column],
                        y=data[y_column],
                        z=data[z_column],
                        mode='markers',
                        marker=dict(
                            size=10 if size_by == "None" else size,
                            opacity=0.7
                        ),
                        text=data.index,
                        hovertemplate=
                            f"{x_column}: %{{x}}<br>" +
                            f"{y_column}: %{{y}}<br>" +
                            f"{z_column}: %{{z}}<br>" +
                            "<extra></extra>"
                    ))
                
                # Update layout
                fig.update_layout(
                    title="3D Scatter Plot",
                    scene=dict(
                        xaxis_title=x_column,
                        yaxis_title=y_column,
                        zaxis_title=z_column
                    ),
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                # Show the visualization
                st.plotly_chart(fig, use_container_width=True)
                
                # Instruction for interaction
                st.info("**Tip**: Click and drag to rotate the 3D view. Double-click to reset the view.")
        
        elif viz_type == "3D Surface":
            # For surface plot, need a grid of z values
            x_column = st.selectbox("X-axis Column", numeric_columns, index=0, key="surface_x_col")
            y_column = st.selectbox("Y-axis Column", numeric_columns, index=min(1, len(numeric_columns)-1), key="surface_y_col")
            z_column = st.selectbox("Z-axis Column (Height)", numeric_columns, index=min(2, len(numeric_columns)-1), key="surface_z_col")
            
            # Grid size for interpolation
            grid_size = st.slider("Grid Size", min_value=10, max_value=50, value=20, key="grid_size_slider")
            
            # Color scale
            color_scale = st.selectbox(
                "Color Scale",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Greens", "Reds", "YlOrRd"],
                key="surface_color_scale"
            )
            
            # Generate visualization
            if st.button("Generate 3D Surface Plot", key="gen_3d_surface"):
                # Create the 3D surface plot
                
                # Prepare data
                data = st.session_state.cleaned_data.dropna(subset=[x_column, y_column, z_column])
                
                if len(data) < 10:
                    st.error("Not enough data points for a surface plot. Need at least 10 points.")
                else:
                    try:
                        # Create a grid for interpolation
                        from scipy.interpolate import griddata
                        
                        # Get min/max values for x and y
                        x_min, x_max = data[x_column].min(), data[x_column].max()
                        y_min, y_max = data[y_column].min(), data[y_column].max()
                        
                        # Create grid points
                        x_grid = np.linspace(x_min, x_max, grid_size)
                        y_grid = np.linspace(y_min, y_max, grid_size)
                        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
                        
                        # Interpolate z values
                        z_mesh = griddata(
                            (data[x_column], data[y_column]),
                            data[z_column],
                            (x_mesh, y_mesh),
                            method='cubic',
                            fill_value=data[z_column].mean()
                        )
                        
                        # Create surface plot
                        fig = go.Figure(data=[go.Surface(
                            x=x_grid,
                            y=y_grid,
                            z=z_mesh,
                            colorscale=color_scale.lower()
                        )])
                        
                        # Add scatter points of original data
                        fig.add_trace(go.Scatter3d(
                            x=data[x_column],
                            y=data[y_column],
                            z=data[z_column],
                            mode='markers',
                            marker=dict(
                                size=4,
                                color=data[z_column],
                                colorscale=color_scale.lower(),
                                opacity=0.7
                            ),
                            name='Data Points'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title="3D Surface Plot",
                            scene=dict(
                                xaxis_title=x_column,
                                yaxis_title=y_column,
                                zaxis_title=z_column
                            ),
                            height=700,
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        
                        # Show the visualization
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Instruction for interaction
                        st.info("**Tip**: Click and drag to rotate the 3D view. Double-click to reset the view.")
                    
                    except Exception as e:
                        st.error(f"Error creating surface plot: {str(e)}")
                        st.info("Surface plots work best with continuous and well-distributed data. Try different columns or increase the grid size.")
        
        elif viz_type == "3D Line":
            # For 3D line, we need a sequence/order
            st.info("3D Line plots are ideal for time series data or sequences with a natural order.")
            
            # Select columns
            x_column = st.selectbox("X-axis Column", numeric_columns, index=0, key="line3d_x_col")
            y_column = st.selectbox("Y-axis Column", numeric_columns, index=min(1, len(numeric_columns)-1), key="line3d_y_col")
            z_column = st.selectbox("Z-axis Column", numeric_columns, index=min(2, len(numeric_columns)-1), key="line3d_z_col")
            
            # Select a column for color (optional)
            color_by = st.selectbox("Color By (optional)", ["None"] + categorical_columns + numeric_columns, key="line3d_color")
            
            # Select a column for ordering (optional, but recommended)
            date_columns = []
            for col in st.session_state.cleaned_data.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.cleaned_data[col]):
                        date_columns.append(col)
                    elif pd.api.types.is_object_dtype(st.session_state.cleaned_data[col]):
                        # Try to convert to datetime
                        if pd.to_datetime(st.session_state.cleaned_data[col], errors='coerce').notna().any():
                            date_columns.append(col)
                except:
                    pass
            
            sequence_column = st.selectbox(
                "Sequence Column (for ordering points)",
                ["Index"] + date_columns + numeric_columns,
                key="line3d_sequence"
            )
            
            # Generate visualization
            if st.button("Generate 3D Line Plot", key="gen_3d_line"):
                # Create the 3D line plot
                
                # Prepare data
                data = st.session_state.cleaned_data.dropna(subset=[x_column, y_column, z_column])
                
                if len(data) < 3:
                    st.error("Not enough data points for a line plot. Need at least 3 points.")
                else:
                    # Sort by sequence column if specified
                    if sequence_column != "Index":
                        if sequence_column in date_columns:
                            # Convert to datetime if needed
                            if not pd.api.types.is_datetime64_any_dtype(data[sequence_column]):
                                data[sequence_column] = pd.to_datetime(data[sequence_column], errors='coerce')
                        
                        # Sort by sequence column
                        data = data.sort_values(sequence_column)
                    
                    # Create figure
                    fig = go.Figure()
                    
                    if color_by != "None" and color_by in categorical_columns:
                        # Split by categorical column
                        for category in data[color_by].unique():
                            subset = data[data[color_by] == category]
                            
                            fig.add_trace(go.Scatter3d(
                                x=subset[x_column],
                                y=subset[y_column],
                                z=subset[z_column],
                                mode='lines+markers',
                                name=str(category)
                            ))
                    else:
                        # Single line with optional color scale
                        if color_by != "None" and color_by in numeric_columns:
                            # Color line by numeric value
                            fig.add_trace(go.Scatter3d(
                                x=data[x_column],
                                y=data[y_column],
                                z=data[z_column],
                                mode='lines+markers',
                                line=dict(
                                    color=data[color_by],
                                    colorscale='Viridis',
                                    width=4
                                ),
                                marker=dict(
                                    size=4,
                                    color=data[color_by],
                                    colorscale='Viridis',
                                    opacity=0.8
                                ),
                                text=[f"{sequence_column}: {i}" for i in (data.index if sequence_column == "Index" else data[sequence_column])],
                                hoverinfo='text'
                            ))
                        else:
                            # Simple line
                            fig.add_trace(go.Scatter3d(
                                x=data[x_column],
                                y=data[y_column],
                                z=data[z_column],
                                mode='lines+markers',
                                line=dict(width=4),
                                marker=dict(size=4)
                            ))
                    
                    # Update layout
                    fig.update_layout(
                        title="3D Line Plot",
                        scene=dict(
                            xaxis_title=x_column,
                            yaxis_title=y_column,
                            zaxis_title=z_column
                        ),
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=30)
                    )
                    
                    # Show the visualization
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add animation controls if we have a sequence
                    if sequence_column != "Index":
                        st.markdown("#### Animation Controls")
                        st.info("Use the slider to trace the path through time or sequence.")
                        
                        # Create a slider to show the progression
                        steps = min(20, len(data))
                        step_idx = st.slider("Position in sequence", 1, steps, 1, key="line3d_pos_slider")
                        
                        # Calculate which points to show
                        point_idx = int(len(data) * (step_idx / steps))
                        point_idx = max(1, min(point_idx, len(data)))
                        
                        # Create animated view
                        subset_data = data.iloc[:point_idx]
                        
                        fig_anim = go.Figure()
                        
                        # Add full path in light color
                        fig_anim.add_trace(go.Scatter3d(
                            x=data[x_column],
                            y=data[y_column],
                            z=data[z_column],
                            mode='lines',
                            line=dict(width=2, color='lightgray'),
                            showlegend=False
                        ))
                        
                        # Add path up to current point
                        fig_anim.add_trace(go.Scatter3d(
                            x=subset_data[x_column],
                            y=subset_data[y_column],
                            z=subset_data[z_column],
                            mode='lines+markers',
                            line=dict(width=4),
                            marker=dict(
                                size=4,
                                color=list(range(len(subset_data))),
                                colorscale='Viridis',
                                opacity=0.8
                            ),
                            name="Path"
                        ))
                        
                        # Add current point
                        current_point = subset_data.iloc[-1]
                        fig_anim.add_trace(go.Scatter3d(
                            x=[current_point[x_column]],
                            y=[current_point[y_column]],
                            z=[current_point[z_column]],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color='red',
                                symbol='circle'
                            ),
                            name="Current Position"
                        ))
                        
                        # Update layout
                        fig_anim.update_layout(
                            title=f"3D Line Animation (Position: {step_idx}/{steps})",
                            scene=dict(
                                xaxis_title=x_column,
                                yaxis_title=y_column,
                                zaxis_title=z_column
                            ),
                            height=700,
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        
                        # Show the animated visualization
                        st.plotly_chart(fig_anim, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Animated Charts")
    st.markdown("""
    Create animated visualizations to show how data changes over time or across different categories.
    These dynamic charts help tell a more compelling data story.
    """)
    
    # Animation options
    st.markdown("#### Configure Animation")
    
    # Get columns for animation
    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
    
    # Check for date columns
    date_columns = []
    for col in st.session_state.cleaned_data.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(st.session_state.cleaned_data[col]):
                date_columns.append(col)
            elif pd.api.types.is_object_dtype(st.session_state.cleaned_data[col]):
                # Try to convert to datetime
                if pd.to_datetime(st.session_state.cleaned_data[col], errors='coerce').notna().any():
                    date_columns.append(col)
        except:
            pass
    
    # Animation type
    animation_type = st.selectbox(
        "Animation Type",
        ["Time Series", "Category Transition", "Cumulative Growth"],
        key="animation_type"
    )
    
    if animation_type == "Time Series":
        if not date_columns:
            st.warning("Time Series animation requires at least one date/time column. No date columns detected in your data.")
        else:
            # Configure time series animation
            st.markdown("##### Time Series Animation Configuration")
            
            # Select time column
            time_column = st.selectbox("Time Column", date_columns, key="time_col_anim")
            
            # Verify and potentially convert time column
            time_data = st.session_state.cleaned_data[time_column]
            if not pd.api.types.is_datetime64_any_dtype(time_data):
                # Try to convert to datetime
                try:
                    time_data = pd.to_datetime(time_data)
                    st.info(f"Converting '{time_column}' to datetime format.")
                except:
                    st.error(f"Could not convert '{time_column}' to datetime format.")
                    time_data = None
            
            if time_data is not None:
                # Time unit for animation
                time_unit = st.selectbox(
                    "Time Grouping",
                    ["Day", "Week", "Month", "Quarter", "Year"],
                    key="time_unit_anim"
                )
                
                # Select chart type
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"],
                    key="chart_type_anim"
                )
                
                # Select metrics
                x_column = st.selectbox("X-axis Column", numeric_columns + categorical_columns, index=0, key="x_col_anim")
                y_column = st.selectbox("Y-axis Column", numeric_columns, index=0, key="y_col_anim")
                
                # Optional color column
                color_column = st.selectbox("Color By (optional)", ["None"] + categorical_columns, key="color_col_anim")
                
                # Animation settings
                animation_speed = st.slider("Animation Speed", 500, 2000, 1000, 100, key="anim_speed")
                
                # Generate animation
                if st.button("Generate Time Series Animation", key="gen_time_anim"):
                    # Create the animated chart
                    
                    # Prepare data
                    data = st.session_state.cleaned_data.copy()
                    
                    # Convert time column to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
                    
                    # Filter out rows with invalid dates
                    data = data.dropna(subset=[time_column])
                    
                    # Create time period column based on selected unit
                    if time_unit == "Day":
                        data['period'] = data[time_column].dt.date
                    elif time_unit == "Week":
                        data['period'] = data[time_column].dt.to_period('W').dt.start_time
                    elif time_unit == "Month":
                        data['period'] = data[time_column].dt.to_period('M').dt.start_time
                    elif time_unit == "Quarter":
                        data['period'] = data[time_column].dt.to_period('Q').dt.start_time
                    else:  # Year
                        data['period'] = data[time_column].dt.to_period('Y').dt.start_time
                    
                    # Get unique periods for animation frames
                    periods = sorted(data['period'].unique())
                    
                    # Create the animation based on chart type
                    if chart_type == "Line Chart":
                        if color_column != "None":
                            fig = px.line(
                                data, 
                                x=x_column, 
                                y=y_column, 
                                color=color_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                        else:
                            fig = px.line(
                                data, 
                                x=x_column, 
                                y=y_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                    
                    elif chart_type == "Bar Chart":
                        if color_column != "None":
                            fig = px.bar(
                                data, 
                                x=x_column, 
                                y=y_column, 
                                color=color_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                        else:
                            fig = px.bar(
                                data, 
                                x=x_column, 
                                y=y_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                    
                    elif chart_type == "Scatter Plot":
                        if color_column != "None":
                            fig = px.scatter(
                                data, 
                                x=x_column, 
                                y=y_column, 
                                color=color_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                        else:
                            fig = px.scatter(
                                data, 
                                x=x_column, 
                                y=y_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                    
                    else:  # Area Chart
                        if color_column != "None":
                            fig = px.area(
                                data, 
                                x=x_column, 
                                y=y_column, 
                                color=color_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                        else:
                            fig = px.area(
                                data, 
                                x=x_column, 
                                y=y_column,
                                animation_frame='period',
                                title=f"{y_column} vs {x_column} Over Time",
                                labels={
                                    x_column: x_column,
                                    y_column: y_column,
                                    'period': 'Time Period'
                                }
                            )
                    
                    # Update animation settings
                    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = animation_speed
                    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
                    
                    # Show the animation
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display animation controls instructions
                    st.info("**Animation Controls**: Use the play button at the bottom left to start/stop the animation. You can also drag the slider to move through time periods manually.")
    
    elif animation_type == "Category Transition":
        # Configure category transition animation
        st.markdown("##### Category Transition Animation Configuration")
        
        # Select category column
        if not categorical_columns:
            st.warning("Category Transition animation requires at least one categorical column. No categorical columns detected in your data.")
        else:
            category_column = st.selectbox("Category Column", categorical_columns, key="category_col_anim")
            
            # Select metrics
            x_column = st.selectbox("X-axis Column", numeric_columns, index=0, key="x_col_cat_anim")
            y_column = st.selectbox("Y-axis Column", numeric_columns, index=min(1, len(numeric_columns)-1), key="y_col_cat_anim")
            
            # Optional size column
            size_column = st.selectbox("Size By (optional)", ["None"] + numeric_columns, key="size_col_cat_anim")
            
            # Animation settings
            animation_speed = st.slider("Animation Speed", 500, 2000, 1000, 100, key="anim_speed_cat")
            
            # Generate animation
            if st.button("Generate Category Transition Animation", key="gen_cat_anim"):
                # Create the animated chart
                
                # Prepare data
                data = st.session_state.cleaned_data.copy()
                
                # Drop rows with missing values in selected columns
                required_columns = [category_column, x_column, y_column]
                if size_column != "None":
                    required_columns.append(size_column)
                
                data = data.dropna(subset=required_columns)
                
                # Check if we have enough data
                categories = data[category_column].unique()
                
                if len(categories) < 2:
                    st.error(f"Need at least 2 categories for animation. Only found {len(categories)} in '{category_column}'.")
                else:
                    # Create the scatter animation
                    if size_column != "None":
                        fig = px.scatter(
                            data,
                            x=x_column,
                            y=y_column,
                            size=size_column,
                            color=category_column,
                            animation_frame=category_column,
                            title=f"{y_column} vs {x_column} by {category_column}",
                            labels={
                                x_column: x_column,
                                y_column: y_column,
                                category_column: category_column,
                                size_column: size_column
                            }
                        )
                    else:
                        fig = px.scatter(
                            data,
                            x=x_column,
                            y=y_column,
                            color=category_column,
                            animation_frame=category_column,
                            title=f"{y_column} vs {x_column} by {category_column}",
                            labels={
                                x_column: x_column,
                                y_column: y_column,
                                category_column: category_column
                            }
                        )
                    
                    # Update animation settings
                    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = animation_speed
                    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
                    
                    # Show the animation
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display animation controls instructions
                    st.info("**Animation Controls**: Use the play button at the bottom left to start/stop the animation. You can also drag the slider to move through categories manually.")
    
    elif animation_type == "Cumulative Growth":
        # Configure cumulative growth animation
        st.markdown("##### Cumulative Growth Animation Configuration")
        
        # Select time or sequence column
        sequence_column = st.selectbox(
            "Sequence Column (for ordering)",
            ["Index"] + date_columns + numeric_columns,
            key="seq_col_growth_anim"
        )
        
        # Select metric to show growth
        value_column = st.selectbox("Value Column (to show growth)", numeric_columns, key="value_col_growth_anim")
        
        # Optional grouping column
        group_column = st.selectbox("Group By (optional)", ["None"] + categorical_columns, key="group_col_growth_anim")
        
        # Chart type
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Area Chart", "Bar Chart"],
            key="chart_type_growth_anim"
        )
        
        # Animation settings
        num_frames = st.slider("Number of Animation Frames", 10, 50, 20, key="num_frames_growth")
        animation_speed = st.slider("Animation Speed", 50, 500, 200, 10, key="anim_speed_growth")
        
        # Generate animation
        if st.button("Generate Cumulative Growth Animation", key="gen_growth_anim"):
            # Create the animated chart
            
            # Prepare data
            data = st.session_state.cleaned_data.copy()
            
            # Sort by sequence column if specified
            if sequence_column != "Index":
                if sequence_column in date_columns:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(data[sequence_column]):
                        data[sequence_column] = pd.to_datetime(data[sequence_column], errors='coerce')
                
                # Drop rows with missing sequence values
                data = data.dropna(subset=[sequence_column])
                
                # Sort by sequence column
                data = data.sort_values(sequence_column)
            else:
                # Sort by index
                data = data.sort_index()
            
            # Create frames for animation
            frames = []
            frame_indices = np.linspace(0, len(data), num_frames, endpoint=True, dtype=int)
            frame_indices = np.unique(frame_indices)  # Remove duplicates
            
            for i, idx in enumerate(frame_indices):
                if idx == 0:
                    continue  # Skip empty frame
                
                # Get subset of data up to this index
                subset = data.iloc[:idx].copy()
                
                # Add frame index for animation
                subset['frame'] = i
                
                frames.append(subset)
            
            # Combine frames
            if frames:
                animation_data = pd.concat(frames)
                
                # Create the animation based on chart type
                if chart_type == "Line Chart":
                    if group_column != "None":
                        fig = px.line(
                            animation_data,
                            x=sequence_column if sequence_column != "Index" else animation_data.index,
                            y=value_column,
                            color=group_column,
                            animation_frame='frame',
                            title=f"Cumulative Growth of {value_column}",
                            labels={
                                sequence_column: "Sequence" if sequence_column != "Index" else "Index",
                                value_column: value_column,
                                group_column: group_column
                            }
                        )
                    else:
                        fig = px.line(
                            animation_data,
                            x=sequence_column if sequence_column != "Index" else animation_data.index,
                            y=value_column,
                            animation_frame='frame',
                            title=f"Cumulative Growth of {value_column}",
                            labels={
                                sequence_column: "Sequence" if sequence_column != "Index" else "Index",
                                value_column: value_column
                            }
                        )
                
                elif chart_type == "Area Chart":
                    if group_column != "None":
                        fig = px.area(
                            animation_data,
                            x=sequence_column if sequence_column != "Index" else animation_data.index,
                            y=value_column,
                            color=group_column,
                            animation_frame='frame',
                            title=f"Cumulative Growth of {value_column}",
                            labels={
                                sequence_column: "Sequence" if sequence_column != "Index" else "Index",
                                value_column: value_column,
                                group_column: group_column
                            }
                        )
                    else:
                        fig = px.area(
                            animation_data,
                            x=sequence_column if sequence_column != "Index" else animation_data.index,
                            y=value_column,
                            animation_frame='frame',
                            title=f"Cumulative Growth of {value_column}",
                            labels={
                                sequence_column: "Sequence" if sequence_column != "Index" else "Index",
                                value_column: value_column
                            }
                        )
                
                else:  # Bar Chart
                    if group_column != "None":
                        fig = px.bar(
                            animation_data,
                            x=sequence_column if sequence_column != "Index" else animation_data.index,
                            y=value_column,
                            color=group_column,
                            animation_frame='frame',
                            title=f"Cumulative Growth of {value_column}",
                            labels={
                                sequence_column: "Sequence" if sequence_column != "Index" else "Index",
                                value_column: value_column,
                                group_column: group_column
                            }
                        )
                    else:
                        fig = px.bar(
                            animation_data,
                            x=sequence_column if sequence_column != "Index" else animation_data.index,
                            y=value_column,
                            animation_frame='frame',
                            title=f"Cumulative Growth of {value_column}",
                            labels={
                                sequence_column: "Sequence" if sequence_column != "Index" else "Index",
                                value_column: value_column
                            }
                        )
                
                # Update animation settings
                fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = animation_speed
                fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 100
                
                # Show the animation
                st.plotly_chart(fig, use_container_width=True)
                
                # Display animation controls instructions
                st.info("**Animation Controls**: Use the play button at the bottom left to start/stop the animation. You can also drag the slider to move through the growth sequence manually.")
            else:
                st.error("Could not create animation frames. Please check your data and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)