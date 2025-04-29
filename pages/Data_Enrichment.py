import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import json
import io
import os
import time
import re
import trafilatura
import functools
from datetime import datetime, timedelta
import requests
from io import StringIO
from urllib.parse import urlparse

# Page configuration
st.set_page_config(
    page_title="Data Enrichment",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title with styling
st.markdown('<h1 class="main-header">Data Enrichment</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enhance your dataset with external data from various sources</p>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using this page.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

# Define utility functions for lookups
def excel_vlookup(df, lookup_value, lookup_col, return_col, exact_match=True):
    """
    Implementation of Excel's VLOOKUP function
    
    Args:
        df (pd.DataFrame): The dataframe to search in
        lookup_value: The value to find in the lookup column
        lookup_col (str): The column to search in
        return_col (str): The column to return a value from
        exact_match (bool): Whether to require an exact match
        
    Returns:
        The matching value or None if not found
    """
    # For exact match
    if exact_match:
        try:
            # Find the matching row and return the value from return_col
            result = df[df[lookup_col] == lookup_value][return_col]
            return result.iloc[0] if not result.empty else None
        except:
            return None
    else:
        # For approximate match (like Excel's VLOOKUP with FALSE)
        try:
            # Sort the dataframe by the lookup column
            sorted_df = df.sort_values(by=lookup_col)
            
            # Find the largest value in lookup_col that is less than or equal to lookup_value
            mask = sorted_df[lookup_col] <= lookup_value
            if not mask.any():
                return None
            
            match_idx = mask.iloc[:-1].argmin() if mask.iloc[-1] else mask.argmin()
            return sorted_df.iloc[match_idx][return_col]
        except:
            return None

def excel_hlookup(df, lookup_value, lookup_row_idx, return_row_idx, exact_match=True):
    """
    Implementation of Excel's HLOOKUP function
    
    Args:
        df (pd.DataFrame): The dataframe to search in
        lookup_value: The value to find in the lookup row
        lookup_row_idx (int): The row index to search in (0-based)
        return_row_idx (int): The row index to return a value from
        exact_match (bool): Whether to require an exact match
        
    Returns:
        The matching value or None if not found
    """
    try:
        # Transpose the dataframe to use the same logic as VLOOKUP
        transposed = df.T
        return excel_vlookup(transposed, lookup_value, lookup_row_idx, return_row_idx, exact_match)
    except:
        return None

def excel_xlookup(df, lookup_value, lookup_array, return_array, if_not_found=None):
    """
    Implementation of Excel's XLOOKUP function
    
    Args:
        df (pd.DataFrame): The dataframe to search in
        lookup_value: The value to find
        lookup_array (str): The column to search in
        return_array (str): The column to return a value from
        if_not_found: Value to return if not found
        
    Returns:
        The matching value or if_not_found if no match
    """
    try:
        result = df[df[lookup_array] == lookup_value][return_array]
        return result.iloc[0] if not result.empty else if_not_found
    except:
        return if_not_found

def dax_lookupvalue(df, result_column, search_column, search_value, *additional_filters):
    """
    Implementation of DAX's LOOKUPVALUE function
    
    Args:
        df (pd.DataFrame): The dataframe to search in
        result_column (str): The column to return a value from
        search_column (str): The column to search in
        search_value: The value to find
        *additional_filters: Additional (column, value) pairs for filtering
        
    Returns:
        The matching value or None if not found
    """
    try:
        # Start with the basic filter
        mask = df[search_column] == search_value
        
        # Add additional filters
        for i in range(0, len(additional_filters), 2):
            if i+1 < len(additional_filters):
                filter_col = additional_filters[i]
                filter_val = additional_filters[i+1]
                mask = mask & (df[filter_col] == filter_val)
        
        # Get the filtered result
        result = df[mask][result_column]
        return result.iloc[0] if not result.empty else None
    except:
        return None

def apply_formula_to_column(df, formula_type, params):
    """
    Apply a formula to an entire column or create a new column
    
    Args:
        df (pd.DataFrame): The dataframe to modify
        formula_type (str): The type of formula to apply
        params (dict): Parameters for the formula
        
    Returns:
        pd.DataFrame: The modified dataframe
    """
    result_df = df.copy()
    
    if formula_type == "VLOOKUP":
        lookup_col = params.get("lookup_column")
        table_df = params.get("table_df")
        table_lookup_col = params.get("table_lookup_column")
        table_return_col = params.get("table_return_column")
        result_col = params.get("result_column")
        exact_match = params.get("exact_match", True)
        
        # Create a vectorized lookup function
        def lookup_func(x):
            return excel_vlookup(table_df, x, table_lookup_col, table_return_col, exact_match)
        
        # Apply the function to create the new column
        result_df[result_col] = result_df[lookup_col].apply(lookup_func)
        
    elif formula_type == "XLOOKUP":
        lookup_col = params.get("lookup_column")
        table_df = params.get("table_df")
        table_lookup_col = params.get("table_lookup_column")
        table_return_col = params.get("table_return_column")
        result_col = params.get("result_column")
        if_not_found = params.get("if_not_found")
        
        # Create a vectorized lookup function
        def lookup_func(x):
            return excel_xlookup(table_df, x, table_lookup_col, table_return_col, if_not_found)
        
        # Apply the function to create the new column
        result_df[result_col] = result_df[lookup_col].apply(lookup_func)
        
    elif formula_type == "DAX_LOOKUPVALUE":
        lookup_col = params.get("lookup_column")
        table_df = params.get("table_df")
        table_lookup_col = params.get("table_lookup_column")
        table_return_col = params.get("table_return_column")
        result_col = params.get("result_column")
        additional_filters = params.get("additional_filters", [])
        
        # Create a vectorized lookup function
        def lookup_func(x):
            return dax_lookupvalue(table_df, table_return_col, table_lookup_col, x, *additional_filters)
        
        # Apply the function to create the new column
        result_df[result_col] = result_df[lookup_col].apply(lookup_func)
    
    return result_df

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
    
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None

if "enriched_data" not in st.session_state:
    st.session_state.enriched_data = None

if "lookup_tables" not in st.session_state:
    st.session_state.lookup_tables = {}

# Check if data is available
if st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using enrichment features.")
else:
    # Create tabs for different enrichment features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Web Scraping Integration",
        "Geospatial Analysis",
        "Data Merging & Transformation",
        "Advanced Formulas & Lookups"
    ])
    
    with tab1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Web Scraping Integration")
        st.markdown("""
        Enhance your data by pulling relevant information from websites.
        Web scraping allows you to gather external data to complement your analysis.
        """)
        
        # Scraping configuration
        st.markdown("#### Web Scraping Configuration")
        
        scrape_method = st.radio(
            "Scraping Method",
            ["Single URL", "Multiple URLs from Dataset", "Search Engine Results"],
            horizontal=True
        )
        
        if scrape_method == "Single URL":
            # Single URL configuration
            url = st.text_input("Enter URL to Scrape", "https://example.com")
            
            # What to extract
            extract_option = st.selectbox(
                "What to Extract",
                ["Main Text Content", "Title and Text", "Structured Data (if available)", "Everything"]
            )
            
            # Add to dataset option
            add_as = st.radio(
                "How to Add to Dataset",
                ["New Column", "New Row"],
                horizontal=True
            )
            
            if add_as == "New Column":
                column_name = st.text_input("New Column Name", "scraped_content")
            
            # Scraping button
            if st.button("Scrape Website"):
                # Validate URL
                if not url.startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL starting with http:// or https://")
                else:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Scraping website...")
                    progress_bar.progress(30)
                    
                    try:
                        # Use trafilatura to extract content
                        downloaded = trafilatura.fetch_url(url)
                        
                        # Update progress
                        progress_bar.progress(60)
                        status_text.text("Processing content...")
                        
                        # Extract based on selected option
                        if extract_option == "Main Text Content":
                            content = trafilatura.extract(downloaded)
                        elif extract_option == "Title and Text":
                            content = trafilatura.extract(downloaded, include_comments=False, 
                                                         include_tables=False, include_links=False,
                                                         include_images=False)
                            title = trafilatura.extract_metadata(downloaded, output_format="python")
                            if title and 'title' in title:
                                content = f"Title: {title['title']}\n\n{content}"
                        elif extract_option == "Structured Data (if available)":
                            # Try to extract structured data
                            try:
                                metadata = trafilatura.extract_metadata(downloaded, output_format="python")
                                if metadata:
                                    content = json.dumps(metadata, indent=2)
                                else:
                                    content = "No structured data found. Extracting main content instead:\n\n"
                                    content += trafilatura.extract(downloaded)
                            except:
                                content = "Could not extract structured data. Extracting main content instead:\n\n"
                                content += trafilatura.extract(downloaded)
                        else:  # Everything
                            content = trafilatura.extract(downloaded, include_comments=True, 
                                                         include_tables=True, include_links=True,
                                                         include_images=True)
                            metadata = trafilatura.extract_metadata(downloaded, output_format="python")
                            if metadata:
                                content = f"Metadata:\n{json.dumps(metadata, indent=2)}\n\nContent:\n{content}"
                        
                        # Update progress
                        progress_bar.progress(90)
                        status_text.text("Adding to dataset...")
                        
                        # Add to dataset
                        if content:
                            if add_as == "New Column":
                                # Add as new column
                                df = st.session_state.cleaned_data.copy()
                                df[column_name] = content
                                st.session_state.enriched_data = df
                                
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Web scraping complete!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Show preview
                                st.success(f"Successfully scraped content from {url} and added as column '{column_name}'")
                                st.dataframe(df[column_name].head(3), use_container_width=True)
                                
                                # Show full content
                                with st.expander("View Full Scraped Content"):
                                    st.text_area("Content", content, height=300)
                            else:
                                # Add as new row
                                df = st.session_state.cleaned_data.copy()
                                
                                # Create new row (use existing columns if possible)
                                new_row = {}
                                
                                # Try to extract structured data for mapping to columns
                                metadata = None
                                try:
                                    metadata = trafilatura.extract_metadata(downloaded, output_format="python")
                                except:
                                    pass
                                
                                # Populate new row
                                for col in df.columns:
                                    # Try to map common column names to extracted data
                                    col_lower = col.lower()
                                    
                                    if metadata:
                                        if 'title' in metadata and ('title' in col_lower or 'name' in col_lower):
                                            new_row[col] = metadata['title']
                                        elif 'author' in metadata and 'author' in col_lower:
                                            new_row[col] = metadata['author']
                                        elif 'date' in metadata and ('date' in col_lower or 'time' in col_lower):
                                            new_row[col] = metadata['date']
                                        elif 'url' in metadata and ('url' in col_lower or 'link' in col_lower):
                                            new_row[col] = metadata['url']
                                        elif 'description' in metadata and ('description' in col_lower or 'summary' in col_lower):
                                            new_row[col] = metadata['description']
                                        else:
                                            # Add content to columns with text, content, or data in the name
                                            if 'text' in col_lower or 'content' in col_lower or 'data' in col_lower:
                                                new_row[col] = content[:1000]  # Limit to 1000 chars
                                            else:
                                                new_row[col] = None
                                    else:
                                        # Without metadata, just add content to text-related columns
                                        if 'text' in col_lower or 'content' in col_lower or 'data' in col_lower:
                                            new_row[col] = content[:1000]  # Limit to 1000 chars
                                        elif 'url' in col_lower or 'link' in col_lower:
                                            new_row[col] = url
                                        elif 'date' in col_lower or 'time' in col_lower:
                                            new_row[col] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        else:
                                            new_row[col] = None
                                
                                # Add new row to dataframe
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                                st.session_state.enriched_data = df
                                
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Web scraping complete!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Show preview
                                st.success(f"Successfully scraped content from {url} and added as new row")
                                st.dataframe(df.tail(1), use_container_width=True)
                        else:
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("No content found!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.error(f"No content could be extracted from {url}")
                    
                    except Exception as e:
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Error during scraping!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.error(f"Error scraping website: {str(e)}")
        
        elif scrape_method == "Multiple URLs from Dataset":
            # Multiple URLs from dataset
            
            # URL column selection
            columns = st.session_state.cleaned_data.columns.tolist()
            url_column = st.selectbox(
                "Select Column Containing URLs",
                columns,
                help="Choose the column that contains URLs to scrape"
            )
            
            # Check if column contains URLs
            if url_column and st.session_state.cleaned_data[url_column].dtype == 'object':
                sample_urls = []
                for val in st.session_state.cleaned_data[url_column].dropna().sample(min(5, len(st.session_state.cleaned_data))):
                    if isinstance(val, str) and val.startswith(('http://', 'https://')):
                        sample_urls.append(val)
                
                if sample_urls:
                    st.info(f"Sample URLs found: {', '.join(sample_urls[:3])}")
                else:
                    st.warning(f"Column '{url_column}' may not contain valid URLs. Please check the data.")
            
            # What to extract
            extract_option = st.selectbox(
                "What to Extract",
                ["Main Text Content", "Title and Text", "Structured Data (if available)"],
                key="extract_option_multi"
            )
            
            # Destination column
            dest_column = st.text_input("Destination Column Name", "scraped_content")
            
            # Sample size
            if url_column:
                max_urls = len(st.session_state.cleaned_data[url_column].dropna())
                sample_size = st.slider(
                    "Number of URLs to Scrape",
                    min_value=1,
                    max_value=min(max_urls, 50),  # Limit to 50 to avoid overload
                    value=min(max_urls, 5)
                )
            
            # Scraping button
            if st.button("Scrape Multiple Websites"):
                if not url_column:
                    st.error("Please select a column containing URLs")
                else:
                    # Get URLs to scrape
                    urls = st.session_state.cleaned_data[url_column].dropna()
                    urls = urls[urls.astype(str).str.startswith(('http://', 'https://'))].head(sample_size)
                    
                    if len(urls) == 0:
                        st.error(f"No valid URLs found in column '{url_column}'")
                    else:
                        # Create progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Prepare result dataframe
                        df = st.session_state.cleaned_data.copy()
                        df[dest_column] = None
                        
                        # Track success count
                        success_count = 0
                        
                        # Scrape each URL
                        for i, (idx, url) in enumerate(urls.items()):
                            # Update progress
                            progress = int((i / len(urls)) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Scraping URL {i+1}/{len(urls)}: {url[:30]}...")
                            
                            try:
                                # Use trafilatura to extract content
                                downloaded = trafilatura.fetch_url(url)
                                
                                # Extract based on selected option
                                if extract_option == "Main Text Content":
                                    content = trafilatura.extract(downloaded)
                                elif extract_option == "Title and Text":
                                    content = trafilatura.extract(downloaded, include_comments=False, 
                                                                include_tables=False, include_links=False,
                                                                include_images=False)
                                    title = trafilatura.extract_metadata(downloaded, output_format="python")
                                    if title and 'title' in title:
                                        content = f"Title: {title['title']}\n\n{content}"
                                else:  # Structured Data
                                    try:
                                        metadata = trafilatura.extract_metadata(downloaded, output_format="python")
                                        if metadata:
                                            content = json.dumps(metadata, indent=2)
                                        else:
                                            content = trafilatura.extract(downloaded)
                                    except:
                                        content = trafilatura.extract(downloaded)
                                
                                # Add content to dataframe
                                if content:
                                    df.loc[idx, dest_column] = content
                                    success_count += 1
                                
                                # Delay to avoid overloading servers
                                time.sleep(0.5)
                            
                            except Exception as e:
                                df.loc[idx, dest_column] = f"Error: {str(e)}"
                        
                        # Update session state with enriched data
                        st.session_state.enriched_data = df
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Web scraping complete!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        st.success(f"Successfully scraped {success_count} out of {len(urls)} URLs")
                        st.dataframe(df[[url_column, dest_column]].head(10), use_container_width=True)
        
        else:  # Search Engine Results
            st.info("This feature simulates search results. It does not perform actual web searches, which would require external APIs.")
            
            # Search query
            search_query = st.text_input("Search Query", "data analytics best practices")
            
            # Number of results
            num_results = st.slider("Number of Results", 1, 10, 5)
            
            # Add to data options
            add_method = st.radio(
                "How to Add Results",
                ["New Rows", "New Dataset"],
                horizontal=True
            )
            
            # Search button
            if st.button("Simulate Search Results"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                status_text.text("Generating simulated search results...")
                progress_bar.progress(50)
                
                # Simulate search results
                # Note: This is a simulation using predefined templates
                results = []
                
                search_terms = search_query.split()
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # Create simulated results
                for i in range(num_results):
                    # Randomize some elements
                    authority = np.random.choice(["high", "medium", "low"])
                    days_ago = np.random.randint(1, 365)
                    pub_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                    
                    # Create a result
                    result = {
                        "title": f"{''.join(np.random.choice(search_terms, min(len(search_terms), 3)))} Research and Analysis",
                        "url": f"https://example.com/research/{i+1}/{''.join(np.random.choice(search_terms, 1))}",
                        "source": np.random.choice(["Research Journal", "Analytics Blog", "University Publication", "Industry Report"]),
                        "publish_date": pub_date,
                        "retrieval_date": current_date,
                        "authority_score": np.random.randint(1, 10),
                        "relevance_score": np.random.uniform(0.5, 1.0),
                        "summary": f"This article discusses {search_query} with a focus on recent developments and practical applications.",
                        "query": search_query
                    }
                    
                    results.append(result)
                
                # Update progress
                progress_bar.progress(80)
                status_text.text("Processing results...")
                
                # Create a DataFrame from results
                results_df = pd.DataFrame(results)
                
                if add_method == "New Rows":
                    # Add as new rows to existing data
                    df = st.session_state.cleaned_data.copy()
                    
                    # Map columns if possible
                    mapped_results = []
                    for result in results:
                        new_row = {}
                        
                        for col in df.columns:
                            col_lower = col.lower()
                            
                            if 'title' in col_lower:
                                new_row[col] = result['title']
                            elif 'url' in col_lower or 'link' in col_lower:
                                new_row[col] = result['url']
                            elif 'date' in col_lower:
                                new_row[col] = result['publish_date']
                            elif 'source' in col_lower:
                                new_row[col] = result['source']
                            elif 'summary' in col_lower or 'description' in col_lower:
                                new_row[col] = result['summary']
                            elif 'score' in col_lower or 'rating' in col_lower:
                                new_row[col] = result['relevance_score']
                            elif 'query' in col_lower or 'search' in col_lower:
                                new_row[col] = result['query']
                            else:
                                new_row[col] = None
                        
                        mapped_results.append(new_row)
                    
                    # Add new rows to dataframe
                    if mapped_results:
                        new_rows_df = pd.DataFrame(mapped_results)
                        df = pd.concat([df, new_rows_df], ignore_index=True)
                        st.session_state.enriched_data = df
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Search simulation complete!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    st.success(f"Added {len(results)} simulated search results as new rows")
                    st.dataframe(df.tail(len(results)), use_container_width=True)
                
                else:  # New Dataset
                    # Store as a new dataset
                    st.session_state.enriched_data = results_df
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Search simulation complete!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    st.success(f"Created new dataset with {len(results)} simulated search results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Option to download
                    csv = results_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="search_results.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        # Save enriched data
        if st.session_state.enriched_data is not None:
            st.markdown("#### Save Enriched Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply Changes to Dataset", key="apply_enriched_data_1"):
                    st.session_state.cleaned_data = st.session_state.enriched_data
                    st.success("Enriched data has been applied as the current dataset")
            
            with col2:
                if st.button("Discard Changes", key="discard_enriched_data_1"):
                    st.session_state.enriched_data = None
                    st.info("Enriched data has been discarded")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Geospatial Analysis")
        st.markdown("""
        Add geographical context to your data by mapping addresses, coordinates, or region names.
        Visualize your data on maps to identify geographical patterns and insights.
        """)
        
        # Geospatial configuration
        st.markdown("#### Geospatial Configuration")
        
        # Check for potential location columns
        df = st.session_state.cleaned_data
        
        # Location columns detection
        location_columns = []
        if df is not None:
            for col in df.columns:
                col_lower = col.lower()
                # Check for common location-related column names
                if any(term in col_lower for term in ['country', 'city', 'state', 'region', 'province', 'address', 'location', 'postal', 'zip', 'county', 'latitude', 'longitude', 'lat', 'lon', 'lng']):
                    location_columns.append(col)
        
        if not location_columns:
            st.warning("No location-related columns detected in your data. Location columns typically include 'country', 'city', 'address', 'postal', 'latitude', 'longitude', etc.")
            
            # Manual column selection
            all_columns = df.columns.tolist() if df is not None else []
            location_column = st.selectbox(
                "Select a Column Containing Location Information",
                all_columns if all_columns else ["No columns available"]
            )
            
            # Mapping type
            map_type = st.radio(
                "Map Type",
                ["Basic Plot", "Choropleth (Region Map)", "Heatmap"],
                horizontal=True
            )
            
            st.info("For best results with geospatial analysis, your data should contain location information like addresses, coordinates, or region names.")
        
        else:
            # Detected location columns
            st.success(f"Detected potential location columns: {', '.join(location_columns)}")
            
            # Select primary location column
            primary_location = st.selectbox(
                "Primary Location Column",
                location_columns
            )
            
            # Determine location type
            location_type = st.radio(
                "Location Type",
                ["Coordinates (Latitude/Longitude)", "Region Names (Countries, States, etc.)", "Addresses (Street, City, etc.)"],
                horizontal=True
            )
            
            if location_type == "Coordinates (Latitude/Longitude)":
                # Detect lat/lon columns
                lat_columns = [col for col in df.columns if any(term in col.lower() for term in ['latitude', 'lat'])]
                lon_columns = [col for col in df.columns if any(term in col.lower() for term in ['longitude', 'lon', 'lng'])]
                
                if lat_columns and lon_columns:
                    lat_column = st.selectbox("Latitude Column", lat_columns)
                    lon_column = st.selectbox("Longitude Column", lon_columns)
                else:
                    st.warning("Could not automatically detect latitude and longitude columns.")
                    lat_column = st.selectbox("Latitude Column", df.columns.tolist())
                    lon_column = st.selectbox("Longitude Column", df.columns.tolist())
                
                # Value column for coloring
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numeric_columns:
                    value_column = st.selectbox("Value Column (for coloring)", ["None"] + numeric_columns)
                else:
                    value_column = "None"
                
                # Map type
                map_type = st.radio(
                    "Map Type",
                    ["Scatter Map", "Bubble Map", "Heatmap"],
                    horizontal=True
                )
                
                # Create map button
                if st.button("Create Coordinate Map"):
                    # Validate columns
                    if lat_column == lon_column:
                        st.error("Latitude and longitude columns must be different")
                    else:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("Processing coordinates...")
                        progress_bar.progress(30)
                        
                        try:
                            # Prepare mapping data
                            map_data = df.copy()
                            
                            # Filter out invalid coordinates
                            valid_coords = (map_data[lat_column].notna() & 
                                          map_data[lon_column].notna() &
                                          (map_data[lat_column].astype(float) >= -90) &
                                          (map_data[lat_column].astype(float) <= 90) &
                                          (map_data[lon_column].astype(float) >= -180) &
                                          (map_data[lon_column].astype(float) <= 180))
                            
                            map_data = map_data[valid_coords]
                            
                            if len(map_data) == 0:
                                raise ValueError("No valid coordinates found in selected columns")
                            
                            # Update progress
                            status_text.text("Creating map visualization...")
                            progress_bar.progress(60)
                            
                            # Create map based on type
                            if map_type == "Scatter Map":
                                if value_column != "None":
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat=lat_column,
                                        lon=lon_column,
                                        color=value_column,
                                        hover_name=primary_location if primary_location in map_data.columns else None,
                                        color_continuous_scale="Viridis",
                                        zoom=1
                                    )
                                else:
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat=lat_column,
                                        lon=lon_column,
                                        hover_name=primary_location if primary_location in map_data.columns else None,
                                        zoom=1
                                    )
                            
                            elif map_type == "Bubble Map":
                                if value_column != "None":
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat=lat_column,
                                        lon=lon_column,
                                        size=value_column,
                                        color=value_column,
                                        hover_name=primary_location if primary_location in map_data.columns else None,
                                        color_continuous_scale="Viridis",
                                        zoom=1
                                    )
                                else:
                                    fig = px.scatter_mapbox(
                                        map_data,
                                        lat=lat_column,
                                        lon=lon_column,
                                        hover_name=primary_location if primary_location in map_data.columns else None,
                                        zoom=1
                                    )
                            
                            else:  # Heatmap
                                if value_column != "None":
                                    fig = px.density_mapbox(
                                        map_data,
                                        lat=lat_column,
                                        lon=lon_column,
                                        z=value_column,
                                        radius=10,
                                        zoom=1
                                    )
                                else:
                                    fig = px.density_mapbox(
                                        map_data,
                                        lat=lat_column,
                                        lon=lon_column,
                                        radius=10,
                                        zoom=1
                                    )
                            
                            # Use OpenStreetMap
                            fig.update_layout(mapbox_style="open-street-map")
                            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                            
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Map created!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Show the map
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics about the mapping
                            st.markdown("#### Mapping Statistics")
                            st.markdown(f"• **Mapped points:** {len(map_data)} out of {len(df)} ({len(map_data)/len(df)*100:.1f}%)")
                            
                            if value_column != "None":
                                st.markdown(f"• **Value column range:** {map_data[value_column].min():.2f} to {map_data[value_column].max():.2f}")
                                st.markdown(f"• **Value column average:** {map_data[value_column].mean():.2f}")
                            
                            # Top locations
                            if primary_location in map_data.columns:
                                top_locations = map_data[primary_location].value_counts().head(5)
                                st.markdown("#### Top Locations")
                                
                                for loc, count in top_locations.items():
                                    st.markdown(f"• **{loc}:** {count} points")
                        
                        except Exception as e:
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Error creating map!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.error(f"Error creating map: {str(e)}")
                            st.info("Make sure your coordinate columns contain valid latitude and longitude values.")
            
            elif location_type == "Region Names (Countries, States, etc.)":
                # Region mapping
                region_level = st.selectbox(
                    "Region Level",
                    ["Country", "US State", "World Cities"]
                )
                
                # Value column for choropleth
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numeric_columns:
                    value_column = st.selectbox("Value Column (for coloring)", numeric_columns)
                    
                    # Aggregation method
                    agg_method = st.selectbox(
                        "Aggregation Method",
                        ["Mean", "Sum", "Count", "Min", "Max"]
                    )
                else:
                    st.warning("Choropleth maps require at least one numeric column for coloring.")
                    value_column = None
                
                # Create map button
                if st.button("Create Region Map"):
                    if not value_column:
                        st.error("Please select a numeric value column for the map")
                    else:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("Processing region data...")
                        progress_bar.progress(30)
                        
                        try:
                            # Prepare mapping data
                            map_data = df.copy()
                            
                            # Validate value column is numeric
                            if not pd.api.types.is_numeric_dtype(map_data[value_column]):
                                # Try to convert to numeric, coerce errors to NaN
                                map_data[value_column] = pd.to_numeric(map_data[value_column], errors='coerce')
                                # Drop rows with NaN values after conversion
                                map_data = map_data.dropna(subset=[value_column])
                                
                                if len(map_data) == 0:
                                    raise ValueError(f"Could not convert '{value_column}' to numeric values. Please select a numeric column.")
                                
                                st.info(f"Converted '{value_column}' to numeric data. Some non-numeric values were removed.")
                            
                            # Derive ISO codes for countries if not present
                            if region_level == "Country":
                                # Check if the column already contains ISO codes
                                unique_values = map_data[primary_location].dropna().unique()
                                if len(unique_values) > 0 and all(len(str(val)) == 2 or len(str(val)) == 3 for val in unique_values):
                                    iso_column = primary_location
                                else:
                                    # Convert country names to ISO codes (simplified example)
                                    # In a real implementation, you would use a proper country code library
                                    iso_column = f"{primary_location}_iso"
                                    map_data[iso_column] = map_data[primary_location]
                            else:
                                iso_column = primary_location
                            
                            # Aggregate data by region
                            if agg_method == "Mean":
                                agg_data = map_data.groupby(iso_column)[value_column].mean().reset_index()
                            elif agg_method == "Sum":
                                agg_data = map_data.groupby(iso_column)[value_column].sum().reset_index()
                            elif agg_method == "Count":
                                agg_data = map_data.groupby(iso_column)[value_column].count().reset_index()
                            elif agg_method == "Min":
                                agg_data = map_data.groupby(iso_column)[value_column].min().reset_index()
                            else:  # Max
                                agg_data = map_data.groupby(iso_column)[value_column].max().reset_index()
                            
                            # Update progress
                            status_text.text("Creating choropleth map...")
                            progress_bar.progress(60)
                            
                            # Create choropleth map
                            if region_level == "Country":
                                fig = px.choropleth(
                                    agg_data,
                                    locations=iso_column,
                                    color=value_column,
                                    hover_name=iso_column,
                                    color_continuous_scale="Viridis",
                                    locationmode='ISO-3'
                                )
                            elif region_level == "US State":
                                fig = px.choropleth(
                                    agg_data,
                                    locations=iso_column,
                                    color=value_column,
                                    hover_name=iso_column,
                                    locationmode='USA-states'
                                )
                            else:  # World Cities - use scatter geo instead
                                fig = px.scatter_geo(
                                    map_data,
                                    locations=primary_location,
                                    color=value_column,
                                    hover_name=primary_location,
                                    locationmode='country names'
                                )
                            
                            # Update layout
                            fig.update_layout(
                                margin={"r":0,"t":0,"l":0,"b":0},
                                height=600
                            )
                            
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Map created!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Show the map
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics about the mapping
                            st.markdown("#### Mapping Statistics")
                            st.markdown(f"• **Regions mapped:** {len(agg_data)}")
                            
                            # Top regions
                            top_regions = agg_data.sort_values(value_column, ascending=False).head(5)
                            st.markdown("#### Top Regions")
                            
                            for _, row in top_regions.iterrows():
                                st.markdown(f"• **{row[iso_column]}:** {row[value_column]:.2f}")
                        
                        except Exception as e:
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Error creating map!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.error(f"Error creating choropleth map: {str(e)}")
                
            else:  # Addresses
                st.info("Address geocoding requires an external API key for production use. This demo uses a simplified implementation.")
                
                # Sample for geocoding
                max_addresses = min(20, len(df))
                num_addresses = st.slider("Number of Addresses to Process", 1, max_addresses, 5)
                
                # Select columns for address components
                address_column = st.selectbox("Full Address Column (if available)", ["None"] + df.columns.tolist())
                
                if address_column == "None":
                    # Individual address components
                    st.markdown("#### Select Address Components")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        street_column = st.selectbox("Street Column (optional)", ["None"] + df.columns.tolist())
                        city_column = st.selectbox("City Column (optional)", ["None"] + df.columns.tolist())
                    
                    with col2:
                        state_column = st.selectbox("State/Region Column (optional)", ["None"] + df.columns.tolist())
                        country_column = st.selectbox("Country Column (optional)", ["None"] + df.columns.tolist())
                
                # Value column for visualization
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numeric_columns:
                    value_column = st.selectbox("Value Column (for coloring)", ["None"] + numeric_columns)
                else:
                    value_column = "None"
                
                # Geocode button
                if st.button("Process Addresses"):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Processing addresses...")
                    progress_bar.progress(20)
                    
                    try:
                        # Sample addresses
                        df_sample = df.sample(num_addresses, random_state=42)
                        
                        # Create geocoding results
                        geocoded_data = df_sample.copy()
                        
                        # Add latitude and longitude columns
                        geocoded_data['latitude'] = np.random.uniform(25, 50, len(geocoded_data))
                        geocoded_data['longitude'] = np.random.uniform(-120, -70, len(geocoded_data))
                        
                        # Update progress
                        status_text.text("Creating map visualization...")
                        progress_bar.progress(70)
                        
                        # Create map visualization
                        if value_column != "None":
                            fig = px.scatter_mapbox(
                                geocoded_data,
                                lat='latitude',
                                lon='longitude',
                                color=value_column,
                                hover_name=primary_location if primary_location in geocoded_data.columns else None,
                                color_continuous_scale="Viridis",
                                zoom=3
                            )
                        else:
                            fig = px.scatter_mapbox(
                                geocoded_data,
                                lat='latitude',
                                lon='longitude',
                                hover_name=primary_location if primary_location in geocoded_data.columns else None,
                                zoom=3
                            )
                        
                        # Use OpenStreetMap
                        fig.update_layout(mapbox_style="open-street-map")
                        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Address processing complete!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show notes about the simulation
                        st.warning("Note: This demonstration uses simulated coordinates. In a production application, you would use a geocoding service with an API key.")
                        
                        # Show the map
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to add coordinates to the dataset
                        if st.checkbox("Add these coordinates to your dataset"):
                            # Merge coordinates back to the main dataset
                            result_data = df.copy()
                            
                            # Add latitude and longitude columns
                            for idx, row in geocoded_data.iterrows():
                                result_data.loc[idx, 'latitude'] = row['latitude']
                                result_data.loc[idx, 'longitude'] = row['longitude']
                            
                            # Update the session state
                            st.session_state.enriched_data = result_data
                            
                            # Show message
                            st.success("Coordinates added to dataset")
                    
                    except Exception as e:
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Error processing addresses!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.error(f"Error processing addresses: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Data Merging & Transformation")
        st.markdown("""
        Combine your dataset with external data sources or transform your data to enhance analysis.
        Create new derived columns, pivot your data, or join with reference datasets.
        """)
        
        # Choose operation
        operation = st.radio(
            "Operation Type",
            ["Add Calculated Columns", "Join with External Data", "Pivot Table", "Data Transformation"],
            horizontal=True
        )
        
        if operation == "Add Calculated Columns":
            # Add calculated columns
            st.markdown("#### Add Calculated Columns")
            st.markdown("Create new columns based on calculations or transformations of existing columns.")
            
            # Check if data is available
            if st.session_state.cleaned_data is None:
                st.warning("Please upload data in the main dashboard before using this feature.")
                st.stop()
                
            # Get existing columns
            columns = st.session_state.cleaned_data.columns.tolist()
            numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Column creation options
            creation_method = st.selectbox(
                "Creation Method",
                ["Basic Arithmetic", "Text Manipulation", "Date Calculation", "Mapping/Categorization"]
            )
            
            if creation_method == "Basic Arithmetic":
                # Basic arithmetic operations
                st.markdown("#### Basic Arithmetic Column")
                
                # Column selection
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    column1 = st.selectbox("First Column", numeric_columns)
                
                with col2:
                    operation = st.selectbox("Operation", ["+", "-", "*", "/", "^", "% (Mod)"])
                
                with col3:
                    column2_type = st.radio("Second Value Type", ["Column", "Value"], horizontal=True)
                    
                    if column2_type == "Column":
                        column2 = st.selectbox("Second Column", numeric_columns)
                    else:
                        column2_value = st.number_input("Value", value=0.0)
                
                # New column name
                new_column = st.text_input("New Column Name", "calculated_column")
                
                # Create column button
                if st.button("Add Arithmetic Column"):
                    if not new_column:
                        st.error("Please enter a name for the new column")
                    else:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("Calculating new column...")
                        progress_bar.progress(50)
                        
                        try:
                            # Create new dataframe
                            result_data = st.session_state.cleaned_data.copy()
                            
                            # Get the second operand
                            if column2_type == "Column":
                                operand2 = result_data[column2]
                            else:
                                operand2 = column2_value
                            
                            # Perform calculation
                            if operation == "+":
                                result_data[new_column] = result_data[column1] + operand2
                            elif operation == "-":
                                result_data[new_column] = result_data[column1] - operand2
                            elif operation == "*":
                                result_data[new_column] = result_data[column1] * operand2
                            elif operation == "/":
                                # Handle division by zero
                                if column2_type == "Value" and operand2 == 0:
                                    raise ValueError("Cannot divide by zero")
                                
                                result_data[new_column] = result_data[column1] / operand2
                            elif operation == "^":
                                result_data[new_column] = result_data[column1] ** operand2
                            else:  # % (Mod)
                                # Handle mod by zero
                                if column2_type == "Value" and operand2 == 0:
                                    raise ValueError("Cannot calculate modulo by zero")
                                
                                result_data[new_column] = result_data[column1] % operand2
                            
                            # Update session state
                            st.session_state.enriched_data = result_data
                            
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Column added!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Show preview
                            st.success(f"Added new column '{new_column}' with arithmetic calculation")
                            st.dataframe(result_data[[column1, new_column]].head(), use_container_width=True)
                            
                            # Statistics
                            st.markdown("#### Column Statistics")
                            
                            stats = {
                                "Min": result_data[new_column].min(),
                                "Max": result_data[new_column].max(),
                                "Mean": result_data[new_column].mean(),
                                "Median": result_data[new_column].median(),
                                "Std Dev": result_data[new_column].std()
                            }
                            
                            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=["Value"])
                            st.dataframe(stats_df, use_container_width=True)
                        
                        except Exception as e:
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Error creating column!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.error(f"Error creating calculated column: {str(e)}")
            
            elif creation_method == "Text Manipulation":
                # Text manipulation operations
                st.markdown("#### Text Manipulation Column")
                
                # Get text columns
                text_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
                
                if not text_columns:
                    st.warning("No text columns found in your data")
                else:
                    # Column selection
                    column1 = st.selectbox("Text Column", text_columns)
                    
                    # Operation
                    text_operation = st.selectbox(
                        "Text Operation",
                        ["Uppercase", "Lowercase", "Title Case", "Extract Substring", "Concatenate with Another Column", "Replace Text", "Length"]
                    )
                    
                    # Operation-specific inputs
                    if text_operation == "Extract Substring":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            start_pos = st.number_input("Start Position", min_value=0, value=0)
                        
                        with col2:
                            end_pos = st.number_input("End Position (leave at -1 for end of string)", value=-1)
                    
                    elif text_operation == "Concatenate with Another Column":
                        second_column = st.selectbox("Second Column", columns)
                        separator = st.text_input("Separator", " ")
                    
                    elif text_operation == "Replace Text":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            find_text = st.text_input("Find Text", "old")
                        
                        with col2:
                            replace_text = st.text_input("Replace With", "new")
                    
                    # New column name
                    new_column = st.text_input("New Column Name", "text_column")
                    
                    # Create column button
                    if st.button("Add Text Column"):
                        if not new_column:
                            st.error("Please enter a name for the new column")
                        else:
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update progress
                            status_text.text("Processing text data...")
                            progress_bar.progress(50)
                            
                            try:
                                # Create new dataframe
                                result_data = st.session_state.cleaned_data.copy()
                                
                                # Perform text operation
                                if text_operation == "Uppercase":
                                    result_data[new_column] = result_data[column1].astype(str).str.upper()
                                elif text_operation == "Lowercase":
                                    result_data[new_column] = result_data[column1].astype(str).str.lower()
                                elif text_operation == "Title Case":
                                    result_data[new_column] = result_data[column1].astype(str).str.title()
                                elif text_operation == "Extract Substring":
                                    if end_pos == -1:
                                        result_data[new_column] = result_data[column1].astype(str).str[start_pos:]
                                    else:
                                        result_data[new_column] = result_data[column1].astype(str).str[start_pos:end_pos]
                                elif text_operation == "Concatenate with Another Column":
                                    result_data[new_column] = result_data[column1].astype(str) + separator + result_data[second_column].astype(str)
                                elif text_operation == "Replace Text":
                                    result_data[new_column] = result_data[column1].astype(str).str.replace(find_text, replace_text, regex=False)
                                else:  # Length
                                    result_data[new_column] = result_data[column1].astype(str).str.len()
                                
                                # Update session state
                                st.session_state.enriched_data = result_data
                                
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Column added!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Show preview
                                st.success(f"Added new column '{new_column}' with text manipulation")
                                st.dataframe(result_data[[column1, new_column]].head(), use_container_width=True)
                            
                            except Exception as e:
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Error creating column!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.error(f"Error creating text column: {str(e)}")
            
            elif creation_method == "Date Calculation":
                # Date calculations
                st.markdown("#### Date Calculation Column")
                
                # Detect date columns
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
                
                if not date_columns:
                    st.warning("No date columns detected in your data")
                else:
                    # Column selection
                    date_column = st.selectbox("Date Column", date_columns)
                    
                    # Operation
                    date_operation = st.selectbox(
                        "Date Operation",
                        ["Extract Year", "Extract Month", "Extract Day", "Extract Weekday", "Date Difference", "Add/Subtract Period"]
                    )
                    
                    # Operation-specific inputs
                    if date_operation == "Date Difference":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            second_date = st.selectbox("Second Date Column", date_columns)
                        
                        with col2:
                            diff_unit = st.selectbox("Difference Unit", ["Days", "Months", "Years"])
                    
                    elif date_operation == "Add/Subtract Period":
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            operation = st.radio("Operation", ["Add", "Subtract"], horizontal=True)
                        
                        with col2:
                            amount = st.number_input("Amount", value=1)
                        
                        with col3:
                            unit = st.selectbox("Unit", ["Days", "Weeks", "Months", "Years"])
                    
                    # New column name
                    new_column = st.text_input("New Column Name", "date_column")
                    
                    # Create column button
                    if st.button("Add Date Column"):
                        if not new_column:
                            st.error("Please enter a name for the new column")
                        else:
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update progress
                            status_text.text("Processing date data...")
                            progress_bar.progress(30)
                            
                            try:
                                # Create new dataframe
                                result_data = st.session_state.cleaned_data.copy()
                                
                                # Ensure date column is datetime
                                if not pd.api.types.is_datetime64_any_dtype(result_data[date_column]):
                                    result_data[date_column] = pd.to_datetime(result_data[date_column], errors='coerce')
                                
                                # Update progress
                                status_text.text("Calculating date transformation...")
                                progress_bar.progress(60)
                                
                                # Perform date operation
                                if date_operation == "Extract Year":
                                    result_data[new_column] = result_data[date_column].dt.year
                                elif date_operation == "Extract Month":
                                    result_data[new_column] = result_data[date_column].dt.month
                                elif date_operation == "Extract Day":
                                    result_data[new_column] = result_data[date_column].dt.day
                                elif date_operation == "Extract Weekday":
                                    result_data[new_column] = result_data[date_column].dt.day_name()
                                elif date_operation == "Date Difference":
                                    # Ensure second date column is datetime
                                    if not pd.api.types.is_datetime64_any_dtype(result_data[second_date]):
                                        result_data[second_date] = pd.to_datetime(result_data[second_date], errors='coerce')
                                    
                                    if diff_unit == "Days":
                                        result_data[new_column] = (result_data[date_column] - result_data[second_date]).dt.days
                                    elif diff_unit == "Months":
                                        result_data[new_column] = ((result_data[date_column].dt.year - result_data[second_date].dt.year) * 12 + 
                                                                  (result_data[date_column].dt.month - result_data[second_date].dt.month))
                                    else:  # Years
                                        result_data[new_column] = (result_data[date_column].dt.year - result_data[second_date].dt.year)
                                else:  # Add/Subtract Period
                                    if unit == "Days":
                                        delta = pd.Timedelta(days=amount)
                                    elif unit == "Weeks":
                                        delta = pd.Timedelta(weeks=amount)
                                    elif unit == "Months":
                                        delta = pd.DateOffset(months=amount)
                                    else:  # Years
                                        delta = pd.DateOffset(years=amount)
                                    
                                    if operation == "Add":
                                        result_data[new_column] = result_data[date_column] + delta
                                    else:  # Subtract
                                        result_data[new_column] = result_data[date_column] - delta
                                
                                # Update session state
                                st.session_state.enriched_data = result_data
                                
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Column added!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Show preview
                                st.success(f"Added new column '{new_column}' with date calculation")
                                st.dataframe(result_data[[date_column, new_column]].head(), use_container_width=True)
                            
                            except Exception as e:
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Error creating column!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.error(f"Error creating date column: {str(e)}")
            
            else:  # Mapping/Categorization
                # Mapping/Categorization operations
                st.markdown("#### Mapping/Categorization Column")
                
                # Column selection
                column1 = st.selectbox("Source Column", columns)
                
                # Mapping method
                mapping_method = st.radio(
                    "Mapping Method",
                    ["Value Ranges (Binning)", "Custom Mapping", "Conditional Logic"],
                    horizontal=True
                )
                
                if mapping_method == "Value Ranges (Binning)":
                    # Check if column is numeric
                    is_numeric = st.session_state.cleaned_data[column1].dtype in ['int64', 'float64']
                    
                    if not is_numeric:
                        st.warning(f"Column '{column1}' is not numeric. Binning works best with numeric columns.")
                    
                    # Bin configuration
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        num_bins = st.slider("Number of Bins", min_value=2, max_value=10, value=4)
                    
                    with col2:
                        bin_labels = st.text_input("Bin Labels (comma-separated)", "Low,Medium,High,Very High")
                    
                    # Get bin labels
                    labels = [label.strip() for label in bin_labels.split(',')]
                    if len(labels) != num_bins:
                        st.warning(f"Number of labels ({len(labels)}) doesn't match the number of bins ({num_bins})")
                
                elif mapping_method == "Custom Mapping":
                    # Custom mapping pairs
                    st.markdown("Enter mapping pairs (one per line in format: source_value = target_value)")
                    mapping_text = st.text_area("Mapping Pairs", "value1 = category1\nvalue2 = category2\nvalue3 = category3")
                    
                    # Default value
                    default_value = st.text_input("Default Value (for unmapped values)", "Other")
                
                else:  # Conditional Logic
                    # Add conditions
                    st.markdown("#### Define Conditions and Output Values")
                    
                    conditions = []
                    for i in range(1, 4):  # Allow up to 3 conditions
                        st.markdown(f"**Condition {i}**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            condition_column = st.selectbox(f"Column {i}", ["Select Column"] + columns, key=f"cond_col_{i}")
                        
                        with col2:
                            operator = st.selectbox(
                                f"Operator {i}",
                                ["==", "!=", ">", "<", ">=", "<=", "contains", "starts with", "ends with"],
                                key=f"cond_op_{i}"
                            )
                        
                        with col3:
                            value = st.text_input(f"Value {i}", key=f"cond_val_{i}")
                        
                        result = st.text_input(f"Output for Condition {i}", f"Result {i}")
                        
                        if condition_column != "Select Column" and value:
                            conditions.append((condition_column, operator, value, result))
                    
                    # Default result
                    default_result = st.text_input("Default Output (if no conditions match)", "Default")
                
                # New column name
                new_column = st.text_input("New Column Name", "category_column")
                
                # Create column button
                if st.button("Add Mapping Column"):
                    if not new_column:
                        st.error("Please enter a name for the new column")
                    else:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("Creating mapping column...")
                        progress_bar.progress(30)
                        
                        try:
                            # Create new dataframe
                            result_data = st.session_state.cleaned_data.copy()
                            
                            # Perform mapping operation
                            if mapping_method == "Value Ranges (Binning)":
                                # Check if column is numeric
                                if not is_numeric:
                                    # Try to convert to numeric
                                    result_data[column1] = pd.to_numeric(result_data[column1], errors='coerce')
                                
                                # Create bins
                                if len(labels) == num_bins:
                                    result_data[new_column] = pd.cut(
                                        result_data[column1],
                                        bins=num_bins,
                                        labels=labels
                                    )
                                else:
                                    # Use default labels if label count doesn't match
                                    result_data[new_column] = pd.cut(
                                        result_data[column1],
                                        bins=num_bins
                                    )
                            
                            elif mapping_method == "Custom Mapping":
                                # Parse mapping pairs
                                mapping = {}
                                for line in mapping_text.strip().split('\n'):
                                    if '=' in line:
                                        source, target = line.split('=', 1)
                                        mapping[source.strip()] = target.strip()
                                
                                # Apply mapping with default
                                result_data[new_column] = result_data[column1].map(mapping).fillna(default_value)
                            
                            else:  # Conditional Logic
                                # Initialize new column with default result
                                result_data[new_column] = default_result
                                
                                # Apply each condition
                                for cond_col, op, val, res in conditions:
                                    if op == "==":
                                        mask = result_data[cond_col] == val
                                    elif op == "!=":
                                        mask = result_data[cond_col] != val
                                    elif op == ">":
                                        mask = pd.to_numeric(result_data[cond_col], errors='coerce') > float(val)
                                    elif op == "<":
                                        mask = pd.to_numeric(result_data[cond_col], errors='coerce') < float(val)
                                    elif op == ">=":
                                        mask = pd.to_numeric(result_data[cond_col], errors='coerce') >= float(val)
                                    elif op == "<=":
                                        mask = pd.to_numeric(result_data[cond_col], errors='coerce') <= float(val)
                                    elif op == "contains":
                                        mask = result_data[cond_col].astype(str).str.contains(val, na=False)
                                    elif op == "starts with":
                                        mask = result_data[cond_col].astype(str).str.startswith(val, na=False)
                                    else:  # ends with
                                        mask = result_data[cond_col].astype(str).str.endswith(val, na=False)
                                    
                                    # Apply condition result
                                    result_data.loc[mask, new_column] = res
                            
                            # Update session state
                            st.session_state.enriched_data = result_data
                            
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Column added!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Show preview
                            st.success(f"Added new column '{new_column}' with mapping/categorization")
                            st.dataframe(result_data[[column1, new_column]].head(10), use_container_width=True)
                            
                            # Value counts
                            st.markdown("#### Value Distribution")
                            value_counts = result_data[new_column].value_counts()
                            
                            # Create bar chart of value counts
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Distribution of {new_column}",
                                labels={'x': new_column, 'y': 'Count'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Error creating column!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.error(f"Error creating mapping column: {str(e)}")
        
        elif operation == "Join with External Data":
            # Join with external data
            st.markdown("#### Join with External Data")
            st.markdown("Combine your dataset with external data by matching on common columns.")
            
            # Data source options
            data_source = st.radio(
                "External Data Source",
                ["Upload CSV File", "Sample Reference Data"],
                horizontal=True
            )
            
            if data_source == "Upload CSV File":
                # File upload
                uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
                
                if uploaded_file:
                    # Read CSV
                    try:
                        external_data = pd.read_csv(uploaded_file)
                        
                        # Show preview
                        st.markdown("#### External Data Preview")
                        st.dataframe(external_data.head(5), use_container_width=True)
                        
                        # Join configuration
                        st.markdown("#### Join Configuration")
                        
                        # Select join columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            left_join_col = st.selectbox("Join Column (Current Data)", st.session_state.cleaned_data.columns.tolist())
                        
                        with col2:
                            right_join_col = st.selectbox("Join Column (External Data)", external_data.columns.tolist())
                        
                        # Join type
                        join_type = st.selectbox(
                            "Join Type",
                            ["Left Join (Keep all current data)", "Inner Join (Keep only matching rows)", "Right Join (Keep all external data)", "Full Join (Keep all data)"]
                        )
                        
                        # Map join type to pandas merge method
                        if join_type == "Left Join (Keep all current data)":
                            how = "left"
                        elif join_type == "Inner Join (Keep only matching rows)":
                            how = "inner"
                        elif join_type == "Right Join (Keep all external data)":
                            how = "right"
                        else:  # Full Join
                            how = "outer"
                        
                        # Select columns to include
                        external_columns_to_include = st.multiselect(
                            "External Columns to Include",
                            [col for col in external_data.columns if col != right_join_col],
                            default=[col for col in external_data.columns if col != right_join_col]
                        )
                        
                        # Join button
                        if st.button("Join Data"):
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update progress
                            status_text.text("Preparing to join data...")
                            progress_bar.progress(20)
                            
                            try:
                                # Create copies of dataframes
                                left_df = st.session_state.cleaned_data.copy()
                                right_df = external_data.copy()
                                
                                # Select only needed columns from right dataframe
                                right_df = right_df[[right_join_col] + external_columns_to_include]
                                
                                # Update progress
                                status_text.text("Joining datasets...")
                                progress_bar.progress(50)
                                
                                # Perform join
                                result_data = pd.merge(
                                    left_df,
                                    right_df,
                                    left_on=left_join_col,
                                    right_on=right_join_col,
                                    how=how,
                                    suffixes=('', '_external')
                                )
                                
                                # Update session state
                                st.session_state.enriched_data = result_data
                                
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Data joined successfully!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Show results
                                st.success(f"Joined data with {len(result_data)} resulting rows and {len(result_data.columns)} columns")
                                
                                # Statistics
                                st.markdown("#### Join Statistics")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Original Rows", f"{len(left_df)}")
                                
                                with col2:
                                    st.metric("External Rows", f"{len(right_df)}")
                                
                                with col3:
                                    st.metric("Result Rows", f"{len(result_data)}")
                                
                                # Show preview
                                st.markdown("#### Result Preview")
                                st.dataframe(result_data.head(5), use_container_width=True)
                            
                            except Exception as e:
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Error joining data!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.error(f"Error joining data: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
            
            else:  # Sample Reference Data
                # Sample reference data types
                reference_type = st.selectbox(
                    "Reference Data Type",
                    ["Country Codes", "US State Data", "Currency Exchange Rates", "Industry Classifications", "Time Periods"]
                )
                
                # Show preview based on type
                if reference_type == "Country Codes":
                    # Create sample country code data
                    data = {
                        "country_name": ["United States", "Canada", "United Kingdom", "France", "Germany", "Japan", "China", "Brazil", "Australia", "India"],
                        "iso_code": ["US", "CA", "GB", "FR", "DE", "JP", "CN", "BR", "AU", "IN"],
                        "iso_code_3": ["USA", "CAN", "GBR", "FRA", "DEU", "JPN", "CHN", "BRA", "AUS", "IND"],
                        "region": ["North America", "North America", "Europe", "Europe", "Europe", "Asia", "Asia", "South America", "Oceania", "Asia"],
                        "population_millions": [331.0, 38.0, 67.8, 65.3, 83.8, 126.5, 1440.0, 212.6, 25.7, 1380.0]
                    }
                    
                    reference_data = pd.DataFrame(data)
                    join_on_options = ["country_name", "iso_code", "iso_code_3"]
                
                elif reference_type == "US State Data":
                    # Create sample US state data
                    data = {
                        "state_name": ["California", "Texas", "Florida", "New York", "Pennsylvania", "Illinois", "Ohio", "Georgia", "North Carolina", "Michigan"],
                        "state_code": ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"],
                        "region": ["West", "South", "South", "Northeast", "Northeast", "Midwest", "Midwest", "South", "South", "Midwest"],
                        "population_millions": [39.5, 29.0, 21.5, 19.5, 12.8, 12.7, 11.7, 10.6, 10.4, 10.0],
                        "gdp_billions": [3.0, 1.8, 1.1, 1.7, 0.8, 0.9, 0.7, 0.6, 0.6, 0.5]
                    }
                    
                    reference_data = pd.DataFrame(data)
                    join_on_options = ["state_name", "state_code"]
                
                elif reference_type == "Currency Exchange Rates":
                    # Create sample currency data
                    data = {
                        "currency_code": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL"],
                        "currency_name": ["US Dollar", "Euro", "British Pound", "Japanese Yen", "Canadian Dollar", "Australian Dollar", "Swiss Franc", "Chinese Yuan", "Indian Rupee", "Brazilian Real"],
                        "usd_rate": [1.0, 0.85, 0.72, 110.0, 1.25, 1.3, 0.92, 6.5, 74.0, 5.25],
                        "symbol": ["$", "€", "£", "¥", "CA$", "A$", "CHF", "CN¥", "₹", "R$"]
                    }
                    
                    reference_data = pd.DataFrame(data)
                    join_on_options = ["currency_code", "currency_name"]
                
                elif reference_type == "Industry Classifications":
                    # Create sample industry data
                    data = {
                        "industry_code": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                        "industry_name": ["Agriculture", "Mining", "Manufacturing", "Utilities", "Construction", "Wholesale", "Retail", "Transportation", "Finance", "Technology"],
                        "sector": ["Primary", "Primary", "Secondary", "Secondary", "Secondary", "Tertiary", "Tertiary", "Tertiary", "Tertiary", "Tertiary"],
                        "avg_growth": [2.1, 1.8, 3.5, 2.2, 4.1, 3.0, 3.2, 2.7, 3.9, 5.6],
                        "employees_thousands": [1200, 600, 12000, 800, 7000, 5800, 15200, 5500, 6300, 7200]
                    }
                    
                    reference_data = pd.DataFrame(data)
                    join_on_options = ["industry_code", "industry_name"]
                
                else:  # Time Periods
                    # Create sample time period data
                    periods = []
                    for year in range(2020, 2025):
                        for quarter in range(1, 5):
                            period_id = f"{year}Q{quarter}"
                            start_date = f"{year}-{(quarter-1)*3+1:02d}-01"
                            if quarter == 4:
                                end_date = f"{year}-12-31"
                            else:
                                end_month = quarter * 3
                                end_day = 30 if end_month in [4, 6, 9] else 31
                                end_date = f"{year}-{end_month:02d}-{end_day}"
                            
                            periods.append({
                                "period_id": period_id,
                                "year": year,
                                "quarter": quarter,
                                "start_date": start_date,
                                "end_date": end_date,
                                "fiscal_period": f"FY{year}Q{quarter}",
                                "days_in_period": pd.Timestamp(end_date) - pd.Timestamp(start_date) + pd.Timedelta(days=1)
                            })
                    
                    reference_data = pd.DataFrame(periods)
                    join_on_options = ["period_id", "year", "quarter", "start_date", "end_date"]
                
                # Show preview
                st.markdown("#### Reference Data Preview")
                st.dataframe(reference_data, use_container_width=True)
                
                # Join configuration
                st.markdown("#### Join Configuration")
                
                # Select join columns
                col1, col2 = st.columns(2)
                
                with col1:
                    left_join_col = st.selectbox("Join Column (Current Data)", st.session_state.cleaned_data.columns.tolist())
                
                with col2:
                    right_join_col = st.selectbox("Join Column (Reference Data)", join_on_options)
                
                # Join type
                join_type = st.selectbox(
                    "Join Type",
                    ["Left Join (Keep all current data)", "Inner Join (Keep only matching rows)"]
                )
                
                # Map join type to pandas merge method
                how = "left" if join_type == "Left Join (Keep all current data)" else "inner"
                
                # Select columns to include
                reference_columns_to_include = st.multiselect(
                    "Reference Columns to Include",
                    [col for col in reference_data.columns if col != right_join_col],
                    default=[col for col in reference_data.columns if col != right_join_col]
                )
                
                # Join button
                if st.button("Join with Reference Data"):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Preparing to join data...")
                    progress_bar.progress(20)
                    
                    try:
                        # Create copies of dataframes
                        left_df = st.session_state.cleaned_data.copy()
                        right_df = reference_data.copy()
                        
                        # Select only needed columns from right dataframe
                        right_df = right_df[[right_join_col] + reference_columns_to_include]
                        
                        # Update progress
                        status_text.text("Joining datasets...")
                        progress_bar.progress(50)
                        
                        # Perform join
                        result_data = pd.merge(
                            left_df,
                            right_df,
                            left_on=left_join_col,
                            right_on=right_join_col,
                            how=how,
                            suffixes=('', '_ref')
                        )
                        
                        # Update session state
                        st.session_state.enriched_data = result_data
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Data joined successfully!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        st.success(f"Joined data with {len(result_data)} resulting rows and {len(result_data.columns)} columns")
                        
                        # Show preview
                        st.markdown("#### Result Preview")
                        st.dataframe(result_data.head(5), use_container_width=True)
                        
                        # Show match statistics
                        match_count = result_data[right_join_col].notna().sum()
                        match_percentage = (match_count / len(result_data)) * 100
                        
                        st.markdown("#### Join Statistics")
                        st.markdown(f"• **Matches:** {match_count} out of {len(result_data)} rows ({match_percentage:.1f}%)")
                        
                        if match_percentage < 100:
                            st.warning(f"Some rows did not have matching values in the reference data.")
                    
                    except Exception as e:
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Error joining data!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.error(f"Error joining data: {str(e)}")
        
        elif operation == "Pivot Table":
            # Pivot table
            st.markdown("#### Create Pivot Table")
            st.markdown("Reshape your data by aggregating and summarizing across dimensions.")
            
            # Get columns
            columns = st.session_state.cleaned_data.columns.tolist()
            numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
            
            # Pivot configuration
            col1, col2 = st.columns(2)
            
            with col1:
                index_column = st.selectbox("Row Dimension", categorical_columns)
                values_column = st.selectbox("Values Column", numeric_columns)
                
            with col2:
                columns_column = st.selectbox("Column Dimension (optional)", ["None"] + categorical_columns)
                aggfunc = st.selectbox("Aggregation Function", ["Mean", "Sum", "Count", "Min", "Max", "Median"])
            
            # Map aggregation function
            if aggfunc == "Mean":
                agg_func = "mean"
            elif aggfunc == "Sum":
                agg_func = "sum"
            elif aggfunc == "Count":
                agg_func = "count"
            elif aggfunc == "Min":
                agg_func = "min"
            elif aggfunc == "Max":
                agg_func = "max"
            else:  # Median
                agg_func = "median"
            
            # Create pivot button
            if st.button("Create Pivot Table"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                status_text.text("Creating pivot table...")
                progress_bar.progress(50)
                
                try:
                    # Get data
                    data = st.session_state.cleaned_data.copy()
                    
                    # Create pivot table
                    if columns_column == "None":
                        pivot_table = pd.pivot_table(
                            data,
                            values=values_column,
                            index=index_column,
                            aggfunc=agg_func
                        )
                    else:
                        pivot_table = pd.pivot_table(
                            data,
                            values=values_column,
                            index=index_column,
                            columns=columns_column,
                            aggfunc=agg_func
                        )
                    
                    # Update session state
                    st.session_state.enriched_data = pivot_table.reset_index()
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Pivot table created!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    st.success(f"Created pivot table with {len(pivot_table)} rows and {len(pivot_table.columns)} columns")
                    
                    # Show pivot table
                    st.markdown("#### Pivot Table")
                    st.dataframe(pivot_table, use_container_width=True)
                    
                    # Create visualization
                    st.markdown("#### Pivot Table Visualization")
                    
                    if columns_column == "None":
                        # Simple bar chart
                        fig = px.bar(
                            pivot_table.reset_index(),
                            x=index_column,
                            y=values_column,
                            title=f"{values_column} by {index_column} ({aggfunc})",
                            height=500
                        )
                    else:
                        # Grouped bar chart
                        fig = px.bar(
                            pivot_table.reset_index().melt(
                                id_vars=index_column,
                                value_name=values_column
                            ),
                            x=index_column,
                            y=values_column,
                            color="variable",
                            title=f"{values_column} by {index_column} and {columns_column} ({aggfunc})",
                            height=500,
                            barmode="group"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Error creating pivot table!")
                    
                    # Clear progress indicators after a delay
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.error(f"Error creating pivot table: {str(e)}")
        
        else:  # Data Transformation
            # Data transformation
            st.markdown("#### Data Transformation")
            st.markdown("Apply transformations to reshape or reformat your entire dataset.")
            
            # Transformation options
            transformation = st.selectbox(
                "Transformation Type",
                ["Melt (Wide to Long)", "Stack & Unstack", "Group By & Aggregate", "One-Hot Encoding", "Sort & Filter"]
            )
            
            if transformation == "Melt (Wide to Long)":
                # Melt transformation
                st.markdown("#### Melt (Wide to Long Format)")
                st.markdown("Convert from wide format (many columns) to long format (fewer columns, more rows).")
                
                # Select ID columns
                id_columns = st.multiselect(
                    "ID Variables (columns to keep as identifier variables)",
                    st.session_state.cleaned_data.columns.tolist()
                )
                
                # Select value columns
                value_columns = st.multiselect(
                    "Value Variables (columns to unpivot into rows)",
                    [col for col in st.session_state.cleaned_data.columns if col not in id_columns]
                )
                
                # Column names
                col1, col2 = st.columns(2)
                
                with col1:
                    var_name = st.text_input("Variable Name Column", "variable")
                
                with col2:
                    value_name = st.text_input("Value Name Column", "value")
                
                # Melt button
                if st.button("Melt Data"):
                    if not id_columns:
                        st.error("Please select at least one ID column")
                    elif not value_columns:
                        st.error("Please select at least one value column")
                    else:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress
                        status_text.text("Melting data...")
                        progress_bar.progress(50)
                        
                        try:
                            # Get data
                            data = st.session_state.cleaned_data.copy()
                            
                            # Perform melt
                            melted_data = pd.melt(
                                data,
                                id_vars=id_columns,
                                value_vars=value_columns,
                                var_name=var_name,
                                value_name=value_name
                            )
                            
                            # Update session state
                            st.session_state.enriched_data = melted_data
                            
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Data melted successfully!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Show results
                            st.success(f"Melted data from {len(data)} rows × {len(data.columns)} columns to {len(melted_data)} rows × {len(melted_data.columns)} columns")
                            
                            # Show preview
                            st.markdown("#### Result Preview")
                            st.dataframe(melted_data.head(), use_container_width=True)
                            
                            # Show visualization
                            if len(melted_data) <= 1000:  # Limit visualization to 1000 rows
                                st.markdown("#### Visualization")
                                
                                # Check if value column is numeric
                                if melted_data[value_name].dtype in ['int64', 'float64']:
                                    # Create a small multiple chart
                                    if len(id_columns) > 0:
                                        fig = px.box(
                                            melted_data,
                                            x=var_name,
                                            y=value_name,
                                            color=id_columns[0] if len(id_columns) > 0 else None,
                                            facet_col=id_columns[1] if len(id_columns) > 1 else None,
                                            title=f"Distribution of {value_name} by {var_name}",
                                            height=500
                                        )
                                    else:
                                        fig = px.box(
                                            melted_data,
                                            x=var_name,
                                            y=value_name,
                                            title=f"Distribution of {value_name} by {var_name}",
                                            height=500
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("Error melting data!")
                            
                            # Clear progress indicators after a delay
                            time.sleep(0.5)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.error(f"Error melting data: {str(e)}")
            
            elif transformation == "Sort & Filter":
                # Sort and filter
                st.markdown("#### Sort & Filter Data")
                st.markdown("Sort your data by one or more columns and filter rows based on conditions.")
                
                # Sort options
                st.markdown("##### Sort Options")
                
                sort_columns = st.multiselect(
                    "Sort By Columns",
                    st.session_state.cleaned_data.columns.tolist()
                )
                
                if sort_columns:
                    sort_orders = []
                    for col in sort_columns:
                        order = st.radio(
                            f"Sort Order for {col}",
                            ["Ascending", "Descending"],
                            horizontal=True,
                            key=f"sort_{col}"
                        )
                        sort_orders.append(order == "Ascending")
                
                # Filter options
                st.markdown("##### Filter Options")
                
                add_filter = st.checkbox("Add Filter Conditions")
                
                filter_conditions = []
                if add_filter:
                    for i in range(3):  # Allow up to 3 filter conditions
                        st.markdown(f"**Filter Condition {i+1}**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            filter_column = st.selectbox(
                                "Column",
                                ["Select Column"] + st.session_state.cleaned_data.columns.tolist(),
                                key=f"filter_col_{i}"
                            )
                        
                        with col2:
                            filter_operator = st.selectbox(
                                "Operator",
                                ["==", "!=", ">", "<", ">=", "<=", "contains", "starts with", "ends with"],
                                key=f"filter_op_{i}"
                            )
                        
                        with col3:
                            filter_value = st.text_input("Value", key=f"filter_val_{i}")
                        
                        if filter_column != "Select Column" and filter_value:
                            filter_conditions.append((filter_column, filter_operator, filter_value))
                
                # Sample options
                st.markdown("##### Sample Options")
                
                take_sample = st.checkbox("Take Sample of Data")
                
                if take_sample:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sample_type = st.radio(
                            "Sample Type",
                            ["Number of Rows", "Percentage"],
                            horizontal=True
                        )
                    
                    with col2:
                        if sample_type == "Number of Rows":
                            sample_size = st.number_input(
                                "Number of Rows",
                                min_value=1,
                                max_value=len(st.session_state.cleaned_data),
                                value=min(100, len(st.session_state.cleaned_data))
                            )
                        else:
                            sample_percentage = st.slider(
                                "Percentage of Rows",
                                min_value=1,
                                max_value=100,
                                value=10
                            )
                            sample_size = int(len(st.session_state.cleaned_data) * sample_percentage / 100)
                    
                    random_sample = st.checkbox("Random Sample", value=True)
                
                # Apply button
                if st.button("Apply Sort & Filter"):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Processing data...")
                    progress_bar.progress(20)
                    
                    try:
                        # Get data
                        result_data = st.session_state.cleaned_data.copy()
                        
                        # Apply sort
                        if sort_columns:
                            result_data = result_data.sort_values(
                                by=sort_columns,
                                ascending=sort_orders
                            )
                            
                            # Update progress
                            status_text.text("Sorting data...")
                            progress_bar.progress(40)
                        
                        # Apply filters
                        if filter_conditions:
                            for column, operator, value in filter_conditions:
                                # Update progress
                                status_text.text(f"Filtering on {column}...")
                                
                                if operator == "==":
                                    result_data = result_data[result_data[column].astype(str) == value]
                                elif operator == "!=":
                                    result_data = result_data[result_data[column].astype(str) != value]
                                elif operator == ">":
                                    result_data = result_data[pd.to_numeric(result_data[column], errors='coerce') > float(value)]
                                elif operator == "<":
                                    result_data = result_data[pd.to_numeric(result_data[column], errors='coerce') < float(value)]
                                elif operator == ">=":
                                    result_data = result_data[pd.to_numeric(result_data[column], errors='coerce') >= float(value)]
                                elif operator == "<=":
                                    result_data = result_data[pd.to_numeric(result_data[column], errors='coerce') <= float(value)]
                                elif operator == "contains":
                                    result_data = result_data[result_data[column].astype(str).str.contains(value, na=False)]
                                elif operator == "starts with":
                                    result_data = result_data[result_data[column].astype(str).str.startswith(value, na=False)]
                                else:  # ends with
                                    result_data = result_data[result_data[column].astype(str).str.endswith(value, na=False)]
                            
                            # Update progress
                            progress_bar.progress(70)
                        
                        # Apply sampling
                        if take_sample:
                            # Update progress
                            status_text.text("Taking sample...")
                            
                            if random_sample:
                                result_data = result_data.sample(min(sample_size, len(result_data)))
                            else:
                                result_data = result_data.head(sample_size)
                        
                        # Update session state
                        st.session_state.enriched_data = result_data
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        st.success(f"Processed data to {len(result_data)} rows")
                        
                        # Show preview
                        st.markdown("#### Result Preview")
                        st.dataframe(result_data.head(10), use_container_width=True)
                        
                        # Show summary
                        st.markdown("#### Processing Summary")
                        
                        summary_points = []
                        
                        if sort_columns:
                            summary_points.append(f"• Sorted by {', '.join(sort_columns)}")
                        
                        if filter_conditions:
                            filter_descriptions = []
                            for column, operator, value in filter_conditions:
                                filter_descriptions.append(f"{column} {operator} {value}")
                            
                            summary_points.append(f"• Filtered with conditions: {', '.join(filter_descriptions)}")
                        
                        if take_sample:
                            sample_description = f"{'Random' if random_sample else 'Sequential'} sample of {sample_size} rows"
                            summary_points.append(f"• {sample_description}")
                        
                        for point in summary_points:
                            st.markdown(point)
                    
                    except Exception as e:
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Error processing data!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.error(f"Error during sort and filter: {str(e)}")
            
            elif transformation == "Group By & Aggregate":
                # Group by and aggregate
                st.markdown("#### Group By & Aggregate")
                st.markdown("Group your data by one or more columns and aggregate values using different functions.")
                
                # Group by columns
                group_columns = st.multiselect(
                    "Group By Columns",
                    st.session_state.cleaned_data.columns.tolist()
                )
                
                if not group_columns:
                    st.warning("Please select at least one column to group by.")
                else:
                    # Aggregation
                    st.markdown("##### Aggregation Configuration")
                    
                    # Get numeric columns
                    numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    
                    # Remove group columns from aggregation options
                    agg_columns = [col for col in numeric_columns if col not in group_columns]
                    
                    if not agg_columns:
                        st.warning("No numeric columns available for aggregation after selecting group columns.")
                    else:
                        # Create aggregation settings
                        agg_settings = {}
                        
                        for col in agg_columns:
                            st.markdown(f"**Aggregations for {col}**")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                use_mean = st.checkbox("Mean", key=f"mean_{col}")
                            
                            with col2:
                                use_sum = st.checkbox("Sum", key=f"sum_{col}")
                            
                            with col3:
                                use_min = st.checkbox("Min", key=f"min_{col}")
                            
                            with col4:
                                use_max = st.checkbox("Max", key=f"max_{col}")
                            
                            agg_funcs = []
                            if use_mean:
                                agg_funcs.append("mean")
                            if use_sum:
                                agg_funcs.append("sum")
                            if use_min:
                                agg_funcs.append("min")
                            if use_max:
                                agg_funcs.append("max")
                            
                            if agg_funcs:
                                agg_settings[col] = agg_funcs
                        
                        # Group by button
                        if st.button("Group and Aggregate"):
                            if not agg_settings:
                                st.error("Please select at least one aggregation function.")
                            else:
                                # Create a progress bar
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Update progress
                                status_text.text("Grouping data...")
                                progress_bar.progress(50)
                                
                                try:
                                    # Get data
                                    data = st.session_state.cleaned_data.copy()
                                    
                                    # Perform group by and aggregation
                                    grouped_data = data.groupby(group_columns).agg(agg_settings)
                                    
                                    # Reset index to make group columns regular columns
                                    grouped_data = grouped_data.reset_index()
                                    
                                    # Update session state
                                    st.session_state.enriched_data = grouped_data
                                    
                                    # Update progress
                                    progress_bar.progress(100)
                                    status_text.text("Grouping complete!")
                                    
                                    # Clear progress indicators after a delay
                                    time.sleep(0.5)
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # Show results
                                    st.success(f"Grouped {len(data)} rows into {len(grouped_data)} groups")
                                    
                                    # Show preview
                                    st.markdown("#### Result Preview")
                                    st.dataframe(grouped_data.head(), use_container_width=True)
                                    
                                    # Create visualization
                                    st.markdown("#### Visualization")
                                    
                                    if len(grouped_data) <= 20:  # Limit visualization to 20 groups
                                        # Find the column with the most aggregations
                                        most_aggs_col = max(agg_settings.items(), key=lambda x: len(x[1]))[0]
                                        
                                        # Create a grouped bar chart
                                        fig = go.Figure()
                                        
                                        for agg in agg_settings[most_aggs_col]:
                                            column_name = (most_aggs_col, agg)
                                            if column_name in grouped_data.columns:
                                                # Get values
                                                if isinstance(grouped_data.columns, pd.MultiIndex):
                                                    values = grouped_data[column_name]
                                                else:
                                                    # Handle flattened column names
                                                    values = grouped_data[f"{most_aggs_col}_{agg}"]
                                                
                                                # Add trace
                                                fig.add_trace(go.Bar(
                                                    x=grouped_data[group_columns[0]],
                                                    y=values,
                                                    name=f"{most_aggs_col} ({agg})"
                                                ))
                                        
                                        # Update layout
                                        fig.update_layout(
                                            title=f"Aggregations of {most_aggs_col} by {group_columns[0]}",
                                            xaxis_title=group_columns[0],
                                            yaxis_title="Value",
                                            barmode="group"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Too many groups to visualize. Showing only data table.")
                                
                                except Exception as e:
                                    # Update progress
                                    progress_bar.progress(100)
                                    status_text.text("Error grouping data!")
                                    
                                    # Clear progress indicators after a delay
                                    time.sleep(0.5)
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    st.error(f"Error during group by and aggregate: {str(e)}")
            
            elif transformation == "One-Hot Encoding":
                # One-hot encoding
                st.markdown("#### One-Hot Encoding")
                st.markdown("Convert categorical columns into multiple binary columns (one for each category).")
                
                # Get categorical columns
                categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
                
                if not categorical_columns:
                    st.warning("No categorical columns found in your data.")
                else:
                    # Select columns to encode
                    columns_to_encode = st.multiselect(
                        "Select Columns to Encode",
                        categorical_columns
                    )
                    
                    # Options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        prefix_sep = st.text_input("Separator", "_")
                    
                    with col2:
                        drop_first = st.checkbox("Drop First Category", value=False, help="Drop the first category to avoid multicollinearity")
                    
                    # One-hot encode button
                    if st.button("Apply One-Hot Encoding"):
                        if not columns_to_encode:
                            st.error("Please select at least one column to encode.")
                        else:
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update progress
                            status_text.text("Encoding data...")
                            progress_bar.progress(50)
                            
                            try:
                                # Get data
                                data = st.session_state.cleaned_data.copy()
                                
                                # Perform one-hot encoding
                                encoded_data = pd.get_dummies(
                                    data,
                                    columns=columns_to_encode,
                                    prefix_sep=prefix_sep,
                                    drop_first=drop_first
                                )
                                
                                # Update session state
                                st.session_state.enriched_data = encoded_data
                                
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Encoding complete!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Calculate new columns
                                new_columns = [col for col in encoded_data.columns if col not in data.columns]
                                
                                # Show results
                                st.success(f"Encoded {len(columns_to_encode)} columns, creating {len(new_columns)} new columns")
                                
                                # Show preview
                                st.markdown("#### Result Preview")
                                st.dataframe(encoded_data.head(), use_container_width=True)
                                
                                # Show new columns
                                st.markdown("#### New Columns Created")
                                st.markdown(f"Created {len(new_columns)} new columns:")
                                
                                # Show in a scrollable area if many columns
                                if len(new_columns) > 20:
                                    st.text_area("New Columns", "\n".join(new_columns), height=200)
                                else:
                                    for col in new_columns:
                                        st.markdown(f"• {col}")
                            
                            except Exception as e:
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Error encoding data!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.error(f"Error during one-hot encoding: {str(e)}")
            
            else:  # Stack & Unstack
                # Stack and unstack
                st.markdown("#### Stack & Unstack")
                st.markdown("Reshape hierarchical data by pivoting the levels of the index.")
                
                # Select index columns
                st.warning("Stack and Unstack operations work best with hierarchical indices. You might want to use 'Group By & Aggregate' first.")
                
                columns = st.session_state.cleaned_data.columns.tolist()
                
                # Operation
                operation_type = st.radio(
                    "Operation Type",
                    ["Stack (make data longer)", "Unstack (make data wider)"],
                    horizontal=True
                )
                
                if operation_type == "Stack (make data longer)":
                    # Stack operation
                    index_columns = st.multiselect(
                        "Select Index Columns",
                        columns
                    )
                    
                    if index_columns:
                        # Stack level
                        stack_level = st.selectbox(
                            "Stack Level (column to convert to index)",
                            [col for col in columns if col not in index_columns]
                        )
                        
                        # Stack button
                        if st.button("Apply Stack"):
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update progress
                            status_text.text("Stacking data...")
                            progress_bar.progress(50)
                            
                            try:
                                # Get data
                                data = st.session_state.cleaned_data.copy()
                                
                                # Set index
                                indexed_data = data.set_index(index_columns)
                                
                                # Stack
                                stacked_data = indexed_data.stack()
                                
                                # Reset index if not empty
                                if not stacked_data.empty:
                                    stacked_data = stacked_data.reset_index()
                                
                                # Update session state
                                st.session_state.enriched_data = stacked_data
                                
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Stacking complete!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Show results
                                st.success(f"Stacked data from {len(data)} rows × {len(data.columns)} columns to {len(stacked_data)} rows × {len(stacked_data.columns)} columns")
                                
                                # Show preview
                                st.markdown("#### Result Preview")
                                st.dataframe(stacked_data.head(), use_container_width=True)
                            
                            except Exception as e:
                                # Update progress
                                progress_bar.progress(100)
                                status_text.text("Error stacking data!")
                                
                                # Clear progress indicators after a delay
                                time.sleep(0.5)
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.error(f"Error during stack operation: {str(e)}")
                
                else:  # Unstack
                    # Unstack operation
                    st.markdown("To unstack, you need data with a hierarchical index (typically created by stacking or groupby).")
                    
                    # Check if data has a hierarchical index
                    has_hierarchical_index = False
                    
                    # Try using reset_index
                    unstack_columns = st.multiselect(
                        "Select Columns to Use as Index",
                        columns
                    )
                    
                    if unstack_columns:
                        # Select value column
                        value_column = st.selectbox(
                            "Select Value Column",
                            [col for col in columns if col not in unstack_columns]
                        )
                        
                        if value_column:
                            # Select unstacking level
                            unstack_level = st.selectbox(
                                "Unstack Level (which index level to convert to columns)",
                                unstack_columns
                            )
                            
                            # Unstack button
                            if st.button("Apply Unstack"):
                                # Create a progress bar
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Update progress
                                status_text.text("Unstacking data...")
                                progress_bar.progress(30)
                                
                                try:
                                    # Get data
                                    data = st.session_state.cleaned_data.copy()
                                    
                                    # Create pivoted data
                                    # Since we can't easily unstack without a hierarchical index,
                                    # we'll use pivot_table which is more flexible
                                    index_cols = [col for col in unstack_columns if col != unstack_level]
                                    
                                    unstacked_data = pd.pivot_table(
                                        data,
                                        values=value_column,
                                        index=index_cols,
                                        columns=unstack_level,
                                        aggfunc='first'
                                    )
                                    
                                    # Reset index
                                    unstacked_data = unstacked_data.reset_index()
                                    
                                    # Update progress
                                    progress_bar.progress(70)
                                    
                                    # Update session state
                                    st.session_state.enriched_data = unstacked_data
                                    
                                    # Update progress
                                    progress_bar.progress(100)
                                    status_text.text("Unstacking complete!")
                                    
                                    # Clear progress indicators after a delay
                                    time.sleep(0.5)
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # Show results
                                    st.success(f"Unstacked data from {len(data)} rows × {len(data.columns)} columns to {len(unstacked_data)} rows × {len(unstacked_data.columns)} columns")
                                    
                                    # Show preview
                                    st.markdown("#### Result Preview")
                                    st.dataframe(unstacked_data.head(), use_container_width=True)
                                
                                except Exception as e:
                                    # Update progress
                                    progress_bar.progress(100)
                                    status_text.text("Error unstacking data!")
                                    
                                    # Clear progress indicators after a delay
                                    time.sleep(0.5)
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    st.error(f"Error during unstack operation: {str(e)}")
        
        # Save enriched data
        if st.session_state.enriched_data is not None:
            st.markdown("#### Save Enriched Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply Changes to Dataset", key="apply_enriched_data_2"):
                    st.session_state.cleaned_data = st.session_state.enriched_data
                    st.success("Enriched data has been applied as the current dataset")
            
            with col2:
                if st.button("Discard Changes", key="discard_enriched_data_2"):
                    st.session_state.enriched_data = None
                    st.info("Enriched data has been discarded")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab4:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Advanced Formulas & Lookups")
        st.markdown("""
        Apply Excel and Power BI style formulas to your data including VLOOKUP, HLOOKUP, XLOOKUP, and DAX formulas.
        These tools help you enhance your dataset with lookups and calculations commonly used in Excel and Power BI.
        """)
        
        # Create subtabs for different formula types
        formula_tabs = st.tabs(["VLOOKUP", "XLOOKUP", "DAX LOOKUPVALUE", "Upload Lookup Table"])
        
        with formula_tabs[0]:  # VLOOKUP
            st.markdown("#### VLOOKUP Formula")
            st.markdown("""
            VLOOKUP searches for a value in the leftmost column of a table, and returns a value in the same row from a column you specify.
            """)
            
            # Show available columns
            columns = st.session_state.cleaned_data.columns.tolist()
            
            # Get available lookup tables
            lookup_tables = list(st.session_state.lookup_tables.keys())
            
            if not lookup_tables:
                st.info("You need to upload a lookup table first. Go to the 'Upload Lookup Table' tab to add one.")
            else:
                # Select lookup column in main data
                lookup_col = st.selectbox(
                    "Select Column with Values to Look Up",
                    columns,
                    key="vlookup_lookup_col"
                )
                
                # Select lookup table
                lookup_table_name = st.selectbox(
                    "Select Lookup Table",
                    lookup_tables,
                    key="vlookup_table"
                )
                
                lookup_table = st.session_state.lookup_tables[lookup_table_name]
                lookup_table_columns = lookup_table.columns.tolist()
                
                # Select lookup column in lookup table
                table_lookup_col = st.selectbox(
                    "Select Column to Search In (in Lookup Table)",
                    lookup_table_columns,
                    key="vlookup_table_lookup_col"
                )
                
                # Select return column in lookup table
                table_return_col = st.selectbox(
                    "Select Column to Return Value From (in Lookup Table)",
                    lookup_table_columns,
                    key="vlookup_table_return_col"
                )
                
                # Name for new column
                result_col = st.text_input(
                    "New Column Name",
                    f"vlookup_{table_return_col}",
                    key="vlookup_result_col"
                )
                
                # Exact match option
                exact_match = st.checkbox(
                    "Require Exact Match",
                    True,
                    key="vlookup_exact_match",
                    help="If checked, only exact matches will return values. If unchecked, approximate matches will be found."
                )
                
                # Example syntax
                st.markdown("#### Formula Preview")
                formula = f"VLOOKUP([{lookup_col}], [{lookup_table_name}], [{table_return_col}], {str(exact_match).upper()})"
                st.code(formula)
                
                # Preview of lookup table
                with st.expander("Preview Lookup Table"):
                    st.dataframe(lookup_table.head(5), use_container_width=True)
                
                # Apply button
                if st.button("Apply VLOOKUP", key="apply_vlookup"):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Applying VLOOKUP formula...")
                    progress_bar.progress(30)
                    
                    try:
                        # Set up parameters
                        params = {
                            "lookup_column": lookup_col,
                            "table_df": lookup_table,
                            "table_lookup_column": table_lookup_col,
                            "table_return_column": table_return_col,
                            "result_column": result_col,
                            "exact_match": exact_match
                        }
                        
                        # Apply formula
                        result_df = apply_formula_to_column(
                            st.session_state.cleaned_data,
                            "VLOOKUP",
                            params
                        )
                        
                        # Update progress
                        progress_bar.progress(70)
                        status_text.text("Processing results...")
                        
                        # Update session state
                        st.session_state.enriched_data = result_df
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("VLOOKUP applied successfully!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        st.success(f"VLOOKUP formula applied! New column '{result_col}' added to your data.")
                        
                        # Preview results
                        st.markdown("#### Results Preview")
                        preview_cols = [lookup_col, result_col]
                        st.dataframe(result_df[preview_cols].head(10), use_container_width=True)
                    
                    except Exception as e:
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Error applying VLOOKUP!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.error(f"Error applying VLOOKUP: {str(e)}")
                
        with formula_tabs[1]:  # XLOOKUP
            st.markdown("#### XLOOKUP Formula")
            st.markdown("""
            XLOOKUP is the modern replacement for VLOOKUP that can search in any direction and return values from any direction.
            It's more flexible and powerful than VLOOKUP.
            """)
            
            # Show available columns
            columns = st.session_state.cleaned_data.columns.tolist()
            
            # Get available lookup tables
            lookup_tables = list(st.session_state.lookup_tables.keys())
            
            if not lookup_tables:
                st.info("You need to upload a lookup table first. Go to the 'Upload Lookup Table' tab to add one.")
            else:
                # Select lookup column in main data
                lookup_col = st.selectbox(
                    "Select Column with Values to Look Up",
                    columns,
                    key="xlookup_lookup_col"
                )
                
                # Select lookup table
                lookup_table_name = st.selectbox(
                    "Select Lookup Table",
                    lookup_tables,
                    key="xlookup_table"
                )
                
                lookup_table = st.session_state.lookup_tables[lookup_table_name]
                lookup_table_columns = lookup_table.columns.tolist()
                
                # Select lookup column in lookup table
                table_lookup_col = st.selectbox(
                    "Select Column to Search In (in Lookup Table)",
                    lookup_table_columns,
                    key="xlookup_table_lookup_col"
                )
                
                # Select return column in lookup table
                table_return_col = st.selectbox(
                    "Select Column to Return Value From (in Lookup Table)",
                    lookup_table_columns,
                    key="xlookup_table_return_col"
                )
                
                # Name for new column
                result_col = st.text_input(
                    "New Column Name",
                    f"xlookup_{table_return_col}",
                    key="xlookup_result_col"
                )
                
                # If not found value
                if_not_found = st.text_input(
                    "Value if Not Found",
                    "Not Found",
                    key="xlookup_if_not_found"
                )
                
                # Example syntax
                st.markdown("#### Formula Preview")
                formula = f"XLOOKUP([{lookup_col}], [{lookup_table_name}].[{table_lookup_col}], [{lookup_table_name}].[{table_return_col}], \"{if_not_found}\")"
                st.code(formula)
                
                # Preview of lookup table
                with st.expander("Preview Lookup Table"):
                    st.dataframe(lookup_table.head(5), use_container_width=True)
                
                # Apply button
                if st.button("Apply XLOOKUP", key="apply_xlookup"):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Applying XLOOKUP formula...")
                    progress_bar.progress(30)
                    
                    try:
                        # Set up parameters
                        params = {
                            "lookup_column": lookup_col,
                            "table_df": lookup_table,
                            "table_lookup_column": table_lookup_col,
                            "table_return_column": table_return_col,
                            "result_column": result_col,
                            "if_not_found": if_not_found
                        }
                        
                        # Apply formula
                        result_df = apply_formula_to_column(
                            st.session_state.cleaned_data,
                            "XLOOKUP",
                            params
                        )
                        
                        # Update progress
                        progress_bar.progress(70)
                        status_text.text("Processing results...")
                        
                        # Update session state
                        st.session_state.enriched_data = result_df
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("XLOOKUP applied successfully!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        st.success(f"XLOOKUP formula applied! New column '{result_col}' added to your data.")
                        
                        # Preview results
                        st.markdown("#### Results Preview")
                        preview_cols = [lookup_col, result_col]
                        st.dataframe(result_df[preview_cols].head(10), use_container_width=True)
                    
                    except Exception as e:
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Error applying XLOOKUP!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.error(f"Error applying XLOOKUP: {str(e)}")
                
        with formula_tabs[2]:  # DAX LOOKUPVALUE
            st.markdown("#### DAX LOOKUPVALUE Formula")
            st.markdown("""
            LOOKUPVALUE is a DAX function from Power BI that returns a value from a table when one or more matches are found.
            It's more powerful than VLOOKUP because it can search using multiple conditions.
            """)
            
            # Show available columns
            columns = st.session_state.cleaned_data.columns.tolist()
            
            # Get available lookup tables
            lookup_tables = list(st.session_state.lookup_tables.keys())
            
            if not lookup_tables:
                st.info("You need to upload a lookup table first. Go to the 'Upload Lookup Table' tab to add one.")
            else:
                # Select lookup column in main data
                lookup_col = st.selectbox(
                    "Select Column with Values to Look Up",
                    columns,
                    key="dax_lookup_col"
                )
                
                # Select lookup table
                lookup_table_name = st.selectbox(
                    "Select Lookup Table",
                    lookup_tables,
                    key="dax_table"
                )
                
                lookup_table = st.session_state.lookup_tables[lookup_table_name]
                lookup_table_columns = lookup_table.columns.tolist()
                
                # Select return column in lookup table
                table_return_col = st.selectbox(
                    "Select Column to Return Value From (in Lookup Table)",
                    lookup_table_columns,
                    key="dax_table_return_col"
                )
                
                # Select lookup column in lookup table
                table_lookup_col = st.selectbox(
                    "Select Column to Search In (in Lookup Table)",
                    lookup_table_columns,
                    key="dax_table_lookup_col"
                )
                
                # Optional additional filter
                add_filter = st.checkbox("Add Additional Filter Condition", key="dax_add_filter")
                
                additional_filters = []
                if add_filter:
                    # Select filter column in lookup table
                    filter_col = st.selectbox(
                        "Select Filter Column (in Lookup Table)",
                        lookup_table_columns,
                        key="dax_filter_col"
                    )
                    
                    # Select filter value from unique values
                    unique_filter_values = lookup_table[filter_col].unique().tolist()
                    filter_value = st.selectbox(
                        "Select Filter Value",
                        unique_filter_values,
                        key="dax_filter_value"
                    )
                    
                    additional_filters = [filter_col, filter_value]
                
                # Name for new column
                result_col = st.text_input(
                    "New Column Name",
                    f"dax_{table_return_col}",
                    key="dax_result_col"
                )
                
                # Example syntax
                st.markdown("#### Formula Preview")
                if additional_filters:
                    formula = f"LOOKUPVALUE([{table_return_col}], [{table_lookup_col}], [{lookup_col}], [{additional_filters[0]}], \"{additional_filters[1]}\")"
                else:
                    formula = f"LOOKUPVALUE([{table_return_col}], [{table_lookup_col}], [{lookup_col}])"
                st.code(formula)
                
                # Preview of lookup table
                with st.expander("Preview Lookup Table"):
                    st.dataframe(lookup_table.head(5), use_container_width=True)
                
                # Apply button
                if st.button("Apply DAX LOOKUPVALUE", key="apply_dax"):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Applying DAX LOOKUPVALUE formula...")
                    progress_bar.progress(30)
                    
                    try:
                        # Set up parameters
                        params = {
                            "lookup_column": lookup_col,
                            "table_df": lookup_table,
                            "table_lookup_column": table_lookup_col,
                            "table_return_column": table_return_col,
                            "result_column": result_col,
                            "additional_filters": additional_filters
                        }
                        
                        # Apply formula
                        result_df = apply_formula_to_column(
                            st.session_state.cleaned_data,
                            "DAX_LOOKUPVALUE",
                            params
                        )
                        
                        # Update progress
                        progress_bar.progress(70)
                        status_text.text("Processing results...")
                        
                        # Update session state
                        st.session_state.enriched_data = result_df
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("DAX LOOKUPVALUE applied successfully!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show results
                        st.success(f"DAX LOOKUPVALUE formula applied! New column '{result_col}' added to your data.")
                        
                        # Preview results
                        st.markdown("#### Results Preview")
                        preview_cols = [lookup_col, result_col]
                        st.dataframe(result_df[preview_cols].head(10), use_container_width=True)
                    
                    except Exception as e:
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Error applying DAX LOOKUPVALUE!")
                        
                        # Clear progress indicators after a delay
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.error(f"Error applying DAX LOOKUPVALUE: {str(e)}")
                
        with formula_tabs[3]:  # Upload Lookup Table
            st.markdown("#### Upload Lookup Table")
            st.markdown("""
            Upload a CSV or Excel file to use as a lookup table for VLOOKUP, XLOOKUP, and DAX formulas.
            """)
            
            # File uploader
            lookup_file = st.file_uploader(
                "Upload a CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key="lookup_file_uploader"
            )
            
            if lookup_file:
                try:
                    # Determine file type
                    file_ext = lookup_file.name.split(".")[-1].lower()
                    
                    # Read file
                    if file_ext == "csv":
                        lookup_df = pd.read_csv(lookup_file)
                    else:
                        lookup_df = pd.read_excel(lookup_file)
                    
                    # Display table
                    st.markdown("#### Preview")
                    st.dataframe(lookup_df.head(5), use_container_width=True)
                    
                    # Table info
                    st.markdown("#### Table Information")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rows", f"{len(lookup_df)}")
                    
                    with col2:
                        st.metric("Columns", f"{len(lookup_df.columns)}")
                    
                    with col3:
                        st.metric("Table Size", f"{lookup_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    
                    # Name for lookup table
                    table_name = st.text_input(
                        "Lookup Table Name",
                        lookup_file.name.split(".")[0],
                        key="lookup_table_name"
                    )
                    
                    # Save button
                    if st.button("Save Lookup Table", key="save_lookup_table"):
                        # Add to session state
                        st.session_state.lookup_tables[table_name] = lookup_df
                        
                        st.success(f"Lookup table '{table_name}' saved and ready to use with lookup formulas!")
                        
                        # Show available tables
                        st.markdown("#### Available Lookup Tables")
                        for name, df in st.session_state.lookup_tables.items():
                            st.markdown(f"- **{name}**: {len(df)} rows, {len(df.columns)} columns")
                
                except Exception as e:
                    st.error(f"Error reading lookup table: {str(e)}")
            
            # Show currently available tables
            if st.session_state.lookup_tables:
                st.markdown("#### Available Lookup Tables")
                for name, df in st.session_state.lookup_tables.items():
                    st.markdown(f"- **{name}**: {len(df)} rows, {len(df.columns)} columns")
                    
                # Option to delete a lookup table
                table_to_delete = st.selectbox(
                    "Select table to delete",
                    list(st.session_state.lookup_tables.keys()),
                    key="delete_lookup_table_selector"
                )
                
                if st.button("Delete Selected Lookup Table", key="delete_lookup_table"):
                    if table_to_delete in st.session_state.lookup_tables:
                        del st.session_state.lookup_tables[table_to_delete]
                        st.success(f"Lookup table '{table_to_delete}' deleted!")
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)