import streamlit as st
import pandas as pd
import numpy as np
import time
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Formulas & Lookups",
    page_icon="🔣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title with styling
st.markdown('<h1 class="main-header">Advanced Formulas & Lookups</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Apply Excel and Power BI style formulas to enhance your data</p>', unsafe_allow_html=True)

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
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using this feature.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

if "enriched_data" not in st.session_state:
    st.session_state.enriched_data = None

if "lookup_tables" not in st.session_state:
    st.session_state.lookup_tables = {}

# Check if data is available
if st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using formula features.")
else:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Excel and Power BI Style Formulas")
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
        columns = st.session_state.data.columns.tolist()
        
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
                        st.session_state.data,
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
        columns = st.session_state.data.columns.tolist()
        
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
                        st.session_state.data,
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
        columns = st.session_state.data.columns.tolist()
        
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
                        st.session_state.data,
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
    
    # Save enriched data
    if st.session_state.enriched_data is not None:
        st.markdown("#### Save Enriched Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Apply Changes to Dataset", key="apply_enriched_data"):
                st.session_state.data = st.session_state.enriched_data
                st.success("Enriched data has been applied as the current dataset")
                st.rerun()
        
        with col2:
            if st.button("Discard Changes", key="discard_enriched_data"):
                st.session_state.enriched_data = None
                st.info("Enriched data has been discarded")
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    if st.button("Logout", key="logout_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()