import streamlit as st
import importlib

# --- PAGE REGISTRY ---
PAGES = {
    "Home": "Home",
    "Data Enrichment": "pages.Data_Enrichment",
    "Advanced Statistics": "pages.Advanced_Statistics",
    "AI Analytics": "pages.AI_Analytics",
    "Business Features": "pages.Business_Features",
    "Database Management": "pages.Database_Management",
    "Advanced Formulas": "pages.Advanced_Formulas",
}

# --- Sidebar Navigation ---
st.sidebar.title("Easy AI Analytics")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# --- Main Routing Logic ---
selected_module = PAGES[selection]
try:
    module = importlib.import_module(selected_module)
    if hasattr(module, "main"):
        module.main()
    else:
        st.error(f"The `{selection}` page does not have a `main()` function.")
except ModuleNotFoundError:
    st.error(f"The module `{selected_module}` could not be found.")
except Exception as e:
    st.error(f"An error occurred while loading `{selection}`: {e}")
