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

# --- MAIN ROUTER ---
# Directly load Home page (or implement your own routing logic here)
module = importlib.import_module("Home")
if hasattr(module, "main"):
    module.main()
else:
    st.error("This page does not have a main() function.") 