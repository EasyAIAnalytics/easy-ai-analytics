import streamlit as st
import database as db
import bcrypt
import importlib
import os

st.set_page_config(page_title="Easy AI Analytics", page_icon="🧠", layout="wide")

# --- Page Registry ---
RAW_PAGES = {
    "Home": "Home",
    "Data Enrichment": "pages.Data_Enrichment",
    "Advanced Statistics": "pages.Advanced_Statistics",
    "AI Analytics": "pages.AI_Analytics",
    "Business Features": "pages.Business_Features",
    "Database Management": "pages.Database_Management",  # Will be filtered if not present
    "Advanced Formulas": "pages.Advanced_Formulas",
}

# Only include pages that physically exist
PAGES = {
    title: path
    for title, path in RAW_PAGES.items()
    if os.path.exists(path.replace(".", "/") + ".py")
}

# --- Logout UI ---
def show_logout():
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Logout", key="logout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# --- Login UI ---
def show_login():
    st.markdown("<h2 style='text-align:center;'>Login to Easy AI Analytics</h2>", unsafe_allow_html=True)
    error = None
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submit = st.form_submit_button("Login")
    if submit:
        user = db.get_user_by_username(username)
        if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            st.session_state.user_id = user["id"]
            st.session_state.username = user["username"]
            st.success(f"Welcome back, {user['username']}!")
            st.rerun()
        else:
            error = "Invalid username or password."
    if error:
        st.error(error)
    st.markdown("<p style='text-align:center;'>Don't have an account? <a href='?page=signup'>Sign up</a></p>", unsafe_allow_html=True)

# --- Signup UI ---
def show_signup():
    st.markdown("<h2 style='text-align:center;'>Create Your Account</h2>", unsafe_allow_html=True)
    error = None
    with st.form("signup_form"):
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        submit = st.form_submit_button("Sign Up")
    if submit:
        if db.get_user_by_username(username):
            error = "Username already exists."
        elif db.get_user_by_email(email):
            error = "Email already registered."
        else:
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            db.create_user(username, email, hashed)
            st.success("Account created! Please log in.")
            st.rerun()
    if error:
        st.error(error)
    st.markdown("<p style='text-align:center;'>Already have an account? <a href='?page=login'>Login</a></p>", unsafe_allow_html=True)

# --- Main App Router ---
page = st.query_params.get("page", ["home"])[0].lower()

if "user_id" not in st.session_state:
    if page == "signup":
        show_signup()
    else:
        show_login()
else:
    show_logout()
    tab_names = list(PAGES.keys())
    selected_tab = st.selectbox("Navigation", tab_names, key="main_nav")
    module_name = PAGES[selected_tab]
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "main"):
            module.main()
        else:
            st.error(f"Module `{module_name}` does not have a `main()` function.")
    except Exception as e:
        st.error(f"Error loading module `{module_name}`: {e}")
