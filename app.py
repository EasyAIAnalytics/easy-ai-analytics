import streamlit as st
import database as db
import bcrypt
import importlib

# --- PAGE REGISTRY ---
PAGES = {
    "Home": "Home",
    "Data Enrichment": "pages.Data_Enrichment",
    "Advanced Statistics": "pages.Advanced_Statistics",
    "AI Analytics": "pages.AI_Analytics",
    "Business Features": "pages.Business_Features",
    "Advanced Formulas": "pages.Advanced_Formulas",
}

# --- ROUTING LOGIC ---
def show_logout():
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Logout", key="logout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

def show_login():
    st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")
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

def show_signup():
    st.set_page_config(page_title="Sign Up", page_icon="📝", layout="centered")
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

# --- MAIN ROUTER ---
query_params = st.query_params
page = query_params.get("page", "home").lower()

if "user_id" not in st.session_state:
    if page == "signup":
        show_signup()
    else:
        show_login()
else:
    show_logout()
    # Main app navigation (tabs)
    tab_names = list(PAGES.keys())
    default_tab = tab_names[0]
    selected_tab = st.selectbox("Navigation", tab_names, key="main_nav")
    module_name = PAGES[selected_tab]
    module = importlib.import_module(module_name)
    if hasattr(module, "main"):
        module.main()
    else:
        st.error("This page does not have a main() function.") 
