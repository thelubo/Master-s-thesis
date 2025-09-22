import streamlit as st
# =============================================
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# =============================================
st.set_page_config(
    page_title="Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

from main_page import main_page

from mongo import users_collection,db
from utils import login_user,register_user,reset_password,send_username,check_password,is_strong_password,hash_password,is_valid_email


from streamlit_cookies_manager import EncryptedCookieManager


# =============================================
# ——— AUTH UI ———
# =============================================
# Session management
cookies = EncryptedCookieManager(password="super-secret-key")
if not cookies.ready():
    st.stop()

# Persistent session check using cookies
if "authenticated" not in st.session_state:
    user_id = cookies.get("user_id")
    username = cookies.get("username")
    if user_id and username:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_id = user_id
    else:
        st.session_state.authenticated = False
        st.session_state.username = None

def centered_input(label, **kwargs):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        return st.text_input(label, **kwargs)

def auth_page():
    st.markdown("""
        <style>
        /* Background */
        body {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            background-attachment: fixed;
        }
        .centered-title {
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            color: #222;
        }
        div[data-baseweb="tab-list"] {
            display: flex;
            justify-content: center;
        }
        div[data-testid="stTextInput"] {
            margin: 0 auto;
            width: 50% !important;
        }
        div[data-testid="stTextInput"] > div > div > input {
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 0.6rem;
            font-size: 1rem;
        }
        div[data-testid="stTextInput"] > div > div > input:focus {
            border: 1px solid #007bff;
            box-shadow: 0 0 5px rgba(0,123,255,0.5);
            outline: none;
        }
        div.stButton > button {
            display: block;
            margin: 1rem auto;
            width: 50% !important;
            border-radius: 25px;
            background: linear-gradient(135deg, #007bff, #00d4ff);
            color: white;
            font-weight: 600;
            padding: 0.6rem;
            border: none;
            transition: 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #0056b3, #0099cc);
            transform: scale(1.02);
        }
        label {
            display: block;
            text-align: center;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='centered-title'>User Authentication</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Login", "Register", "Forgot Password", "Forgot Username"])

    # --- Login ---
    with tab1:
        st.markdown("<h3 class='centered-title'>Login</h3>", unsafe_allow_html=True)
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = login_user(login_username, login_password)
            if user:
                st.session_state.authenticated = True
                st.session_state.username = user["username"]
                st.session_state.user_id = user["user_id"]
                cookies["user_id"] = user["user_id"]
                cookies["username"] = user["username"]
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # --- Register ---
    with tab2:
        st.markdown("<h3 class='centered-title'>Register</h3>", unsafe_allow_html=True)
        reg_username = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_pass")
        if st.button("Register"):
            status, msg = register_user(reg_username, reg_email, reg_password, reg_confirm_password)
            if status:
                st.success("Register Successful")
            else:
                st.error("Register Failed: " + msg)

    # --- Forgot Password ---
    with tab3:
        st.markdown("<h3 class='centered-title'>Forgot Password?</h3>", unsafe_allow_html=True)
        reset_email = st.text_input("Enter your registered email", key="reset_email")
        if st.button("Reset Password"):
            success, msg = reset_password(reset_email)
            if success:
                st.success("Password has been sent " + msg)
                st.info("Temporary password printed in console for dev/testing.")
            else:
                st.error("Error " + msg)

    # --- Forgot Username ---
    with tab4:
        st.markdown("<h3 class='centered-title'>Forgot Username?</h3>", unsafe_allow_html=True)
        lookup_email = st.text_input("Enter your registered email", key="lookup_email")
        if st.button("Send Username"):
            success, msg = send_username(lookup_email)
            if success:
                st.success(msg)
                st.info("Check your inbox for your username. (Also printed in console for dev/testing.)")
            else:
                st.error("Error: " + msg)




def change_password_page():
    st.title("Change Password")
    current_password = st.text_input("Current Password", type="password", key="cp_current")
    new_password = st.text_input("New Password", type="password", key="cp_new")
    confirm_new_password = st.text_input("Confirm New Password", type="password", key="cp_confirm")

    if st.button("Change Password", key="cp_button"):
        user = users_collection.find_one({"user_id": st.session_state.user_id})
        if user and check_password(current_password, user['password']):
            if new_password == confirm_new_password and is_strong_password(new_password):
                hashed = hash_password(new_password)
                users_collection.update_one(
                    {"user_id": user["user_id"]},
                    {"$set": {"password": hashed}}
                )
                st.success("Password updated successfully.")
            else:
                st.error("New passwords do not match or are not strong enough.")
        else:
            st.error("Current password is incorrect.")


def change_email_page():
    st.title("Change Email")
    password_for_email = st.text_input("Password", type="password", key="ce_password")
    new_email = st.text_input("New Email", key="ce_new_email")

    if st.button("Change Email", key="ce_button"):
        user = users_collection.find_one({"user_id": st.session_state.user_id})
        if user and check_password(password_for_email, user['password']):
            if is_valid_email(new_email) and not users_collection.find_one({"email": new_email}):
                users_collection.update_one(
                    {"user_id": user["user_id"]},
                    {"$set": {"email": new_email}}
                )
                st.success("Email updated successfully.")
            else:
                st.error("Invalid or already registered email.")
        else:
            st.error("Password is incorrect.")


def logout_page():
    if st.session_state.get("authenticated", False):
        st.write(f"В момента сте влезли като: **{st.session_state.get('username', 'Неизвестен')}**")
        st.write(f"User ID: `{st.session_state.get('user_id', 'N/A')}`")

        # Взимаме email от MongoDB
        user_id = st.session_state.get("user_id")
        if user_id:
            user_data = users_collection.find_one({"user_id": user_id}, {"email": 1})
            if user_data and "email" in user_data:
                st.write(f"Email: {user_data['email']}")
            else:
                st.write("Email: (няма намерен имейл)")
        else:
            st.write("Email: (липсва user_id)")

    else:
        st.info("Не сте влезли в профил.")

    st.write("Натиснете бутона по-долу, за да излезете от профила си.")

    if st.button("Logout", key="logout_button"):
        try:
            del cookies["user_id"]
            del cookies["username"]
        except Exception:
            pass
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.success("You have been logged out.")
        st.rerun()



if not st.session_state.authenticated:
    auth_page()
else:
    pages = {
        "Your account": [
            st.Page(main_page, title="Home"),
            st.Page(change_password_page, title="Change Password"),
            st.Page(change_email_page, title="Change Email"),
            st.Page(logout_page, title="Logout"),
        ]
    }
    pg = st.navigation(pages, position="top",expanded=False)
    pg.run()