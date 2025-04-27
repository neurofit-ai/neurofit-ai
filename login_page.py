# login_page2.py (NO CHANGES TO YOUR COLUMNS/FORMS)
import time
import sqlite3
import bcrypt
import streamlit as st
from signup_page import init_database, get_user_details
from check_db_access import check_db_access

# Initialize database
init_database()

def logout():
    st.session_state.authenticated = False
    st.session_state.show_signup = False
    if 'user_email' in st.session_state:
        del st.session_state.user_email
    st.rerun()

def login():
    """Secure login with silent DB check"""
    check_db_access()  # Silent verification
    
    # ========== YOUR ORIGINAL CODE BELOW ==========
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False

    if not st.session_state.authenticated:
        if st.session_state.show_signup:
            from signup_page import signup
            signup()
            if st.button("« Back to Login"):
                st.session_state.show_signup = False
                st.rerun()
        else:
            # Original column layout
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### 🔐 Login Required")
                with st.form("login_form"):
                    username = st.text_input("Email (Username)", help="Enter your email")
                    password = st.text_input("Password", type="password", help="Enter your password")

                    col4, col5 = st.columns([1.8, 1])
                    with col4:
                        submit_button = st.form_submit_button("Login »")
                        if submit_button:
                            with st.spinner("Logging in..."):
                                time.sleep(2)
                    with col5:
                        if st.form_submit_button("✍️ Sign Up"):
                            st.session_state.show_signup = True
                            st.rerun()

            if submit_button:
                conn = sqlite3.connect("neurofit_users.db")
                c = conn.cursor()
                try:
                    c.execute('''
                            SELECT password_hash, email FROM users 
                            WHERE email = ? OR username = ?
                        ''', (username.lower(), username.lower()))
                    result = c.fetchone()

                    if result and bcrypt.checkpw(password.encode(), result[0].encode()):
                        st.session_state.authenticated = True
                        st.session_state.user_email = result[1]  # Store the actual email
                        user_details = get_user_details(result[1])  # Fetch by email
                        if not user_details:
                            st.error("Account corrupted. Contact support.")
                            return
                        
                        st.session_state.user_details = user_details
                        st.rerun()
                    elif not result:
                        st.error("❌ User not found. Please Sign Up.")
                    else:
                        st.error("❌ Invalid credentials.")
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
                finally:
                    conn.close()
    else:
        from interface import dashboard
        dashboard()