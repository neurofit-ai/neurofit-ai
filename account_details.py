import streamlit as st
import time
from signup_page import get_user_details, delete_user
from login_page import logout

def account():
    # Back to Dashboard button
    if st.button("← Back to Dashboard", key="back_button"):
        st.session_state.show_account = False
        st.rerun()

    """Display authenticated user's account details with deletion option"""
    if 'user_email' not in st.session_state:
        st.error("No active session. Please login first.")
        return
    
    details = get_user_details(st.session_state.user_email)
    
    if not details:
        st.error("User details not found")
        return
    
    st.header("Account Details")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"**Name:** *{details['Name']}*")
        st.write(f"**Member since:** *{details['Date']}*")
        st.write(f"**Height:** *{st.session_state.user_details['Height']}* cm")
        st.write(f"**Age:** *{st.session_state.user_details['Age']}* years")
    with col2:
        st.write(f"**Username:** *{details['Username']}*")
        st.write(f"**Email:** *{details['Email']}*")
        st.write(f"**Weight:** *{st.session_state.user_details['Weight']}* kg")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Logout", on_click=logout, key="logout_btn", icon="🚪", type="secondary", help="Logout from your account")
    
    # Danger Zone Section
    st.divider()
    st.html("<h2 style='color: red;'>⚠️ Danger Zone</h2>")
    
    with st.container(border=True):
        st.write("**Permanent Account Deletion**")
        st.caption("⚠️ This action cannot be undone. All your data will be permanently removed from our systems.")
        
        # Confirmation workflow
        confirm = st.checkbox("I understand this action is irreversible")
        if confirm:
            if st.button("🗑️ Delete My Account", 
                        type="primary",
                        help="Permanently delete your account and all associated data"):
                if delete_user(st.session_state.user_email):
                    st.success("Account deleted successfully. Redirecting...")
                    time.sleep(1.5)
                    # Clear all session states
                    st.session_state.clear()
                    st.rerun()
                else:
                    st.error("Failed to delete account. Please try again or contact support.")