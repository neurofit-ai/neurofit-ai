# check_db_access.py
import streamlit as st

# check_db_access.py (modified)
def check_db_access(key_suffix=""):  # 🆕 Add parameter
    """Silent database access verification using Streamlit Secrets"""
    if "db_password" not in st.secrets:
        st.error("Database password not configured in secrets.toml!")
        st.stop()
    
    DB_PASSWORD = st.secrets["db_password"]  # 🆕 Use secrets management

    # Auto-grant access if secret exists (no UI elements)
    st.session_state.db_access_granted = True

    if not st.session_state.db_access_granted:
        user_input_password = st.text_input(
            "🔒 Enter Database Access Password", 
            type="password",
            help="Only authorized users can access.",
            key=f"db_password_input_{key_suffix}"  # 🆕 Unique key based on suffix
        )
        # if st.button("Unlock Database", key=f"unlock_db_{key_suffix}"):  # 🆕 Unique button key
        if user_input_password == DB_PASSWORD:
            st.session_state.db_access_granted = True
            st.success("✅ Access granted!")
            st.rerun()
        else:
            st.error("❌ Incorrect database password!")
            st.stop()
    else:
        pass  # Already granted