import re
import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime
from check_db_access import check_db_access

def init_database():
    """Initialize database with CORRECT schema"""
    conn = sqlite3.connect("neurofit_users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            height REAL,
            weight REAL,
            age INTEGER,
            gender TEXT,
            date_joined TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def get_user_details(email):
    """Fetch user details SAFELY with schema validation"""
    conn = sqlite3.connect("neurofit_users.db")
    c = conn.cursor()
    try:
        c.execute('''
            SELECT name, email, username, height, weight, age, gender, date_joined 
            FROM users WHERE email = ?
        ''', (email.lower(),))
        result = c.fetchone()
        if not result:
            return None
        return {
            "Name": result[0],
            "Email": result[1],
            "Username": result[2],
            "Height": result[3],
            "Weight": result[4],
            "Age": result[5],
            "Gender": result[6],
            "Date": result[7]
        }
    except sqlite3.OperationalError as e:
        st.error("Database corruption detected. Delete neurofit_users.db and restart.")
        st.stop()
    finally:
        conn.close()

def delete_user(email):
    conn = sqlite3.connect("neurofit_users.db")
    c = conn.cursor()
    try:
        c.execute("DELETE FROM users WHERE email = ?", (email.lower(),))
        conn.commit()
        return c.rowcount > 0
    except Exception as e:
        st.error(f"Deletion error: {str(e)}")
        return False
    finally:
        conn.close()

def signup():
    """Secure signup with silent DB check"""
    check_db_access()  # Silent verification
    
    init_database()
    col1, col2, col3 = st.columns([1, 50, 2])
    with col2:
        with st.form("signup_form"):
            st.markdown("### 📝 Create New Account")

            col1, col2 = st.columns([1, 1])
            with col1:
                name = st.text_input("Name")
                username = st.text_input("Username")
                new_pass = st.text_input("New Password", type="password")
                height = st.number_input("Height (cm)")
                gender = st.selectbox("Gender", ("Male", "Female"))
            with col2:
                email = st.text_input("Email")
                age = st.number_input("Age", max_value=100)
                confirm_pass = st.text_input("Confirm Password", type="password")
                weight = st.number_input("Weight (kg)")

            if st.form_submit_button("Register", icon="📄"):
                if not username.strip():
                    st.error("Username is required.")
                    return
                if not name.strip():
                    st.error("Name is required.")
                    return
                if not is_valid_email(email):
                    st.error("Invalid email address.")
                    return
                if new_pass != confirm_pass:
                    st.error("Passwords do not match!")
                    return
                if len(new_pass) < 8:
                    st.error("Password must be at least 8 characters long.")
                    return

                with st.spinner("Creating account..."):
                    password_hash = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
                    join_date = datetime.now().strftime("%Y-%m-%d")

                    conn = sqlite3.connect("neurofit_users.db")
                    c = conn.cursor()
                    try:
                        c.execute('''
                            INSERT INTO users
                            (name, email, username, password_hash, height, weight, age, gender, date_joined)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                            (name.strip(), email.lower(), username.strip(), password_hash, height, weight, age, gender, join_date))
                        conn.commit()
                        st.session_state.user_details = get_user_details(email.lower())
                        st.success("✅ Account created!")
                        st.balloons()
                    except sqlite3.IntegrityError:
                        st.error("Email already exists! Try logging in.")
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
                    finally:
                        conn.close()