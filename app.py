import time
import base64
import streamlit as st
from login_page import login
from check_db_access import check_db_access  # 🆕 Add this

st.set_page_config(page_title="NeuroFit", page_icon=":runner:", layout="centered")

def set_bg_black():
    st.markdown(
        """
        <style>
        /* Main app background and text */
        .stApp {
            background: #252525;
        }
        
        /* All text elements */
        .stMarkdown, .stText, .stAlert, 
        .stSelectbox label, .stSlider label, .stRadio label,
        .stNumberInput label, .stTextInput label, .stTextArea label,
        h1, h2, h3, h4, h5, h6, p, div {
            color: white;
        }
        
        /* Buttons - text color */
        .stButton>button {
            color: white;
        }
        
        /* Dataframes */
        .stDataFrame {
            color: white;
            background-color: #252525;
        }
        
        /* Input widgets */
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background-color: #333333;
            color: white;
            border-color: #555555;
        }
        
        /* Password eye button - keep it default */
        .stTextInput button[data-testid="baseButton-headerNoPadding"] {
            background-color: transparent;
            color: var(--text-color);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def set_bg_white():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f1f1f1;
        }

        .stAppHeader {
            background-color: #f1f1f1;
        }
        /* Reset all text back to dark */
        .stMarkdown, .stText, .stAlert, .stBorder,
        h1, h2, h3, h4, h5, h6, p, div {
            color: black;
        }

        hr
        {
            color: black;
            background-color: #252525;
            /* border: 1px solid black; */
        }

        .stForm {
            border-color: black;
        }
        
        /* Form non-submit buttons (e.g., Sign Up button) */
        .stForm button:not(.stFormSubmitButton>button) {
            color: white;
            background-color: #4CAF50;
            border: 1px solid #4CAF50;
        }

        .st-b2
        {
            background-color: #252525;
        }

        .st-an
        {
            background-color: skyblue;
        }

        /* Regular buttons */
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border: 1px solid #4CAF50;
        }

        .stDownloadButton>button {
            color: white !important;
            background-color: #4CAF50;
            border: 1px solid #4CAF50;
        }

        /* Form submit buttons */
        .stFormSubmitButton>button {
            color: white;
            background-color: #4CAF50;
            border: 1px solid #4CAF50;
        }

        /* Hover effects */
        .stButton>button:hover,
        .stFormSubmitButton>button:hover,
        .stForm button:hover {
            background-color: #45a049;
            border: 1px solid #45a049;
        }
        
        /* Password eye button - keep it default */
        .stTextInput button[data-testid="baseButton-headerNoPadding"] {
            background-color: transparent;
            color: var(--text-color);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Create two columns
col1, col2 = st.columns([1, 16])

# Use col1 for the toggle button and col2 for the label
with col1:
    toggle = st.toggle("", value=st.session_state.dark_mode)

with col2:
    # Immediately update dark_mode in session
    st.session_state.dark_mode = toggle

    # Based on new dark_mode, set background and toggle_label
    if st.session_state.dark_mode:
        set_bg_black()
        toggle_label = "Turn on Light Mode"
    else:
        set_bg_white()
        toggle_label = "Turn on Dark Mode"

    # Display the label on the same row
    st.write(toggle_label)

st.title(":runner: NeuroFit")
st.markdown("## AI Platform To Improve Athletic Performance")
st.html("<hr>")

# 👇 Always check database access first
check_db_access()

login()