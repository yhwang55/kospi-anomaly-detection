import streamlit as st

# Configure the page
st.set_page_config(
    page_title="KOSPI Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for customization
st.markdown("""
<style>
body {
    background-color: #F0F2F5;
    font-family: 'Arial', sans-serif;
}

h1, h2, h3 {
    color: #333;
}

.sidebar .sidebar-content {
    background-color: #F7F9FC;
}
</style>
"", unsafe_allow_html=True)

# Your remaining code continues here...