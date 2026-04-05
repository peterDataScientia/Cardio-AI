import streamlit as st
import os

st.set_page_config(page_title="TNF-α Predictor", page_icon="🧪", layout="wide")

# Sidebar branding
if os.path.exists("LOGO.png"):
    st.sidebar.image("LOGO.png")

st.sidebar.title("Navigation")
st.sidebar.info("Select a page above.")

# Main page
st.markdown("<h1 style='text-align:center;'>🧪 TNF-α Inhibitor Prediction Platform</h1>", unsafe_allow_html=True)

st.markdown("""
### Welcome 👋

This AI-powered platform allows you to:

- 🔬 Predict TNF-α inhibition for a single compound  
- 📊 Perform batch screening  
- 📘 Understand model methodology  

---

### Model Details
- Algorithm: Random Forest  
- Features: Morgan Fingerprints (radius = 2, 1024 bits)  
- Applicability Domain: Tanimoto similarity ≥ 0.30  
""")