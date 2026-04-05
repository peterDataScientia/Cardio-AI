import streamlit as st

st.title("📘 Model Information")

st.markdown("""
### Model Overview
- Algorithm: Random Forest
- Features: Morgan Fingerprints (radius = 2, 1024 bits)

### Applicability Domain (AD)
- Based on Tanimoto similarity
- Threshold: 0.30

### Interpretation
- 🟢 Inside AD → Reliable prediction
- 🔴 Outside AD → Use with caution

### Application
Designed for screening TNF-α inhibitors for cardioprotective drug discovery.
""")