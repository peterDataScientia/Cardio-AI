# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from io import BytesIO
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="TNF-α Inhibitor Predictor",
    page_icon="🧪",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; background-color: #f9f9f9; }
h1, h2, h3, h4 { color: #1f77b4; text-align: center; }
.stButton>button, .stDownloadButton>button {
    border-radius: 8px; height: 40px; font-weight: bold;
}
.stButton>button { background-color: #1f77b4; color: white; }
.stDownloadButton>button { background-color: #ff7f0e; color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header with Logo & Optional Developer Photo
# -------------------------
col1, col2, = st.columns([1,5])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=400, height=400)
with col2:
    st.markdown("<h1>🧪 TNF-α Inhibitor Prediction Platform</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>AI-Powered Bioactivity Classification<br>"
        "Random Forest | Morgan Fingerprints | Applicability Domain</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("About")
st.sidebar.info(
    "Predict TNF-α inhibitory activity from SMILES input "
    "and assess reliability using molecular similarity."
)
st.sidebar.markdown("### 👨‍🔬 Developer")
if os.path.exists("DEVELOPER.jpeg"):
    st.sidebar.image("DEVELOPER.jpeg", width=150, caption="Peter et al. (2026)")
else:
    st.sidebar.info("Peter et al. (2026)")

# -------------------------
# Load Model & Fingerprints
# -------------------------
model = joblib.load("random_forest_model.pkl")
train_fps = np.load("train_fingerprints.npy")  # shape: (n_samples, 1024)
radius = 2
n_bits = 1024

# -------------------------
# Functions
# -------------------------
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1), mol

def tanimoto_similarity_numpy(fp, train_fps):
    """Compute max Tanimoto similarity between input FP and training FPs"""
    fp_bool = fp.astype(bool).reshape(-1)
    train_bool = train_fps.astype(bool)
    intersection = np.logical_and(train_bool, fp_bool).sum(axis=1)
    union = np.logical_or(train_bool, fp_bool).sum(axis=1)
    similarity = intersection / (union + 1e-8)
    return np.max(similarity)

def predict_single(smiles):
    fp, mol = smiles_to_fp(smiles)
    if fp is None:
        return None
    prediction = model.predict(fp)[0]
    probabilities = model.predict_proba(fp)[0]
    confidence = np.max(probabilities) * 100
    sim_score = tanimoto_similarity_numpy(fp, train_fps)
    threshold = 0.30
    ad_status = "✅ Inside Applicability Domain" if sim_score >= threshold else "⚠️ Outside Applicability Domain"
    class_labels = {0: "Inactive", 1: "Active"}
    pred_label = class_labels.get(prediction, str(prediction))
    return {
        "SMILES": smiles,
        "Prediction": pred_label,
        "Confidence (%)": confidence,
        "AD_Status": ad_status,
        "Max_Tanimoto": sim_score,
        "Probabilities": probabilities
    }

def predict_batch(smiles_list):
    results = []
    for smi in smiles_list:
        res = predict_single(smi)
        if res:
            results.append(res)
    return pd.DataFrame(results)

# -------------------------
# Input Tabs
# -------------------------
st.subheader("🔬 Input Options")
tab1, tab2 = st.tabs(["Single Compound", "Batch Prediction"])

# ---------- Single Compound ----------
with tab1:
    smiles_input = st.text_input("Enter SMILES string:", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    if st.button("Predict Single Compound"):
        result = predict_single(smiles_input)
        if result is None:
            st.error("❌ Invalid SMILES string")
        else:
            st.markdown("### 📊 Prediction Results")
            st.metric("Predicted Class", result['Prediction'])
            st.metric("Confidence (%)", f"{result['Confidence (%)']:.2f}")
            st.metric("Applicability Domain", result['AD_Status'])
            st.write(f"Max Tanimoto Similarity: {result['Max_Tanimoto']:.2f}")
            
            # Probability bar chart
            prob_df = pd.DataFrame({
                "Class": ["Inactive", "Active"],
                "Probability": result['Probabilities']
            })
            st.bar_chart(prob_df.set_index("Class"))

# ---------- Batch Prediction ----------
with tab2:
    uploaded_file = st.file_uploader("Upload CSV/TXT with Compound_ID and SMILES", type=["csv", "txt"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_csv(uploaded_file, sep="\t", header=0)
            smiles_list = df.iloc[:,1].tolist()
            results_df = predict_batch(smiles_list)
            results_df.insert(0, "Compound_ID", df.iloc[:,0])
            
            st.markdown("### 📊 Batch Prediction Results")
            st.dataframe(results_df.drop(columns=["Probabilities"]))
            
            csv_buffer = BytesIO()
            results_df.drop(columns=["Probabilities"]).to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button("⬇️ Download CSV", data=csv_buffer, file_name="tnf_alpha_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="position: fixed; bottom: 8px; width: 100%; text-align: center; margin-bottom: 60px;">
    Developed by Peter et al. (2026) | UDSM RIW 2026 Showcase
</div>
""", unsafe_allow_html=True)