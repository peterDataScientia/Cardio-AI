# app.py (Streamlit-ready, headless)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem

from io import BytesIO

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="TNF-α Inhibitor Predictor",
    page_icon="🧪",
    layout="wide"
)

# -------------------------
# Custom CSS for Professional Look
# -------------------------
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f5f5f5;
}
h1, h2, h3, h4 {
    color: #1f77b4;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    height: 40px;
    font-weight: bold;
}
.stDownloadButton>button {
    background-color: #ff7f0e;
    color: white;
    border-radius: 8px;
    height: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.title("🧪 TNF-α Inhibitor Prediction Platform")
st.markdown(
    "**AI-Powered Bioactivity Classification**  \n"
    "*Random Forest | Morgan Fingerprints | Applicability Domain*  \n"
    "*Developed by Peter et al. (2026)*"
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
st.sidebar.info("Peter et al. (2026)")

# -------------------------
# Load Model + Training FP
# -------------------------
model = joblib.load("random_forest_model.pkl")
train_fps = np.load("train_fingerprints.npy")  # shape (n,1024)
radius = 2
n_bits = 1024

# -------------------------
# Functions
# -------------------------
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1), mol

def tanimoto_similarity_numpy(fp, train_fps):
    fp = fp.astype(bool)
    train_fps = train_fps.astype(bool)
    intersection = np.logical_and(train_fps, fp).sum(axis=1)
    union = np.logical_or(train_fps, fp).sum(axis=1)
    similarity = intersection / (union + 1e-8)
    return np.max(similarity)

def predict_single(smiles):
    fp, mol = smiles_to_fp(smiles)
    if fp is None:
        return None
    prediction = model.predict(fp)[0]
    probabilities = model.predict_proba(fp)[0]
    confidence = np.max(probabilities) * 100
    sim_score = tanimoto_similarity_numpy(fp.flatten(), train_fps)
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
        "MolObj": mol,
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
    smiles_input = st.text_input(
        "Enter SMILES string:",
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
    )
    if st.button("Predict Single Compound", key="single"):
        result = predict_single(smiles_input)
        if result is None:
            st.error("❌ Invalid SMILES string")
        else:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            st.metric("Predicted Class", result['Prediction'])
            st.metric("Confidence (%)", f"{result['Confidence (%)']:.2f}")
            st.metric("Applicability Domain", result['AD_Status'])
            st.write(f"Max Tanimoto Similarity: {result['Max_Tanimoto']:.2f}")
            st.write("### Probability Distribution")
            prob_dict = {"Inactive": result['Probabilities'][0], "Active": result['Probabilities'][1]}
            st.bar_chart(pd.DataFrame(prob_dict, index=[0]))

# ---------- Batch Prediction ----------
with tab2:
    uploaded_file = st.file_uploader("Upload CSV/TXT with Compound_ID and SMILES", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            # Read file with header
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_csv(uploaded_file, sep="\t", header=0)
            
            # Use second column as SMILES
            smiles_list = df.iloc[:, 1].tolist()
            
            # Predict
            results_df = predict_batch(smiles_list)
            
            # Add Compound_ID back to results
            results_df.insert(0, "Compound_ID", df.iloc[:, 0])
            
            st.markdown("---")
            st.subheader("📊 Batch Prediction Results")
            st.dataframe(results_df.drop(columns=["MolObj", "Probabilities"]))
            
            # Download button
            csv_buffer = BytesIO()
            results_df.drop(columns=["MolObj", "Probabilities"]).to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_buffer,
                file_name="tnf_alpha_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "Developed by Peter et al. (2026) | "
    "Random Forest model trained on Morgan fingerprints (radius = 2, 1024-bit) | "
    "Applicability domain based on Tanimoto similarity | "
    "UDSM RIW 2026 Showcase"
)