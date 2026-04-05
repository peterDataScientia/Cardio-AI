# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from io import BytesIO
import os
from streamlit_drawable_canvas import st_canvas

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
# Sidebar: Logo + Developer Photo
# -------------------------
if os.path.exists("LOGO.png"):
    st.sidebar.image("LOGO.png", width=200, use_column_width=False)

st.sidebar.markdown("### 👨‍🔬 Developer")
if os.path.exists("DEVELOPER.jpeg"):
    st.sidebar.image("DEVELOPER.jpeg", width=150, caption="Peter et al. (2026)")
else:
    st.sidebar.info("Peter et al. (2026)")

st.sidebar.markdown("---")
st.sidebar.title("About")
st.sidebar.info(
    "Predict TNF-α inhibitory activity from SMILES input "
    "and assess reliability using molecular similarity."
)

# -------------------------
# Main Page Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧪 AI-Powered Digital Platform for Predicting Cardioprotective Drug Candidates Targeting TNF-α Inhibition</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'><br>"
    "Random Forest | Morgan Fingerprints | Applicability Domain</p>",
    unsafe_allow_html=True
)
st.markdown("---")

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

# ---------- Single Compound Tab ----------
with tab1:
    st.subheader("🔬 Single Compound Prediction")

    # Input method
    input_option = st.radio("Select input method:", ("Paste SMILES", "Draw Molecule"))

    mol = None
    smiles_input = None

    if input_option == "Paste SMILES":
        smiles_input = st.text_input("Enter SMILES string (mandatory):", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    
    else:  # Draw Molecule
        st.info("Draw molecule on canvas (visual only). SMILES is still mandatory for prediction.")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas"
        )
        smiles_input = st.text_input("Paste SMILES here (mandatory for prediction):", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")

    # Validate SMILES and display molecule
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            st.markdown("### 🧬 Molecule Structure")
            st.image(Draw.MolToImage(mol, size=(300, 300)))
        else:
            st.error("❌ Invalid SMILES string")
            mol = None

    # Predict button
    if st.button("Predict Single Compound"):
        if mol is None:
            st.error("❌ Please provide a valid SMILES string for prediction.")
        else:
            result = predict_single(smiles_input)
            if result is None:
                st.error("❌ Prediction failed. Check SMILES format.")
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

# ---------- Batch Prediction Tab ----------
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
    Developed by Peter et al. (2026) | Chemistry with AI
</div>
""", unsafe_allow_html=True)