import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import joblib
from io import BytesIO
import os

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Cardio-AI Predictor",
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
# Sidebar
# -------------------------
if os.path.exists("LOGO.png"):
    st.sidebar.image("LOGO.png", width=200)

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
# Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧪 Cardio-AI Predictor: TNF-α Inhibitors</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Random Forest | Morgan Fingerprints | Applicability Domain</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Load Model
# -------------------------
@st.cache_data
def load_model():
    model = joblib.load("random_forest_model.pkl")
    train_fps = np.load("train_fingerprints.npy")
    return model, train_fps

model, train_fps = load_model()
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
        "Mol": mol,
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
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

# ---------- Tab 1: Single Prediction ----------
with tab1:
    st.subheader("🔬 Single Compound Prediction")
    smiles_input = st.text_input("Paste SMILES string:", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    if st.button("Predict Single Compound", key="single_predict"):
        if smiles_input.strip() == "":
            st.warning("Please enter a SMILES string!")
        else:
            result = predict_single(smiles_input)
            if result is None:
                st.error("❌ Invalid SMILES string")
            else:
                st.subheader("📊 Prediction Results")
                st.metric("Predicted Class", result['Prediction'])
                st.metric("Confidence (%)", f"{result['Confidence (%)']:.2f}")
                st.metric("Applicability Domain", result['AD_Status'])
                st.write(f"Max Tanimoto Similarity: {result['Max_Tanimoto']:.2f}")

                # Display molecule
                st.subheader("Molecule")
                mol_img = Draw.MolToImage(result['Mol'], size=(300,300))
                st.image(mol_img)

                # Probability bar chart
                prob_df = pd.DataFrame({
                    "Class": ["Inactive", "Active"],
                    "Probability": result['Probabilities']
                })
                st.bar_chart(prob_df.set_index("Class"))

# ---------- Tab 2: Batch Prediction ----------
with tab2:
    st.subheader("📊 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV/TXT with Compound_ID and SMILES", type=["csv","txt"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_csv(uploaded_file, sep="\t", header=0)
            smiles_list = df.iloc[:,1].tolist()
            results_df = predict_batch(smiles_list)
            results_df.insert(0, "Compound_ID", df.iloc[:,0])

            st.dataframe(results_df.drop(columns=["Probabilities"]))

            csv_buffer = BytesIO()
            results_df.drop(columns=["Probabilities"]).to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button("⬇️ Download CSV", data=csv_buffer, file_name="tnf_alpha_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ---------- Tab 3: Model Info ----------
with tab3:
    st.subheader("📘 Model Information")
    try:
        st.write("Random Forest Model Loaded ✅")
        st.write(f"Training fingerprints shape: {train_fps.shape}")
    except:
        st.error("❌ Could not load model/fingerprints")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div style="position: fixed; bottom: 8px; width: 100%; text-align: center; margin-bottom: 60px;">
    Developed by Peter et al. (2026) | Chemistry with AI
</div>
""", unsafe_allow_html=True)