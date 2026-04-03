# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
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
# CSS Styling
# -------------------------
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; background-color: #f9f9f9; }
h1, h2, h3 { color: #1f77b4; text-align: center; }
.stButton>button, .stDownloadButton>button {
    border-radius: 8px; height: 40px; font-weight: bold;
}
.stButton>button { background-color: #1f77b4; color: white; }
.stDownloadButton>button { background-color: #ff7f0e; color: white; }
.badge-active {background-color: #28a745; color: white; padding: 5px 12px; border-radius: 5px;}
.badge-inactive {background-color: #dc3545; color: white; padding: 5px 12px; border-radius: 5px;}
.badge-ad-ok {background-color: #17a2b8; color: white; padding: 5px 12px; border-radius: 5px;}
.badge-ad-out {background-color: #ffc107; color: black; padding: 5px 12px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("LOGO.png"):
        st.image("LOGO.png", width=120)
with col2:
    st.markdown("<h1>🧪 TNF-α Inhibitor Prediction Platform</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>AI-Powered Bioactivity Classification<br>"
        "Random Forest | Morgan Fingerprints | Applicability Domain<br>"
        "Developed by Peter et al. (2026)</p>", unsafe_allow_html=True
    )

st.markdown("---")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("About")
st.sidebar.info(
    "Predict TNF-α inhibitory activity from SMILES input and assess reliability "
    "using molecular similarity. Supports single compounds and batch CSV upload."
)
st.sidebar.markdown("### 👨‍🔬 Developer")
st.sidebar.info("Peter et al. (2026)")

# -------------------------
# Load Model + Training FP
# -------------------------
model = joblib.load("random_forest_model.pkl")
train_fps = np.load("train_fingerprints.npy")
radius = 2
n_bits = 1024

# -------------------------
# Functions
# -------------------------
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)

def tanimoto_similarity_numpy(fp, train_fps):
    fp = fp.astype(bool)
    train_fps = train_fps.astype(bool)
    intersection = np.logical_and(train_fps, fp).sum(axis=1)
    union = np.logical_or(train_fps, fp).sum(axis=1)
    return np.max(intersection / (union + 1e-8))

def predict_single(smiles):
    fp = smiles_to_fp(smiles)
    if fp is None:
        return None
    pred = model.predict(fp)[0]
    prob = model.predict_proba(fp)[0]
    confidence = np.max(prob)*100
    sim_score = tanimoto_similarity_numpy(fp.flatten(), train_fps)
    threshold = 0.30
    ad_status = "Inside AD" if sim_score >= threshold else "Outside AD"
    return {
        "SMILES": smiles,
        "Prediction": pred,
        "Confidence": confidence,
        "AD_Status": ad_status,
        "Max_Tanimoto": sim_score,
        "Probabilities": prob
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
st.subheader("🔬 Prediction Modes")
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Exploratory Analysis"])

# ---------- Single ----------
with tab1:
    smiles = st.text_input("Enter SMILES string:", placeholder="CC(=O)Oc1ccccc1C(=O)O")
    if st.button("Predict Single Compound"):
        result = predict_single(smiles)
        if result is None:
            st.error("❌ Invalid SMILES")
        else:
            # Styled badges
            pred_badge = f"<span class='badge-active'>Active</span>" if result['Prediction']==1 else f"<span class='badge-inactive'>Inactive</span>"
            ad_badge = f"<span class='badge-ad-ok'>Inside AD</span>" if result['AD_Status']=="Inside AD" else f"<span class='badge-ad-out'>Outside AD</span>"
            
            st.markdown("### Prediction Results")
            st.markdown(f"**Predicted Class:** {pred_badge}", unsafe_allow_html=True)
            st.markdown(f"**Applicability Domain:** {ad_badge}", unsafe_allow_html=True)
            st.metric("Confidence (%)", f"{result['Confidence']:.2f}")
            st.write(f"Max Tanimoto Similarity: {result['Max_Tanimoto']:.2f}")
            
            st.bar_chart(pd.DataFrame({
                "Inactive": [result['Probabilities'][0]],
                "Active": [result['Probabilities'][1]]
            }))

# ---------- Batch ----------
with tab2:
    uploaded_file = st.file_uploader("Upload CSV/TXT with Compound_ID and SMILES", type=["csv","txt"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_csv(uploaded_file, sep="\t")
        smiles_list = df.iloc[:,1].tolist()
        results_df = predict_batch(smiles_list)
        results_df.insert(0,"Compound_ID",df.iloc[:,0])
        
        # Interactive slider to filter by confidence
        min_conf = st.slider("Minimum Confidence (%) for display", 0, 100, 0)
        filtered = results_df[results_df['Confidence'] >= min_conf]
        
        def badge(val, ad):
            return f"<span class='badge-active'>Active</span>" if val==1 else f"<span class='badge-inactive'>Inactive</span>", \
                   f"<span class='badge-ad-ok'>Inside AD</span>" if ad=="Inside AD" else f"<span class='badge-ad-out'>Outside AD</span>"
        
        filtered["Pred_Class"] = [badge(r,a)[0] for r,a in zip(filtered['Prediction'], filtered['AD_Status'])]
        filtered["AD_Badge"] = [badge(r,a)[1] for r,a in zip(filtered['Prediction'], filtered['AD_Status'])]
        
        st.markdown("### Batch Prediction Results")
        st.dataframe(filtered.drop(columns=["Probabilities","Prediction","AD_Status"]), use_container_width=True)
        
        # Download
        csv_buffer = BytesIO()
        filtered.drop(columns=["Probabilities"]).to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button("⬇️ Download Filtered CSV", data=csv_buffer, file_name="tnf_alpha_batch.csv", mime="text/csv")

# ---------- Exploratory Analysis ----------
with tab3:
    st.markdown("### Interactive Data Exploration")
    st.info("Upload batch SMILES CSV to see summary stats and visualize prediction distribution.")
    uploaded_file2 = st.file_uploader("Upload CSV for Analysis", type=["csv"], key="analysis")
    if uploaded_file2:
        df2 = pd.read_csv(uploaded_file2)
        st.write("**Summary Stats:**")
        st.write(df2.describe())
        if "Prediction" in df2.columns:
            st.bar_chart(df2['Prediction'].value_counts())
        if "Confidence" in df2.columns:
            st.line_chart(df2['Confidence'])

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Developed by Peter et al. (2026) | UDSM RIW 2026 Showcase</div>",
    unsafe_allow_html=True
)