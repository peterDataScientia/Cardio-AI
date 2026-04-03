# app.py (Streamlit-ready, interactive batch display)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder

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
.badge-active {
    background-color: #2ca02c;
    color: white;
    padding: 3px 8px;
    border-radius: 5px;
}
.badge-inactive {
    background-color: #d62728;
    color: white;
    padding: 3px 8px;
    border-radius: 5px;
}
.badge-ad-ok {
    background-color: #1f77b4;
    color: white;
    padding: 3px 8px;
    border-radius: 5px;
}
.badge-ad-out {
    background-color: #ff7f0e;
    color: white;
    padding: 3px 8px;
    border-radius: 5px;
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
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
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
    ad_status = "Inside AD" if sim_score >= threshold else "Outside AD"
    pred_label = "Active" if prediction == 1 else "Inactive"
    return {
        "SMILES": smiles,
        "Prediction": pred_label,
        "Confidence": confidence,
        "AD_Status": ad_status
    }

def predict_batch(smiles_list, compound_ids=None):
    results = []
    for i, smi in enumerate(smiles_list):
        res = predict_single(smi)
        if res:
            if compound_ids is not None:
                res["Compound_ID"] = compound_ids[i]
            results.append(res)
    df = pd.DataFrame(results)
    
    # Add HTML badges
    df["Prediction_Badge"] = df["Prediction"].apply(
        lambda x: f"<span class='badge-active'>{x}</span>" if x=="Active" else f"<span class='badge-inactive'>{x}</span>"
    )
    df["AD_Badge"] = df["AD_Status"].apply(
        lambda x: f"<span class='badge-ad-ok'>{x}</span>" if x=="Inside AD" else f"<span class='badge-ad-out'>{x}</span>"
    )
    return df

# -------------------------
# Input Tabs
# -------------------------
tab1, tab2 = st.tabs(["Single Compound", "Batch Prediction"])

# ---------- Single Compound ----------
with tab1:
    smiles_input = st.text_input("Enter SMILES string:", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    if st.button("Predict Single Compound", key="single"):
        result = predict_single(smiles_input)
        if result is None:
            st.error("❌ Invalid SMILES string")
        else:
            st.markdown("### Prediction Results")
            st.write(f"**Prediction:** {result['Prediction']}")
            st.write(f"**Confidence (%):** {result['Confidence']:.2f}")
            st.write(f"**Applicability Domain:** {result['AD_Status']}")

# ---------- Batch Prediction ----------
with tab2:
    uploaded_file = st.file_uploader("Upload CSV/TXT with Compound_ID and SMILES", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_csv(uploaded_file, sep="\t", header=0)
            
            smiles_list = df.iloc[:, 1].tolist()
            compound_ids = df.iloc[:, 0].tolist()
            
            results_df = predict_batch(smiles_list, compound_ids)
            
            # Display interactive table with st-aggrid
            st.markdown("### Batch Prediction Results (Interactive)")
            gb = GridOptionsBuilder.from_dataframe(results_df[["Compound_ID","Prediction_Badge","AD_Badge","Confidence"]])
            gb.configure_column("Prediction_Badge", cellRenderer='html')
            gb.configure_column("AD_Badge", cellRenderer='html')
            gridOptions = gb.build()
            AgGrid(results_df[["Compound_ID","Prediction_Badge","AD_Badge","Confidence"]], gridOptions=gridOptions, enable_enterprise_modules=False)
            
            # Download CSV
            csv_buffer = BytesIO()
            results_df.drop(columns=["Prediction_Badge","AD_Badge"]).to_csv(csv_buffer, index=False)
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
    "Random Forest model trained on Morgan fingerprints (radius=2, 1024-bit) | "
    "Applicability domain based on Tanimoto similarity | "
    "UDSM RIW 2026 Showcase"
)