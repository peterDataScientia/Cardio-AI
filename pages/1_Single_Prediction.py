import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

st.set_page_config(page_title="Single Prediction", layout="wide")

st.title("🔬 Single Compound Prediction")

# Load model
model = joblib.load("random_forest_model.pkl")
train_fps = np.load("train_fingerprints.npy")

radius = 2
n_bits = 1024
threshold = 0.30

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
    return np.max(intersection / (union + 1e-8))

# Input
smiles = st.text_input("Enter SMILES:", "CC(=O)Oc1ccccc1C(=O)O")

if smiles:
    fp, mol = smiles_to_fp(smiles)

    if mol:
        st.image(Draw.MolToImage(mol, size=(300,300)))

        if st.button("Predict"):
            pred = model.predict(fp)[0]
            prob = model.predict_proba(fp)[0]
            conf = np.max(prob)*100
            sim = tanimoto_similarity_numpy(fp, train_fps)

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", "Active" if pred==1 else "Inactive")
            col2.metric("Confidence (%)", f"{conf:.2f}")
            col3.metric("Similarity", f"{sim:.2f}")

            # AD Highlight
            if sim >= threshold:
                st.markdown("<h3 style='color:green; text-align:center;'>🟢 Inside Applicability Domain</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color:red; text-align:center;'>🔴 Outside Applicability Domain</h3>", unsafe_allow_html=True)
                st.warning("⚠️ Prediction may be unreliable")

            st.bar_chart({
                "Inactive": prob[0],
                "Active": prob[1]
            })

    else:
        st.error("❌ Invalid SMILES")