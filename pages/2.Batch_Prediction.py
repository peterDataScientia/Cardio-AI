import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from io import BytesIO

st.title("📊 Batch Prediction")

model = joblib.load("random_forest_model.pkl")
train_fps = np.load("train_fingerprints.npy")

radius = 2
n_bits = 1024
threshold = 0.30

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)

def tanimoto_similarity_numpy(fp, train_fps):
    fp_bool = fp.astype(bool).reshape(-1)
    train_bool = train_fps.astype(bool)
    intersection = np.logical_and(train_bool, fp_bool).sum(axis=1)
    union = np.logical_or(train_bool, fp_bool).sum(axis=1)
    return np.max(intersection / (union + 1e-8))

file = st.file_uploader("Upload CSV with Compound_ID and SMILES")

if file:
    df = pd.read_csv(file)

    results = []

    for _, row in df.iterrows():
        smi = row.iloc[1]

        try:
            fp = smiles_to_fp(smi)
            if fp is None:
                raise ValueError

            pred = model.predict(fp)[0]
            prob = model.predict_proba(fp)[0]
            conf = np.max(prob)*100
            sim = tanimoto_similarity_numpy(fp, train_fps)

            ad = "Inside AD" if sim >= threshold else "Outside AD"

            results.append([row.iloc[0], smi,
                            "Active" if pred==1 else "Inactive",
                            conf, sim, ad])

        except:
            results.append([row.iloc[0], smi, "Error", None, None, "Invalid"])

    result_df = pd.DataFrame(results, columns=[
        "Compound_ID", "SMILES", "Prediction",
        "Confidence (%)", "Similarity", "AD_Status"
    ])

    def highlight(val):
        if val == "Inside AD":
            return "background-color:#c6f7c6"
        elif val == "Outside AD":
            return "background-color:#f7c6c6"
        return ""

    st.dataframe(result_df.style.applymap(highlight, subset=["AD_Status"]))

    buffer = BytesIO()
    result_df.to_csv(buffer, index=False)

    st.download_button("⬇️ Download Results", buffer.getvalue(), "predictions.csv")