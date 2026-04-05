from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_drawable_canvas import st_canvas
import pandas as pd

with tab1:
    st.subheader("🔬 Single Compound Prediction")

    input_option = st.radio("Select input method:", ("Paste SMILES", "Draw Molecule"))

    mol = None
    smiles_input = None

    if input_option == "Paste SMILES":
        smiles_input = st.text_input("Enter SMILES string:", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.image(Draw.MolToImage(mol, size=(300,300)))
            else:
                st.error("❌ Invalid SMILES string")

    else:  # Draw Molecule
        st.info("Draw molecule on canvas or paste SMILES")
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

        smiles_input = st.text_input("Or paste SMILES here:", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")

        # If user pasted SMILES, convert to molecule
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
        # Optionally: we can try to auto-detect drawing as image → molecule later

        if mol:
            st.image(Draw.MolToImage(mol, size=(300,300)))

    # Predict button
    if st.button("Predict Single Compound"):
        if mol is None:
            st.error("❌ Provide valid SMILES or drawn molecule")
        else:
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