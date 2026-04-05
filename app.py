# Input Tabs
st.subheader("🔬 Input Options")
tab1, tab2 = st.tabs(["Single Compound", "Batch Prediction"])

# ---------- Single Compound Tab ----------
with tab1:
    st.subheader("🔬 Single Compound Prediction")

    # Input method: Paste or Draw
    input_option = st.radio("Select input method:", ("Paste SMILES", "Draw Molecule"))

    mol = None
    smiles_input = None

    if input_option == "Paste SMILES":
        smiles_input = st.text_input("Enter SMILES string (mandatory):", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    
    else:  # Draw Molecule
        st.info("Draw molecule on canvas (visual only). SMILES is still mandatory for prediction.")
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # transparent
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

# ---------- Batch Prediction Tab (unchanged) ----------
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