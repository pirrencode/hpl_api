import streamlit as st
import pandas as pd
import os

# Load data files (assuming the CSVs are in the same directory)
files = {
    "Economic Viability": "economic_viability_dynamics.csv",
    "Reliability": "reliability_curve_dynamics.csv",
    "Scalability": "scalability_curve_dynamics.csv",
    "Infrastructure Integration": "infrastructure_integration_dynamics.csv",
    "Environmental Sustainability": "environmental_sustainability_dynamics.csv",
    "Governance Compliance": "governance_compliance_simulation_2.csv",
}

def serve_data(criterion):
    if criterion in files:
        return pd.read_csv(files[criterion]).to_csv(index=False)
    else:
        return "Criterion not found", 404

st.title("Criteria Data API")

# Handle query params
criterion = st.experimental_get_query_params().get("criterion", [None])[0]

if criterion:
    # CSV data for the specified criterion
    data = serve_data(criterion)
    st.write(data)

# Dropdown to select which criterion to fetch for interactive use
selected_criterion = st.selectbox("Select Criterion", list(files.keys()))

if st.button("Get Data"):
    data = pd.read_csv(files[selected_criterion])
    st.write(data)
    # Convert DataFrame to CSV for download
    st.download_button(
        label="Download CSV",
        data=data.to_csv(index=False),
        file_name=f"{selected_criterion.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

st.write("### API Access")
st.write(f"To access the data via API, use the URL format:")
st.code(f"Streamlit_App_URL/?criterion=<name_of_criterion>", language="python")
