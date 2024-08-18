import streamlit as st
import pandas as pd
import os

# Load data files (assuming the CSVs are in the same directory)
files = {
    "economic_viability": "economic_viability_dynamics.csv",
    "reliability": "reliability_curve_dynamics.csv",
    "scalability": "scalability_curve_dynamics.csv",
    "infrastructure_integration": "infrastructure_integration_dynamics.csv",
    "environmental_sustainability": "environmental_sustainability_dynamics.csv",
    "governance_compliance": "governance_compliance_simulation_2.csv",
}

# Function to serve data with correct headers
def serve_data(criterion):
    if criterion in files:
        csv = pd.read_csv(files[criterion]).to_csv(index=False)
        st.experimental_set_query_params(criterion=criterion)
        st.write(csv)
        # Set the proper header to signal that this is a CSV file
        st.markdown(f'<a href="data:file/csv;base64,{csv}" download="{criterion.lower().replace(" ", "_")}.csv">Download {criterion} CSV</a>', unsafe_allow_html=True)
    else:
        return "Criterion not found", 404

# Streamlit UI
st.title("Criteria Data API")

# Handle query params
criterion = st.experimental_get_query_params().get("criterion", [None])[0]

if criterion:
    st.header(f"Data for {criterion}")
    serve_data(criterion)

# Dropdown to select which criterion to fetch for interactive use
selected_criterion = st.selectbox("Select Criterion", list(files.keys()))

if st.button("Get Data"):
    serve_data(selected_criterion)
