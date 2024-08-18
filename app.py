import streamlit as st
import pandas as pd
import os

# Load data files (assuming the CSVs are in the same directory)
files = {
    "Economic Viability": "economic_viability_dynamics.csv",
    "Reliability": "reliability_curve_dynamics.csv",
    "Scalability": "scalability_curve_dynamics.csv",
    "Infrastructure Integration": "infrastructure_integration_dynamics.csv",
    # Add other criteria files here
}

# Streamlit UI
st.title("Criteria Data API")

# Dropdown to select which criterion to fetch
selected_criterion = st.selectbox("Select Criterion", list(files.keys()))

# Serve the selected CSV data
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

# Streamlit provides a way to use the API
@st.experimental_memo
def get_csv(criterion):
    if criterion in files:
        return pd.read_csv(files[criterion])
    else:
        return None

st.write("### API Access")
st.write(f"To access the data via API, use the URL format:")
st.code(f"Streamlit_App_URL/?criterion=<name_of_criterion>", language="python")
