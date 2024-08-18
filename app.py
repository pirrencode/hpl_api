import streamlit as st
import pandas as pd

# Load data files (assuming the CSVs are in the same directory)
files = {
    "economic_viability": "economic_viability_dynamics.csv",
    "reliability": "reliability_curve_dynamics.csv",
    "scalability": "scalability_curve_dynamics.csv",
    "infrastructure_integration": "infrastructure_integration_dynamics.csv",
    # Add other criteria files here
}

def serve_csv(criterion):
    if criterion in files:
        data = pd.read_csv(files[criterion])
        return data.to_csv(index=False)
    else:
        return "Criterion not found!"

# Streamlit UI
st.title("Criteria Data API")

# Check if the URL has the 'criterion' parameter
query_params = st.experimental_get_query_params()
criterion = query_params.get("criterion", [None])[0]
raw = query_params.get("raw", ["false"])[0].lower()

if criterion:
    if raw == "true":
        # Serve raw CSV data without rendering the Streamlit UI
        csv_data = serve_csv(criterion)
        st.write(csv_data)
    else:
        # Display UI for downloading the CSV
        st.write(f"Fetching data for: **{criterion.replace('_', ' ').title()}**")
        csv_data = serve_csv(criterion)
        st.download_button(
            label=f"Download {criterion.replace('_', ' ').title()} CSV",
            data=csv_data,
            file_name=f"{criterion}.csv",
            mime="text/csv"
        )
else:
    st.write("Please select a criterion from the dropdown below.")
    selected_criterion = st.selectbox("Select Criterion", list(files.keys()))
    if st.button("Get Data"):
        st.write(f"Fetching data for: **{selected_criterion.replace('_', ' ').title()}**")
        csv_data = serve_csv(selected_criterion)
        st.download_button(
            label=f"Download {selected_criterion.replace('_', ' ').title()} CSV",
            data=csv_data,
            file_name=f"{selected_criterion}.csv",
            mime="text/csv"
        )

# Instructions for the user
st.write("### API Access")
st.write("You can access the CSV data by adding a criterion query parameter to the URL.")
st.write("Example: ")
st.code("https://your-app-name.streamlit.app/?criterion=economic_viability&raw=true", language="markdown")

