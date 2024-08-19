import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from snowflake.snowpark import Session
import os
import tempfile

# Set up Snowflake connection parameters
connection_parameters = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "role": st.secrets["snowflake"]["role"],
    "warehouse": st.secrets["snowflake"]["warehouse"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"]
}

# Function to generate Safety criterion data
def generate_safety_data():
    time_steps = 100
    risk_scores = np.random.rand(time_steps, 5)  # Simulate risk scores for 5 components
    min_risks = risk_scores.min(axis=0)
    max_risks = risk_scores.max(axis=0)

    # Calculate Safety Criterion
    safety_criterion = np.zeros(time_steps)
    for t in range(time_steps):
        safety_criterion[t] = 1 / np.sum((risk_scores[t, :] - min_risks) / (max_risks - min_risks))

    # Create a DataFrame
    df = pd.DataFrame({
        "TIME": np.arange(time_steps),
        "RISK_SCORE_COMPONENT_1": risk_scores[:, 0],
        "RISK_SCORE_COMPONENT_2": risk_scores[:, 1],
        "RISK_SCORE_COMPONENT_3": risk_scores[:, 2],
        "RISK_SCORE_COMPONENT_4": risk_scores[:, 3],
        "RISK_SCORE_COMPONENT_5": risk_scores[:, 4],
        "SAFETY_CRITERION": safety_criterion
    })
    
    return df

def save_to_snowflake(df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        csv_file = tmp.name
        df.to_csv(csv_file, index=False)

    session = Session.builder.configs(connection_parameters).create()

    # Optional: Clear the table before inserting new data
    session.sql("DELETE FROM SAFETY_CRITERION_RESULTS").collect()

    stage_name = "my_temp_stage"
    file_name = os.path.basename(csv_file)

    session.sql(f"CREATE TEMPORARY STAGE IF NOT EXISTS {stage_name}").collect()

    session.file.put(f"file://{csv_file}", f"@{stage_name}")

    session.sql(f"""
        COPY INTO SAFETY_CRITERION_RESULTS
        FROM @{stage_name}/{file_name}
        FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1)
    """).collect()

    session.sql(f"REMOVE @{stage_name}/{file_name}").collect()

    session.close()
    os.remove(csv_file)

# Function to load data from Snowflake
def load_from_snowflake():
    session = Session.builder.configs(connection_parameters).create()
    df = session.table("SAFETY_CRITERION_RESULTS").to_pandas()
    session.close()
    return df

# Streamlit UI
st.title("Criteria Data Generator, Viewer, and Power BI Dashboard")

# Criterion selection
criterion = st.selectbox("Select Criterion", ["Safety"])

if st.button("Generate and Save Data"):
    if criterion == "Safety":
        st.write("Generating data for Safety criterion...")
        df = generate_safety_data()
        st.write("Data generated successfully!")
        st.dataframe(df.head())
        st.write("Saving data to Snowflake...")
        save_to_snowflake(df)
        st.write("Data saved to Snowflake successfully!")

# Button to view data from Snowflake
if st.button("View Data from Snowflake"):
    st.write("Loading data from Snowflake...")
    df = load_from_snowflake()
    st.write("Data loaded successfully!")
    st.dataframe(df)

# Button to upload a CSV file and save to Snowflake
uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.write("CSV file loaded successfully!")
    st.dataframe(df_uploaded.head())

    if st.button("Save Uploaded CSV Data to Snowflake"):
        st.write("Saving uploaded data to Snowflake...")
        save_to_snowflake(df_uploaded)
        st.write("Uploaded data saved to Snowflake successfully!")

# New Feature: Visualize Safety Parameters
if st.button("Visualize Safety Parameters"):
    st.write("Loading data for visualization...")
    df = load_from_snowflake()

    st.write("Visualizing Safety Parameters...")
    # Plot each risk score component against time
    for component in ["RISK_SCORE_COMPONENT_1", "RISK_SCORE_COMPONENT_2", "RISK_SCORE_COMPONENT_3", 
                      "RISK_SCORE_COMPONENT_4", "RISK_SCORE_COMPONENT_5", "SAFETY_CRITERION"]:
        fig = px.line(df, x="TIME", y=component, title=f"{component} over Time")
        st.plotly_chart(fig)