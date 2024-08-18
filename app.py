import streamlit as st
import numpy as np
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

# Set up Snowflake connection parameters
connection_parameters = {
    "account": "your_account_name",
    "user": "your_username",
    "password": "your_password",
    "role": "your_role",
    "warehouse": "your_warehouse",
    "database": "HPL_SYSTEM_DYNAMICS",
    "schema": "SYSTEM_DYNAMICS"
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

# Function to save data to Snowflake
def save_to_snowflake(df):
    # Create a Snowflake session
    session = Session.builder.configs(connection_parameters).create()

    # Write the DataFrame to the Snowflake table, overwriting existing data
    session.write_pandas(df, "SAFETY_CRITERION_RESULTS", mode="overwrite")
    
    # Close the session
    session.close()

# Streamlit UI
st.title("Criteria Data Generator")

# Criterion selection (extend this list as you add more criteria)
criterion = st.selectbox("Select Criterion", ["Safety"])

if st.button("Generate and Save Data"):
    if criterion == "Safety":
        st.write("Generating data for Safety criterion...")
        
        # Generate the data
        df = generate_safety_data()
        st.write("Data generated successfully!")

        # Show a preview of the data
        st.dataframe(df.head())

        # Save the data to Snowflake
        st.write("Saving data to Snowflake...")
        save_to_snowflake(df)
        st.write("Data saved to Snowflake successfully!")
