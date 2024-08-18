import streamlit as st
import numpy as np
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

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

import tempfile

def save_to_snowflake_csv(df):
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        csv_file = tmp.name
        df.to_csv(csv_file, index=False)
        
        session = Session.builder.configs(connection_parameters).create()
        
        # Use PUT command to upload the CSV file
        session.file.put(local_path=csv_file, stage_location="@~")
        session.sql(f"COPY INTO SAFETY_CRITERION_RESULTS FROM @~/{csv_file.split('/')[-1]} FILE_FORMAT = (type = csv field_optionally_enclosed_by='\"')").collect()
        
        session.close()

import tempfile

def save_to_snowflake_csv(df):
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        csv_file = tmp.name
        df.to_csv(csv_file, index=False)

    session = Session.builder.configs(connection_parameters).create()

    # Define the stage name and file path
    stage_name = "@my_stage"
    file_name = os.path.basename(csv_file)

    # Create a stage (temporary storage in Snowflake) if it doesn't exist
    session.sql(f"CREATE TEMPORARY STAGE IF NOT EXISTS {stage_name}").collect()

    # Upload the CSV file to the stage
    session.file.put(f"file://{csv_file}", stage_name)

    # Copy the data from the stage into the Snowflake table
    session.sql(f"""
        COPY INTO SAFETY_CRITERION_RESULTS
        FROM {stage_name}/{file_name}
        FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1)
    """).collect()

    # Clean up: Remove the file from the stage
    session.sql(f"REMOVE {stage_name}/{file_name}").collect()

    # Close the session
    session.close()

    # Optionally remove the temporary file from the local filesystem
    os.remove(csv_file)



# Function to save data to Snowflake
def save_to_snowflake(df):
    # Create a Snowflake session
    session = Session.builder.configs(connection_parameters).create()

    # Write the DataFrame to the Snowflake table, overwriting existing data
    session.write_pandas(df, "SAFETY_CRITERION_RESULTS", mode="overwrite")
    
    # Close the session
    session.close() 


# Function to retrieve data from Snowflake
def load_from_snowflake():
    # Create a Snowflake session
    session = Session.builder.configs(connection_parameters).create()

    # Retrieve the data from Snowflake
    df = session.table("SAFETY_CRITERION_RESULTS").to_pandas()

    # Close the session
    session.close()

    return df

# Streamlit UI
st.title("Criteria Data Generator and Viewer")

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
        save_to_snowflake_csv(df)
        st.write("Data saved to Snowflake successfully!")

# Add a button to view data from Snowflake
if st.button("View Data from Snowflake"):
    st.write("Loading data from Snowflake...")
    
    # Load data from Snowflake
    df = load_from_snowflake()
    st.write("Data loaded successfully!")

    # Show the data
    st.dataframe(df)

