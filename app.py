import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import logging
from io import BytesIO
from snowflake.snowpark import Session
from criterion_factors_logic import generate_safety_data, generate_environmental_impact_data

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up Snowflake connection parameters (parameterized for reusability)
def get_snowflake_connection_params():
    return {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"]
    }

def calculate_cr_sfy():
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    # Load data from CR_SFY_SOURCE table
    df = session.table("CR_SFY_SOURCE").to_pandas()

    # Calculate CR_SFY for each time period
    epsilon = 1e-6  # Avoid division by zero
    cr_sfy = np.array([1 / np.sum((df.iloc[t]["RISK_SCORE"] - df.iloc[t]["MIN_RISK_SCORE"]) / 
                                  (df.iloc[t]["MAX_RISK_SCORE"] - df.iloc[t]["MIN_RISK_SCORE"] + epsilon))
                       for t in range(len(df))])

    # Create DataFrame with TIME and CR_SFY
    df_cr_sfy = pd.DataFrame({
        "TIME": df["TIME"],
        "CR_SFY": cr_sfy
    })

    # Save the calculated CR_SFY to CALC_CR_SF table in Snowflake
    session.write_pandas(df_cr_sfy, "CALC_CR_SF", mode="overwrite")
    session.close()

    return df_cr_sfy

# General function to load data from Snowflake
def load_data_from_snowflake(table_name):
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    df = session.table(table_name).to_pandas()
    session.close()
    return df

def save_data_to_snowflake(df, table_name):
    try:
        # Use BytesIO to create an in-memory CSV file
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset buffer to start

        # Establish a Snowflake session
        session = Session.builder.configs(get_snowflake_connection_params()).create()

        stage_name = "my_temp_stage"
        
        # Ensure the stage is created
        session.sql(f"CREATE TEMPORARY STAGE IF NOT EXISTS {stage_name}").collect()
        logging.info("Temporary stage created or already exists.")

        # Upload the in-memory file to the stage
        put_result = session.file.put_stream(csv_buffer, f"@{stage_name}/temp_file.csv")
        logging.info(f"PUT command result: {put_result}")

        # Load the data into the Snowflake table
        copy_result = session.sql(f"""
            COPY INTO {table_name}
            FROM @{stage_name}/temp_file.csv
            FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1)
        """).collect()
        logging.info(f"COPY command result: {copy_result}")

        # Clean up: remove the file from the stage
        session.sql(f"REMOVE @{stage_name}/temp_file.csv").collect()
        logging.info("Temporary file removed from stage.")

    except Exception as e:
        logging.error(f"Error saving data to Snowflake: {e}")
        st.error(f"An error occurred while saving data to Snowflake: {str(e)}")
    finally:
        if session:
            session.close()
        csv_buffer.close()

    st.success(f"Data successfully saved to {table_name} in Snowflake!")

# Function to handle homepage navigation
def render_homepage():
    st.title("HDME")
    st.subheader("v0.02-dev")
    st.write("""
        Welcome to the Hyperloop Project System Dynamics Dashboard. 
        This application allows you to upload, manage, and visualize data related to various criteria 
        of the Hyperloop Project's system dynamics.
    """)

    # Display clickable emojis
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀\n\nUpload Data to Ecosystem", use_container_width=True):
            st.session_state['page'] = 'upload_data'

    with col2:
        if st.button("📊\n\nHyperloop Project System Dynamics Project", use_container_width=True):
            st.session_state['page'] = 'visualizations'

# Function to handle the data upload and management page
def render_upload_data_page():
    st.title("Upload Data to Ecosystem")

    # Criterion selection
    criterion = st.selectbox("Select Criterion", ["Safety", "Environmental Impact"])

    table_mapping = {
        "Safety": "CR_SFY_SOURCE",
        "Environmental Impact": "CR_ENV_SOURCE"
    }

    generate_function_mapping = {
        "Safety": generate_safety_data,
        "Environmental Impact": generate_environmental_impact_data
    }

    selected_table = table_mapping.get(criterion, "CR_SFY_SOURCE")
    generate_function = generate_function_mapping.get(criterion, generate_safety_data)

    if st.button("Generate and Save Data"):
        df = generate_function()
        st.write(f"Data generated successfully for {criterion}!")
        st.dataframe(df.head())
        save_data_to_snowflake(df, selected_table)

    if st.button("Calculate Criterion and Save Data"):
        df_cr_sfy = calculate_cr_sfy()
        st.write("CR_SFY calculated and saved successfully!")
        st.dataframe(df_cr_sfy)

    if st.button("View Source Data from Snowflake"):
        df = load_data_from_snowflake(selected_table)
        st.write(f"Loading {criterion} data from Snowflake...")
        st.dataframe(df)

    if st.button("View Hyperloop System Dynamics Input Criterion"):
        df = load_data_from_snowflake("CALC_CR_SF")
        st.write(f"Loading Hyperloop System Dynamics Input Criterion from Snowflake...")
        st.dataframe(df)

    uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("CSV file loaded successfully!")
        st.dataframe(df_uploaded.head())
        if st.button("Save Uploaded CSV Data to Snowflake"):
            save_data_to_snowflake(df_uploaded, selected_table)

    if st.button("⬅️ Back"):
        st.session_state['page'] = 'home'

# Function to handle the visualizations page
def render_visualizations_page():
    st.title("Hyperloop Project System Dynamics Dashboard")
    
    if st.button("Visualize Safety Criterion"):
        df_source = load_data_from_snowflake("CR_SFY_SOURCE")
        df_summary = load_data_from_snowflake("HPL_SD_CRS")

        # Visualize source data
        for component in ["RISK_SCORE", "MIN_RISK_SCORE", "MAX_RISK_SCORE"]:
            fig = px.line(df_source, x="TIME", y=component, title=f"{component} over Time")
            st.plotly_chart(fig)

        # Visualize CR_SFY summary data
        fig = px.line(df_summary, x="TIME", y="CR_SFY", title="CR_SFY over Time")
        st.plotly_chart(fig)

    if st.button("Visualize Environmental Impact"):
        df_source = load_data_from_snowflake("CR_ENV_SOURCE")
        df_summary = load_data_from_snowflake("HPL_SD_CRS")

        # Visualize source data
        for component in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY"]:
            fig = px.line(df_source, x="TIME", y=component, title=f"{component} over Time")
            st.plotly_chart(fig)

        # Visualize CR_ENV summary data
        fig = px.line(df_summary, x="TIME", y="CR_ENV", title="CR_ENV over Time")
        st.plotly_chart(fig)

    if st.button("⬅️ Back"):
        st.session_state['page'] = 'home'

# Main application logic to switch between pages
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    render_homepage()
elif st.session_state['page'] == 'upload_data':
    render_upload_data_page()
elif st.session_state['page'] == 'visualizations':
    render_visualizations_page()
