import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
from snowflake.snowpark import Session
from criterion_factors_logic import generate_safety_data

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

# General function to load data from Snowflake
def load_data_from_snowflake(table_name):
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    df = session.table(table_name).to_pandas()
    session.close()
    return df

# General function to save data to Snowflake
def save_data_to_snowflake(df, table_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        csv_file = tmp.name
        df.to_csv(csv_file, index=False)

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    # Clear the table before inserting new data
    session.sql(f"DELETE FROM {table_name}").collect()

    stage_name = "my_temp_stage"
    file_name = os.path.basename(csv_file)

    session.sql(f"CREATE TEMPORARY STAGE IF NOT EXISTS {stage_name}").collect()
    session.file.put(f"file://{csv_file}", f"@{stage_name}")

    session.sql(f"""
        COPY INTO {table_name}
        FROM @{stage_name}/{file_name}
        FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1)
    """).collect()

    session.sql(f"REMOVE @{stage_name}/{file_name}").collect()
    session.close()
    os.remove(csv_file)

# Function to handle homepage navigation
def render_homepage():
    st.title("HDME")
    st.subheader("v0.01-dev")
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
    criterion = st.selectbox("Select Criterion", ["Safety"])

    if st.button("Generate and Save Data"):
        if criterion == "Safety":
            st.write("Generating data for Safety criterion...")
            df = generate_safety_data()
            st.write("Data generated successfully!")
            st.dataframe(df.head())
            st.write("Saving data to Snowflake...")
            save_data_to_snowflake(df, "SAFETY_CRITERION_RESULTS")
            st.write("Data saved to Snowflake successfully!")

    # Button to view data from Snowflake
    if st.button("View Data from Snowflake"):
        st.write(f"Loading {criterion} data from Snowflake...")
        df = load_data_from_snowflake("SAFETY_CRITERION_RESULTS")
        st.write("Data loaded successfully!")
        st.dataframe(df)

    # Button to upload a CSV file and save to Snowflake
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("CSV file loaded successfully!")
        st.dataframe(df_uploaded.head())

        if st.button("Save Uploaded CSV Data to Snowflake"):
            st.write(f"Saving uploaded data to {criterion} table in Snowflake...")
            save_data_to_snowflake(df_uploaded, "SAFETY_CRITERION_RESULTS")
            st.write("Uploaded data saved to Snowflake successfully!")

# Function to handle the visualizations page
def render_visualizations_page():
    st.title("Hyperloop Project System Dynamics Dashboard")
    
    if st.button("Visualize Safety Criterion"):
        st.write("Loading data for visualization...")
        df = load_data_from_snowflake("SAFETY_CRITERION_RESULTS")

        st.write("Visualizing Safety Parameters...")
        for component in ["RISK_SCORE_COMPONENT_1", "RISK_SCORE_COMPONENT_2", "RISK_SCORE_COMPONENT_3", 
                          "RISK_SCORE_COMPONENT_4", "RISK_SCORE_COMPONENT_5", "SAFETY_CRITERION"]:
            fig = px.line(df, x="TIME", y=component, title=f"{component} over Time")
            st.plotly_chart(fig)

# Main application logic to switch between pages
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    render_homepage()
elif st.session_state['page'] == 'upload_data':
    render_upload_data_page()
elif st.session_state['page'] == 'visualizations':
    render_visualizations_page()
