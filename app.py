import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import logging
from io import BytesIO
from snowflake.snowpark import Session
from criterion_factors_logic import generate_safety_data, generate_environmental_impact_data, generate_social_acceptance_data, generate_technical_feasibility_data, generate_regulatory_approval_data, generate_quantum_factor_data, generate_economic_viability_data, generate_usability_data, generate_reliability_data

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

def calculate_cr_env():
    """
    Calculate the Environmental Impact criterion based on the source data.

    :param df_source: DataFrame containing the source data from CR_ENV_SOURCE
    :param w1: Weight for energy consumption impact
    :param w2: Weight for CO2 emissions impact
    :param w3: Weight for material sustainability impact
    :param w4: Weight for environmental impact score
    :return: DataFrame correlating to the CALC_CR_ENV schema
    """
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    df_source = session.table("CR_ENV_SOURCE").to_pandas()
    # Calculate the CR_ENV using the formula
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']
    df_result['CR_ENV'] = (w1 * (df_source['ENERGY_CONSUMED'] / (df_source['DISTANCE'] * df_source['LOAD_WEIGHT'])) +
                           w2 * (df_source['CO2_EMISSIONS'] / (df_source['DISTANCE'] * df_source['LOAD_WEIGHT'])) +
                           w3 * df_source['MATERIAL_SUSTAINABILITY'] +
                           w4 * df_source['ENV_IMPACT_SCORE'])
    st.write("df_result is created")

    return df_result

def calculate_cr_sfy():
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    # Load data from CR_SFY_SOURCE table
    df = session.table("CR_SFY_SOURCE").to_pandas()
    st.write("DF is defined")
    # Calculate CR_SFY for each time period
    epsilon = 1e-6  # Avoid division by zero
    cr_sfy = np.array([1 / np.sum((df.iloc[t]["RISK_SCORE"] - df.iloc[t]["MIN_RISK_SCORE"]) / 
                                  (df.iloc[t]["MAX_RISK_SCORE"] - df.iloc[t]["MIN_RISK_SCORE"] + epsilon))
                       for t in range(len(df))])
    st.write("CR_SFY is calculated")
    # Create DataFrame with TIME and CR_SFY
    df_cr_sfy = pd.DataFrame({
        "TIME": df["TIME"],
        "CR_SFY": cr_sfy
    })
    st.write("df_cr_sfy is created")

    return df_cr_sfy

def calculate_cr_sac():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("CR_SAC_SOURCE").to_pandas()
    st.write("DF is defined")

    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']
    
    # Calculate Social Acceptance criterion
    cr_sac_raw = df_source['POSITIVE_FEEDBACK'] / (df_source['NEGATIVE_FEEDBACK'] + 1e-6)  # Avoid division by zero
    
    # Normalize CR_SAC to be in the range [0, 1]
    cr_sac_min = cr_sac_raw.min()
    cr_sac_max = cr_sac_raw.max()
    df_result['CR_SAC'] = (cr_sac_raw - cr_sac_min) / (cr_sac_max - cr_sac_min)

    st.write("df_result is created")

    return df_result

def calculate_cr_tfe():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("CR_TFE_SOURCE").to_pandas()
    st.write("DF is defined")

    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']

    # Calculate CR_TFE using the provided formula
    w1 = 0.5
    w2 = 0.5
    
    # Formula implementation
    cr_tfe_raw = (w1 * (df_source['CURRENT_TRL'] / df_source['TARGET_TRL']) +
                  w2 * (df_source['ENG_CHALLENGES_RESOLVED'] / df_source['TARGET_ENG_CHALLENGES']))

    # Normalize CR_TFE to be in the range [0, 1]
    cr_tfe_min = cr_tfe_raw.min()
    cr_tfe_max = cr_tfe_raw.max()
    df_result['CR_TFE'] = (cr_tfe_raw - cr_tfe_min) / (cr_tfe_max - cr_tfe_min)

    st.write("df_result is created")

    return df_result

def calculate_cr_reg():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("CR_REG_SOURCE").to_pandas()
    st.write("DF is defined")

    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']

    # Evenly distributed weights
    w1 = w2 = w3 = w4 = w5 = 0.2
    
    # Formula implementation
    df_result['CR_REG'] = (w1 * df_source['ETHICAL_COMPLIANCE'] +
                           w2 * df_source['LEGAL_COMPLIANCE'] +
                           w3 * df_source['LAND_USAGE_COMPLIANCE'] +
                           w4 * df_source['INT_LAW_COMPLIANCE'] +
                           w5 * df_source['TRL_COMPLIANCE'])
    
    st.write("df_result is created")    
    
    return df_result

def calculate_cr_qmf():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("CR_QMF_SOURCE").to_pandas()
    st.write("DF is defined")

    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']

    # Calculate CR_QMF as TOTAL_DISRUPTIVE_TECH / 12 (max possible value)
    df_result['CR_QMF'] = df_source['TOTAL_DISRUPTIVE_TECH'] / 12.0

    # Ensure CR_QMF is in the range [0, 1]
    df_result['CR_QMF'] = df_result['CR_QMF'].clip(0, 1)

    st.write("df_result is created")      
    
    return df_result

def calculate_cr_ecv():
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    # Fetch the data from Snowflake
    cr_ecv_source_df = session.table("CR_ECV_SOURCE").to_pandas()
    st.write("DF is defined")

    calc_data = []
    
    for _, row in cr_ecv_source_df.iterrows():
        # Convert values to appropriate data types
        time = int(row['TIME'])
        revenue = float(row['REVENUE'])
        opex = float(row['OPEX'])
        capex = float(row['CAPEX'])
        discount_rate = float(row['DISCOUNT_RATE'])
        project_lifetime = int(row['PROJECT_LIFETIME'])

        # Check for invalid project_lifetime
        if project_lifetime <= 0:
            st.error(f"Invalid project lifetime {project_lifetime} for time {time}. Skipping this record.")
            continue
        
        # Calculate NPV
        try:
            npv = sum((revenue - opex) / ((1 + discount_rate) ** t) for t in range(1, project_lifetime + 1))
        except ZeroDivisionError:
            st.error(f"Discount rate caused a division by zero error at time {time}. Skipping this record.")
            continue

        # Calculate CR_ECV, ensuring it is within the range [0, 1]
        cr_ecv = max(0, min(npv / capex, 1))
        
        calc_data.append({"TIME": time, "CR_ECV": cr_ecv})
    
    calc_df = pd.DataFrame(calc_data)
    
    return calc_df

def calculate_cr_usb():
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    
    cr_usb_source_df = session.table("CR_USB_SOURCE").to_pandas()
    st.write("CR_USB_SOURCE DataFrame Loaded")
    
    calc_data = []
    
    for _, row in cr_usb_source_df.iterrows():
        time = int(row['TIME'])
        p = float(row['PRODUCTION_OUTPUT'])
        e = float(row['USER_EXP_RATIO'])
        a = float(row['ACCESSIBILITY_AGEING'])
        
        cr_usb = max(0, min((p + e + a) / 3, 1))
        
        calc_data.append({"TIME": time, "CR_USB": cr_usb})
    
    calc_df = pd.DataFrame(calc_data)
     
    return calc_df

def calculate_cr_rlb():
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    
    cr_rlb_source_df = session.table("CR_RLB_SOURCE").to_pandas()
    st.write("CR_RLB_SOURCE DataFrame Loaded")
    
    calc_data = []
    
    for _, row in cr_rlb_source_df.iterrows():
        time = int(row['TIME'])
        d = float(row['DURABILITY'])
        c = float(row['DIGITAL_RELIABILITY'])
        w = float(row['WEATHER_DISASTER_RESILIENCE'])
        u = float(row['POLLUTION_PRODUCED'])

        cr_rlb = max(0, min((d + c + w + u) / 4, 1))
        
        calc_data.append({"TIME": time, "CR_RLB": cr_rlb})
    
    calc_df = pd.DataFrame(calc_data)
    
    return calc_df

def load_data_from_snowflake(table_name):
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    df = session.table(table_name).to_pandas()
    session.close()
    return df

def save_data_to_snowflake(df, table_name):
    try:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset buffer to start

        session = Session.builder.configs(get_snowflake_connection_params()).create()

        stage_name = "my_temp_stage"
        
        session.sql(f"CREATE TEMPORARY STAGE IF NOT EXISTS {stage_name}").collect()
        logging.info("Temporary stage created or already exists.")

        # Upload the in-memory file to the stage
        put_result = session.file.put_stream(csv_buffer, f"@{stage_name}/temp_file.csv")
        logging.info(f"PUT command result: {put_result}")

        delete_data = session.sql(f"TRUNCATE TABLE IF EXISTS {table_name}").collect()
        logging.info(f"TRUNCATE command result: {delete_data}")

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
    st.subheader("v0.03-dev")
    st.write("""
        Welcome to the Hyperloop Project System Dynamics Dashboard. 
        This application allows you to upload, manage, and visualize data related to various criteria 
        of the Hyperloop Project's system dynamics.
    """)

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
    criterion = st.selectbox("Select Criterion", ["Safety", 
                                                  "Environmental Impact", 
                                                  "Social Acceptance", 
                                                  "Technical Feasibility", 
                                                  "Regulatory Approval", 
                                                  "Quantum Factor",
                                                  "Economical Viability",
                                                  "Usability",
                                                  "Reliability",
                                                  ])

    source_table_mapping = {
        "Safety": "CR_SFY_SOURCE",
        "Environmental Impact": "CR_ENV_SOURCE",
        "Social Acceptance": "CR_SAC_SOURCE",
        "Technical Feasibility": "CR_TFE_SOURCE",
        "Regulatory Approval": "CR_REG_SOURCE",
        "Quantum Factor": "CR_QMF_SOURCE",
        "Economical Viability": "CR_ECV_SOURCE",
        "Usability": "CR_USB_SOURCE",
        "Reliability": "CR_RLB_SOURCE",
    }

    criterion_table_mapping = {
        "Safety": "CALC_CR_SFY",
        "Environmental Impact": "CALC_CR_ENV",
        "Social Acceptance": "CALC_CR_SAC",
        "Technical Feasibility": "CALC_CR_TFE",
        "Regulatory Approval": "CALC_CR_REG",
        "Quantum Factor": "CALC_CR_QMF",
        "Economical Viability": "CALC_CR_ECV",
        "Usability": "CALC_CR_USB",
        "Reliability": "CALC_CR_RLB",
    }    

    generate_function_mapping = {
        "Safety": generate_safety_data,
        "Environmental Impact": generate_environmental_impact_data,
        "Social Acceptance": generate_social_acceptance_data,
        "Technical Feasibility": generate_technical_feasibility_data,
        "Regulatory Approval": generate_regulatory_approval_data,
        "Quantum Factor": generate_quantum_factor_data,
        "Economical Viability": generate_economic_viability_data,      
        "Usability": generate_usability_data,
        "Reliability": generate_reliability_data,
    }

    criterion_function_mapping = {
        "Safety": calculate_cr_sfy,
        "Environmental Impact": calculate_cr_env,
        "Social Acceptance": calculate_cr_sac,
        "Technical Feasibility": calculate_cr_tfe,
        "Regulatory Approval": calculate_cr_reg,
        "Quantum Factor": calculate_cr_qmf,
        "Economical Viability": calculate_cr_ecv,
        "Usability": calculate_cr_usb,
        "Reliability": calculate_cr_rlb,
    }    

    selected_source_table = source_table_mapping.get(criterion, "CR_SFY_SOURCE")

    selected_criterion_table = criterion_table_mapping.get(criterion, "CALC_CR_SFY")
    
    generate_function = generate_function_mapping.get(criterion, generate_safety_data)

    criterion_function = criterion_function_mapping.get(criterion, calculate_cr_sfy)

    if st.button("Generate and Save Data"):
        df = generate_function()
        st.write(f"Data generated for {criterion}:")
        st.dataframe(df.head())
        save_data_to_snowflake(df, selected_source_table)

    if st.button("Calculate Criterion and Save Data"):
        st.write(f"DEBUG: {criterion_function}")
        df = criterion_function()
        st.write(f"Criterion {criterion} data generated.")
        st.dataframe(df.head())
        save_data_to_snowflake(df, selected_criterion_table)

    if st.button("View Source Data from Snowflake"):
        df = load_data_from_snowflake(selected_source_table)
        st.write(f"Loading {criterion} data from Snowflake...")
        st.dataframe(df)

    if st.button("View Hyperloop System Dynamics Input Criterion"):
        df = load_data_from_snowflake(selected_criterion_table)
        st.write(f"Loading Hyperloop System Dynamics Input Criterion from Snowflake... Table: {selected_criterion_table}")
        st.dataframe(df)

    uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("CSV file loaded successfully!")
        st.dataframe(df_uploaded.head())
        if st.button("Save Uploaded CSV Data to Snowflake"):
            save_data_to_snowflake(df_uploaded, selected_source_table)

    if st.button("⬅️ Back"):
        st.session_state['page'] = 'home'

def component_visualization(df_source, component):
    fig = px.line(df_source, x="TIME", y=component, title=f"{component} over Time")
    st.plotly_chart(fig)

def criterion_visualization(df_summary, crt):
    fig = px.line(df_summary, x="TIME", y=f"CR_{crt}", title=f"CR_{crt} over Time")
    st.plotly_chart(fig)

# Function to handle the visualizations page
def render_visualizations_page():
    st.title("Hyperloop Project System Dynamics Dashboard")
    
    if st.button("Visualize Safety Criterion"):
        crt = "SFY"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["RISK_SCORE", "MIN_RISK_SCORE", "MAX_RISK_SCORE"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("Visualize Environmental Impact"):
        crt = "ENV"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("Social Acceptance"):
        crt = "SAC"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("Technical Feasibility"):
        crt = "TFE"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("Regulatory Approval"):
        crt = "REG"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["ETHICAL_COMPLIANCE", "LEGAL_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "INT_LAW_COMPLIANCE", "TRL_COMPLIANCE"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("Quantum Factor"):
        crt = "QMF"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["MAGLEV_LEVITATION", 
                          "AMBIENT_INTELLIGENCE", 
                          "GENERATIVE_AI", 
                          "AI_MACHINE_LEARNING", 
                          "DIGITAL_TWINS", 
                          "FIVE_G", 
                          "QUANTUM_COMPUTING", 
                          "AUGMENTED_REALITY", 
                          "VIRTUAL_REALITY", 
                          "PRINTING_AT_SCALE", 
                          "BLOCKCHAIN", 
                          "SELF_DRIVING_AUTONOMOUS_VEHICLES",
                          "TOTAL_DISRUPTIVE_TECH"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)

    if st.button("Economical Viability"):
        crt = "ECV"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)

    if st.button("Usability"):
        crt = "USB"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)

    if st.button("Reliability"):
        crt = "RLB"
        df_source = load_data_from_snowflake(f"CR_{crt}_SOURCE")
        df_summary = load_data_from_snowflake(f"CALC_CR_{crt}")

        for component in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)               

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
