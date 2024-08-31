import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import logging
from io import BytesIO
from snowflake.snowpark import Session
from criterion_factors_logic import generate_safety_data, generate_environmental_impact_data, generate_social_acceptance_data, generate_technical_feasibility_data, generate_regulatory_approval_data, generate_quantum_factor_data, generate_economic_viability_data, generate_usability_data, generate_reliability_data, generate_infrastructure_integration_data, generate_scalability_data
import openai
import time
import requests

# LOG LEVEL SETUP
logging.basicConfig(level=logging.INFO)

##################################
# CSS Style
##################################

st.markdown("""
    <style>
    .stButton button {
        font-size: 20px;
        padding: 15px 50px;
        width: 100%;
        margin-bottom: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

###########################################
# Snowflake connection parameters
###########################################

def get_snowflake_connection_params():
    return {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
    }

############################################
#GEN AI INTEGRATION
############################################

def get_openai_api_key():
    return st.secrets["openai"]["openai_api_key"]

def get_mistral_api_key():
    return st.secrets["mistral"]["mistral_api_key"]

############################################
#GEN AI INSIGHTS GENERATION
############################################

def test_openai_api_key():
    try:
        api_key = st.secrets["openai"]["openai_api_key"]
        st.write(f"API Key successfully retrieved: {api_key[:5]}...")  # Display only the first few characters
    except KeyError:
        st.error("API Key not found. Please check the secrets configuration.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def get_genai_insights(df, model):

    data_summary = df.describe().to_string()

    prompt = (
        "You are an expert in project performance analysis. Based on the following data summary of a Hyperloop project, please provide detailed insights on how the project is performing and offer recommendations for improvement:\n\n"
        f"{data_summary}"
    )

    openai.api_key = get_openai_api_key()

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        insights = response.choices[0].message.content.strip()
        return insights

    except Exception as e:
        st.error(f"An error occurred while fetching insights from ChatGPT: {str(e)}")
        return None

def get_mistral_insights(df):
    data_summary = df.describe().to_string()

    prompt = (
        "You are an expert in project performance analysis. Based on the following data summary of a Hyperloop project, please provide detailed insights on how the project is performing and offer recommendations for improvement:\n\n"
        f"{data_summary}"
    )

    mistral_api_key = get_mistral_api_key()
    st.write(f"key: {mistral_api_key[:5]}")

    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-large",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 300,
    }

    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        st.write(response.text)

        result = response.json()
        insights = result["choices"][0]["text"].strip()
        return insights

    except requests.exceptions.RequestException as e:
        st.write(e)
        st.error(f"An error occurred while fetching insights from Mistral AI: {str(e)}")
        return None

def analyze_hyperloop_project(model):
    df = load_data_from_snowflake("ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE")

    if df is not None:
        st.write("Data loaded successfully.")
        st.dataframe(df)

        start_time = time.time()
        insights = get_genai_insights(df, model)
        st.write(f"ChatGPT response time: {time.time() - start_time} seconds")

        if insights:
            st.write("GenAI Insights:")
            st.write(insights)
    else:
        st.error("Failed to load data, analysis cannot proceed.")

def analyze_hyperloop_project_using_mistral():
    df = load_data_from_snowflake("ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE")

    if df is not None:
        st.write("Data loaded successfully.")
        st.dataframe(df)

        start_time = time.time()
        insights = get_mistral_insights(df)
        st.write(f"Gen AI response time: {time.time() - start_time} seconds")

        if insights:
            st.write("GenAI Insights:")
            st.write(insights)
    else:
        st.error("Failed to load data, analysis cannot proceed.")

#############################################
# ETL IMPROVEMENT
#############################################

def clean_data_with_genai(df, model):
    # Convert the DataFrame to a string (JSON format) for better handling by the API
    data_json = df.to_json(orient='split')

    prompt = (
        "You are given a dataset in JSON format. Check if the 'CR_SCL' column contains any value larger than 1. "
        "If so, normalize those values so they fall within the range 0..1. Other values should stay as they are. "
        "Return the cleaned dataset in JSON format "
        "without any additional text or explanation.\n\n"
        f"Dataset: {data_json}"
    )

    openai.api_key = get_openai_api_key()

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        cleaned_data_json = response.choices[0].message.content.strip()

        cleaned_df = pd.read_json(cleaned_data_json, orient='split')
        return cleaned_df

    except Exception as e:
        st.error(f"An error occurred while processing data with ChatGPT: {str(e)}")
        return None

def normalize_cr_scl_data(model):

    df = load_data_from_snowflake("STAGING_STORE.CALC_CR_SCL_STAGING")

    if df is not None:
        st.write("Data loaded successfully.")
        st.dataframe(df)

        start_time = time.time()
        cleaned_df = clean_data_with_genai(df, model)
        st.write(f"ChatGPT response time: {time.time() - start_time} seconds")
        
        if cleaned_df is not None:
            st.write("Data cleaned successfully.")
            st.dataframe(cleaned_df)
            return cleaned_df
        else:
            st.error("Failed to clean data using ChatGPT.")
    else:
        st.error("Failed to load data from Snowflake.")
        return None

import random

def populate_calc_cr_scl_staging(time_periods):

    data = {
        "TIME_PERIOD": list(range(1, time_periods + 1)),
        "CR_SCL": [random.randint(1, 50) for _ in range(time_periods)]
    }

    df = pd.DataFrame(data)

    return df

#############################################
# MIGRATION SCRIPTS
#############################################

def fusion_to_staging_migration(source_table, dest_table):
    
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    try:
        truncate_target_table = session.sql(f"TRUNCATE TABLE {dest_table}")
        truncate_target_table.collect()
        migration_result = session.sql(f"INSERT INTO {dest_table} SELECT * FROM {source_table}")
        migration_result.collect()

        logging.info(f"MIGRATION command executed successfully: Data copied from {source_table} to {dest_table}.")
        st.write(f"DEBUG: Data successfully copied from {source_table} to {dest_table}.")
        
    except Exception as e:
        logging.error(f"Error saving data to Snowflake: {e}")
        st.error(f"An error occurred while saving data to Snowflake: {str(e)}")
        
    finally:
        if session:
            session.close()

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

        session.sql(f"USE SCHEMA FUSION_STORE").collect()
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

#############################################
# BACKUP SCRIPTS
#############################################

def backup_table_script(source_table, backup_table):
    
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    try:
        truncate_backup_table = session.sql(f"TRUNCATE TABLE {backup_table}")
        truncate_backup_table.collect()
        backup_result = session.sql(f"INSERT INTO {backup_table} SELECT * FROM {source_table}")
        backup_result.collect()
        
    except Exception as e:
        logging.error(f"Error saving data to Snowflake: {e}")
        st.error(f"An error occurred while saving data to Snowflake: {str(e)}")
        
    finally:
        if session:
            session.close()

def backup_fusion_store():
    # List of tables in the FUSION_STORE schema
    tables = [
        "CR_ECV_SOURCE",
        "CR_ENV_SOURCE",
        "CR_INF_SOURCE",
        "CR_QMF_SOURCE",
        "CR_REG_SOURCE",
        "CR_RLB_SOURCE",
        "CR_SAC_SOURCE",
        "CR_SCL_SOURCE",
        "CR_SFY_SOURCE",
        "CR_TFE_SOURCE",
        "CR_USB_SOURCE"
    ]
    
    progress_bar = st.progress(0)
    
    total_tables = len(tables)
    for index, table in enumerate(tables):
        source_table = f"FUSION_STORE.{table}"
        backup_table_name = f"FUSION_STORE.{table}_BCK"
        backup_table_script(source_table, backup_table_name)
        
        progress = (index + 1) / total_tables
        progress_bar.progress(progress)
        
    st.write(f"Backup for Fusion store is completed")        

def backup_staging_store():
    
    tables = [
        "CALC_CR_ECV_STAGING",
        "CALC_CR_ENV_STAGING",
        "CALC_CR_INF_STAGING",
        "CALC_CR_QMF_STAGING",
        "CALC_CR_REG_STAGING",
        "CALC_CR_RLB_STAGING",
        "CALC_CR_SAC_STAGING",
        "CALC_CR_SCL_STAGING",
        "CALC_CR_SFY_STAGING",
        "CALC_CR_TFE_STAGING",
        "CALC_CR_USB_STAGING",
        "CR_ECV_STAGING",
        "CR_ENV_STAGING",
        "CR_INF_STAGING",
        "CR_QMF_STAGING",
        "CR_REG_STAGING",
        "CR_RLB_STAGING",
        "CR_SAC_STAGING",
        "CR_SCL_STAGING",
        "CR_SFY_STAGING",
        "CR_TFE_STAGING",
        "CR_USB_STAGING"
    ]

    progress_bar = st.progress(0)
    
    total_tables = len(tables)
    for index, table in enumerate(tables):
        source_table = f"STAGING_STORE.{table}"
        backup_table_name = f"STAGING_STORE.{table}_BCK"
        backup_table_script(source_table, backup_table_name)
        
        progress = (index + 1) / total_tables
        progress_bar.progress(progress)
         
    st.write(f"Backup for Staging store is completed")                 

def backup_alliance_store():
    
    tables = [
        "HPL_SD_CRS_ALLIANCE"
    ]

    progress_bar = st.progress(0)
    
    total_tables = len(tables)
    for index, table in enumerate(tables):
        source_table = f"ALLIANCE_STORE.{table}"
        backup_table_name = f"ALLIANCE_STORE.{table}_BCK"
        backup_table_script(source_table, backup_table_name)
        
        progress = (index + 1) / total_tables
        progress_bar.progress(progress)
         
    st.write(f"Backup for Alliance store is completed")                           


#############################################
# CRITERION CALCULATION
#############################################

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
    df_source = session.table("STAGING_STORE.CR_ENV_STAGING").to_pandas()

    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']
    df_result['CR_ENV'] = (w1 * (df_source['ENERGY_CONSUMED'] / (df_source['DISTANCE'] * df_source['LOAD_WEIGHT'])) +
                           w2 * (df_source['CO2_EMISSIONS'] / (df_source['DISTANCE'] * df_source['LOAD_WEIGHT'])) +
                           w3 * df_source['MATERIAL_SUSTAINABILITY'] +
                           w4 * df_source['ENV_IMPACT_SCORE'])

    return df_result

def calculate_cr_sfy():
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    # Load data from CR_SFY_SOURCE table
    df = session.table("STAGING_STORE.CR_SFY_STAGING").to_pandas()
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

    return df_cr_sfy

def calculate_cr_sac():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("STAGING_STORE.CR_SAC_STAGING").to_pandas()

    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']
    
    # Calculate Social Acceptance criterion
    cr_sac_raw = df_source['POSITIVE_FEEDBACK'] / (df_source['NEGATIVE_FEEDBACK'] + 1e-6)  # Avoid division by zero
    
    # Normalize CR_SAC to be in the range [0, 1]
    cr_sac_min = cr_sac_raw.min()
    cr_sac_max = cr_sac_raw.max()
    df_result['CR_SAC'] = (cr_sac_raw - cr_sac_min) / (cr_sac_max - cr_sac_min)

    return df_result

def calculate_cr_tfe():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("STAGING_STORE.CR_TFE_STAGING").to_pandas()

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

    return df_result

def calculate_cr_reg():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("STAGING_STORE.CR_REG_STAGING").to_pandas()

    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']

    # Weights
    w1 = w2 = w3 = w4 = w5 = 0.2
    
    # Formula implementation
    df_result['CR_REG'] = (w1 * df_source['ETHICAL_COMPLIANCE'] +
                           w2 * df_source['LEGAL_COMPLIANCE'] +
                           w3 * df_source['LAND_USAGE_COMPLIANCE'] +
                           w4 * df_source['INT_LAW_COMPLIANCE'] +
                           w5 * df_source['TRL_COMPLIANCE'])
    
    return df_result

def calculate_cr_qmf():

    session = Session.builder.configs(get_snowflake_connection_params()).create()

    df_source = session.table("STAGING_STORE.CR_QMF_STAGING").to_pandas()

    df_result = pd.DataFrame()
    df_result['TIME'] = df_source['TIME']

    df_result['CR_QMF'] = df_source['TOTAL_DISRUPTIVE_TECH'] / 12.0

    df_result['CR_QMF'] = df_result['CR_QMF'].clip(0, 1) 
    
    return df_result

def calculate_cr_ecv():
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    cr_ecv_source_df = session.table("STAGING_STORE.CR_ECV_STAGING").to_pandas()

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
    
    cr_usb_source_df = session.table("STAGING_STORE.CR_USB_STAGING").to_pandas()
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
    
    cr_rlb_source_df = session.table("STAGING_STORE.CR_RLB_STAGING").to_pandas()
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

def calculate_cr_inf():
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    
    cr_inf_source_df = session.table("STAGING_STORE.CR_INF_STAGING").to_pandas()
    st.write("CR_INF_SOURCE DataFrame Loaded")
    
    calc_data = []
    
    for _, row in cr_inf_source_df.iterrows():
        time = int(row['TIME'])
        C = float(row['COMMON_INFRA_FEATURES'])
        E = float(row['CONSTRUCTION_BARRIERS'])
        M = float(row['INTERMODAL_CONNECTIONS'])
        A = float(row['INFRA_ADAPTABILITY_FEATURES'])

        cr_inf = max(0, min((C + E + M + A) / 4, 1))
        
        calc_data.append({"TIME": time, "CR_INF": cr_inf})
    
    calc_df = pd.DataFrame(calc_data)
    
    return calc_df

def calculate_cr_scl():
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    
    cr_scl_source_df = session.table("STAGING_STORE.CR_SCL_STAGING").to_pandas()
    st.write("CR_SCL_SOURCE DataFrame Loaded")
    
    calc_data = []
    
    for _, row in cr_scl_source_df.iterrows():
        time = int(row['TIME'])
        L1 = float(row['RESOURCE_MILEAGE'])
        Q = float(row['PLANNED_VOLUME'])
        K1 = float(row['ADJUSTMENT_COEF_1'])
        K2 = float(row['ADJUSTMENT_COEF_2'])
        K3 = float(row['ADJUSTMENT_COEF_3'])

        cr_scl = max(0, min((L1 * Q * K1 * K2 * K3) ** (1/3), 1))
        
        calc_data.append({"TIME": time, "CR_SCL": cr_scl})
    
    calc_df = pd.DataFrame(calc_data)
    
    return calc_df


#############################################
# ALLIANCE STORE OPERATIONS
#############################################

def populate_hpl_sd_crs():
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    # Load data from each CALC_CR_* table
    cr_env_df = session.table("STAGING_STORE.CALC_CR_ENV_STAGING").to_pandas()
    cr_sac_df = session.table("STAGING_STORE.CALC_CR_SAC_STAGING").to_pandas()
    cr_tfe_df = session.table("STAGING_STORE.CALC_CR_TFE_STAGING").to_pandas()
    cr_sfy_df = session.table("STAGING_STORE.CALC_CR_SFY_STAGING").to_pandas()
    cr_reg_df = session.table("STAGING_STORE.CALC_CR_REG_STAGING").to_pandas()
    cr_qmf_df = session.table("STAGING_STORE.CALC_CR_QMF_STAGING").to_pandas()
    cr_ecv_df = session.table("STAGING_STORE.CALC_CR_ECV_STAGING").to_pandas()
    cr_usb_df = session.table("STAGING_STORE.CALC_CR_USB_STAGING").to_pandas()
    cr_rlb_df = session.table("STAGING_STORE.CALC_CR_RLB_STAGING").to_pandas()
    cr_inf_df = session.table("STAGING_STORE.CALC_CR_INF_STAGING").to_pandas()
    cr_scl_df = session.table("STAGING_STORE.CALC_CR_SCL_STAGING").to_pandas()

    # Start with the first DataFrame and merge sequentially, keeping only TIME and the relevant criterion column
    combined_df = cr_env_df[['TIME', 'CR_ENV']]\
        .merge(cr_sac_df[['TIME', 'CR_SAC']], on="TIME", how="outer")\
        .merge(cr_tfe_df[['TIME', 'CR_TFE']], on="TIME", how="outer")\
        .merge(cr_sfy_df[['TIME', 'CR_SFY']], on="TIME", how="outer")\
        .merge(cr_reg_df[['TIME', 'CR_REG']], on="TIME", how="outer")\
        .merge(cr_qmf_df[['TIME', 'CR_QMF']], on="TIME", how="outer")\
        .merge(cr_ecv_df[['TIME', 'CR_ECV']], on="TIME", how="outer")\
        .merge(cr_usb_df[['TIME', 'CR_USB']], on="TIME", how="outer")\
        .merge(cr_rlb_df[['TIME', 'CR_RLB']], on="TIME", how="outer")\
        .merge(cr_inf_df[['TIME', 'CR_INF']], on="TIME", how="outer")\
        .merge(cr_scl_df[['TIME', 'CR_SCL']], on="TIME", how="outer")

    # Ensure there are no extra columns
    expected_columns = ['TIME', 'CR_ENV', 'CR_SAC', 'CR_TFE', 'CR_SFY', 'CR_REG', 'CR_QMF', 'CR_ECV', 'CR_USB', 'CR_RLB', 'CR_INF', 'CR_SCL']
    combined_df = combined_df[expected_columns]

    # Check for any unexpected columns
    if list(combined_df.columns) != expected_columns:
        st.error("The DataFrame columns do not match the expected structure.")
        st.write("DataFrame columns:", combined_df.columns)
        return

    # Fill any NaN values that may have been introduced during the outer joins
    combined_df.fillna(0, inplace=True)

    st.write("DEBUG: ALL")
    st.write(combined_df) 
    st.write("HEAD")
    st.dataframe(combined_df.head())

    return combined_df

##############################################################
# HOMEPAGE CREATION
##############################################################

def render_homepage():
    st.title("HDME")
    st.subheader("v0.1.1-dev")
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
        if st.button("📊\n\nHyperloop Project System Dynamics", use_container_width=True):
            st.session_state['page'] = 'visualizations'

    if st.button("SCENARIOS SIMULATION 🌐"):
        st.session_state['page'] = 'scenarious'    

    if st.button("EGTL EXPERIMENT ⚗️"):
        st.session_state['page'] = 'experiment'

    if st.button("UTILITIES 🛠️"):
        st.session_state['page'] = 'utility'

    model = st.radio(
        "Select GPT model for analysis:",
        options=["gpt-3.5-turbo", "gpt-4", ],
        index=0
    )

    if st.button("ANALYZE HYPERLOOP PROJECT 🧠"):
        analyze_hyperloop_project(model)

    if st.button("ANALYZE HYPERLOOP PROJECT USING MISTRAL"):
        analyze_hyperloop_project_using_mistral()        

##############################################################
# Data upload and management page
##############################################################

def render_upload_data_page():
    st.title("Upload data to ecosystem")

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
                                                  "Infrastructure Integration",
                                                  "Scalability",
                                                  ])

    # Handling custom time periods logic
    time_period_raw = st.text_input('Time period:', value='100')
    time_periods = int(time_period_raw)   

    source_table_mapping = {
        "Safety": "FUSION_STORE.CR_SFY_SOURCE",
        "Environmental Impact": "FUSION_STORE.CR_ENV_SOURCE",
        "Social Acceptance": "FUSION_STORE.CR_SAC_SOURCE",
        "Technical Feasibility": "FUSION_STORE.CR_TFE_SOURCE",
        "Regulatory Approval": "FUSION_STORE.CR_REG_SOURCE",
        "Quantum Factor": "FUSION_STORE.CR_QMF_SOURCE",
        "Economical Viability": "FUSION_STORE.CR_ECV_SOURCE",
        "Usability": "FUSION_STORE.CR_USB_SOURCE",
        "Reliability": "FUSION_STORE.CR_RLB_SOURCE",
        "Infrastructure Integration": "FUSION_STORE.CR_INF_SOURCE",
        "Scalability": "FUSION_STORE.CR_SCL_SOURCE",
    }

    staging_table_mapping = {
        "Safety": "STAGING_STORE.CR_SFY_STAGING",
        "Environmental Impact": "STAGING_STORE.CR_ENV_STAGING",
        "Social Acceptance": "STAGING_STORE.CR_SAC_STAGING",
        "Technical Feasibility": "STAGING_STORE.CR_TFE_STAGING",
        "Regulatory Approval": "STAGING_STORE.CR_REG_STAGING",
        "Quantum Factor": "STAGING_STORE.CR_QMF_STAGING",
        "Economical Viability": "STAGING_STORE.CR_ECV_STAGING",
        "Usability": "STAGING_STORE.CR_USB_STAGING",
        "Reliability": "STAGING_STORE.CR_RLB_STAGING",
        "Infrastructure Integration": "STAGING_STORE.CR_INF_STAGING",
        "Scalability": "STAGING_STORE.CR_SCL_STAGING",
    }    

    criterion_table_mapping = {
        "Safety": "STAGING_STORE.CALC_CR_SFY_STAGING",
        "Environmental Impact": "STAGING_STORE.CALC_CR_ENV_STAGING",
        "Social Acceptance": "STAGING_STORE.CALC_CR_SAC_STAGING",
        "Technical Feasibility": "STAGING_STORE.CALC_CR_TFE_STAGING",
        "Regulatory Approval": "STAGING_STORE.CALC_CR_REG_STAGING",
        "Quantum Factor": "STAGING_STORE.CALC_CR_QMF_STAGING",
        "Economical Viability": "STAGING_STORE.CALC_CR_ECV_STAGING",
        "Usability": "STAGING_STORE.CALC_CR_USB_STAGING",
        "Reliability": "STAGING_STORE.CALC_CR_RLB_STAGING",
        "Infrastructure Integration": "STAGING_STORE.CALC_CR_INF_STAGING",
        "Scalability": "STAGING_STORE.CALC_CR_SCL_STAGING",
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
        "Infrastructure Integration": generate_infrastructure_integration_data,
        "Scalability": generate_scalability_data,
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
        "Infrastructure Integration": calculate_cr_inf,
        "Scalability": calculate_cr_scl,
    }    

    selected_source_table = source_table_mapping.get(criterion, "FUSION_STORE.CR_SFY_SOURCE")
    selected_staging_table = staging_table_mapping.get(criterion, "STAGING_STORE.CR_SFY_STAGING")

    selected_criterion_table = criterion_table_mapping.get(criterion, "CALC_CR_SFY")
    
    generate_function = generate_function_mapping.get(criterion, generate_safety_data)

    criterion_function = criterion_function_mapping.get(criterion, calculate_cr_sfy)

    if st.button("🗃️ Generate and Save Data"):
        df = generate_function(time_periods)
        st.write(f"Data generated for {criterion}:")
        st.dataframe(df.head())
        save_data_to_snowflake(df, selected_source_table)
        fusion_to_staging_migration(selected_source_table, selected_staging_table)
        st.write(f"Data loaded for {criterion}.")

    if st.button("🔢 Calculate Criterion and Save Data"):
        st.write(f"DEBUG: {criterion_function}")
        df = criterion_function()
        st.write(f"Criterion {criterion} data generated.")
        st.dataframe(df.head())
        save_data_to_snowflake(df, selected_criterion_table)

    if st.button("🔎 View Fusion Store Data from Snowflake"):
        df = load_data_from_snowflake(selected_source_table)
        st.write(f"Loading {criterion} data from Snowflake...")
        st.dataframe(df)

    if st.button("🔎 View Staging Store Data from Snowflake"):
        df = load_data_from_snowflake(selected_staging_table)
        st.write(f"Loading {criterion} data from Snowflake...")
        st.dataframe(df)

    if st.button("🔎 View Hyperloop System Dynamics Input Criterion"):
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
            fusion_to_staging_migration(selected_source_table, selected_staging_table)

    if st.button("🗝 POPULATE SUCCESS FACTORS TABLE"):
        df = populate_hpl_sd_crs()
        st.write(f"Criterion data preview.")
        st.dataframe(df.head())
        save_data_to_snowflake(df, "ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE")
        st.write(f"Table population completed. Please proceed to visualization tab")

    if st.button("⬅️ BACK"):
        st.session_state['page'] = 'home'

#############################
# HELPERS FOR VISUALIZATION
#############################

def component_visualization(df_source, component):
    fig = px.line(df_source, x="TIME", y=component, title=f"{component} over Time")
    st.plotly_chart(fig)

def criterion_visualization(df_summary, crt):
    fig = px.line(df_summary, x="TIME", y=f"CR_{crt}", title=f"CR_{crt} over Time")
    st.plotly_chart(fig)

def visualize_all_success_factors():
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    hpl_sd_crs_df = session.table("ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE").to_pandas()

    criteria = ['CR_ENV', 'CR_SAC', 'CR_TFE', 'CR_SFY', 'CR_REG', 'CR_QMF', 'CR_ECV', 'CR_USB', 'CR_RLB', 'CR_INF', 'CR_SCL']

    cols = st.columns(3)
    
    for i, criterion in enumerate(criteria):
        fig = px.line(hpl_sd_crs_df, x='TIME', y=criterion, title=f"{criterion} over Time")
        col_idx = i % 3 
        with cols[col_idx]: 
            st.plotly_chart(fig)

import plotly.express as px

def calculate_maturity_level(dmmi_df):
    dmmi_df['Governance and Management'] = dmmi_df['Governance and Management'].clip(0, 1)
    dmmi_df['Strategy and Planning'] = dmmi_df['Strategy and Planning'].clip(0, 1)
    dmmi_df['Technology and Infrastructure'] = dmmi_df['Technology and Infrastructure'].clip(0, 1)
    dmmi_df['Processes and Methodologies'] = dmmi_df['Processes and Methodologies'].clip(0, 1)
    dmmi_df['People and Culture'] = dmmi_df['People and Culture'].clip(0, 1)
    dmmi_df['Data and Information Management'] = dmmi_df['Data and Information Management'].clip(0, 1)
    dmmi_df['Performance Measurement'] = dmmi_df['Performance Measurement'].clip(0, 1)
    
    weights = {
        'Governance and Management': 0.15,
        'Strategy and Planning': 0.15,
        'Technology and Infrastructure': 0.15,
        'Processes and Methodologies': 0.15,
        'People and Culture': 0.15,
        'Data and Information Management': 0.15,
        'Performance Measurement': 0.10,
    }
    
    dmmi_df['Maturity Level'] = (
        dmmi_df['Governance and Management'] * weights['Governance and Management'] +
        dmmi_df['Strategy and Planning'] * weights['Strategy and Planning'] +
        dmmi_df['Technology and Infrastructure'] * weights['Technology and Infrastructure'] +
        dmmi_df['Processes and Methodologies'] * weights['Processes and Methodologies'] +
        dmmi_df['People and Culture'] * weights['People and Culture'] +
        dmmi_df['Data and Information Management'] * weights['Data and Information Management'] +
        dmmi_df['Performance Measurement'] * weights['Performance Measurement']
    )
    
    dmmi_df['Maturity Level'] = 1 + 4 * dmmi_df['Maturity Level']
    
    return dmmi_df

def visualize_ddmi_factors(dmmi_df):
    cols = st.columns(4)
    
    factors = [
        'Governance and Management', 'Strategy and Planning', 
        'Technology and Infrastructure', 'Processes and Methodologies',
        'People and Culture', 'Data and Information Management',
        'Performance Measurement', 'Maturity Level'
    ]
    
    for i, factor in enumerate(factors):
        fig = px.line(dmmi_df, x='TIME', y=factor, title=f"{factor} over Time")
        col_idx = i % 4  # Get column index: 0, 1, 2, 3
        with cols[col_idx]:  # Place the figure in the corresponding column
            st.plotly_chart(fig)

def render_ddmi_dashboard():
    st.title("DMMI Factors and Project Maturity Level Visualization")
    
    session = Session.builder.configs(get_snowflake_connection_params()).create()
    hpl_sd_crs_df = session.table("ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE").to_pandas()

    dmmi_df = pd.DataFrame({
        'TIME': hpl_sd_crs_df['TIME'],
        'Governance and Management': hpl_sd_crs_df[['CR_REG', 'CR_TFE', 'CR_SFY']].mean(axis=1),
        'Strategy and Planning': hpl_sd_crs_df[['CR_TFE', 'CR_ENV', 'CR_ECV']].mean(axis=1),
        'Technology and Infrastructure': hpl_sd_crs_df[['CR_TFE', 'CR_INF', 'CR_RLB']].mean(axis=1),
        'Processes and Methodologies': hpl_sd_crs_df[['CR_SFY', 'CR_TFE']].mean(axis=1),
        'People and Culture': hpl_sd_crs_df[['CR_SAC', 'CR_USB']].mean(axis=1),
        'Data and Information Management': hpl_sd_crs_df[['CR_QMF', 'CR_RLB']].mean(axis=1),
        'Performance Measurement': hpl_sd_crs_df[['CR_ECV', 'CR_USB']].mean(axis=1)
    })

    dmmi_df = calculate_maturity_level(dmmi_df)

    visualize_ddmi_factors(dmmi_df)          

########################
# Visualizations page
########################

def render_visualizations_page():
    st.title("Hyperloop Project System Dynamics Dashboard")

    if st.button("📊 HYPERLOOP SUCCESS FACTORS DASHBOARD"):
        visualize_all_success_factors()

    if st.button("📈 HYPERLOOP PROJECT DMMI DASHBOARD"):
        render_ddmi_dashboard()

    st.title("SUCCESS FACTORS BREAKDOWN")

    if st.button("🔍 SAFETY CRITERION"):
        crt = "SFY"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["RISK_SCORE", "MIN_RISK_SCORE", "MAX_RISK_SCORE"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("🌍 ENVIRNOMENTAL IMPACT"):
        crt = "ENV"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("📰 SOCIAL ACCEPTANCE"):
        crt = "SAC"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("🔧 TECHNICAL FEASIBILITY"):
        crt = "TFE"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("✔️ REGULATORY APPROVAL"):
        crt = "REG"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["ETHICAL_COMPLIANCE", "LEGAL_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "INT_LAW_COMPLIANCE", "TRL_COMPLIANCE"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt) 

    if st.button("⚛️ QUANTUM FACTOR"):
        crt = "QMF"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

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

    if st.button("💶 ECONOMICAL VIABILITY "):
        crt = "ECV"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)

    if st.button("💡USABILITY"):
        crt = "USB"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)

    if st.button("⚖️ RELIABILITY"):
        crt = "RLB"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)

    if st.button("🏭 INFRASTRUCTURE INTEGRATION"):
        crt = "INF"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)

    if st.button("🛬SCALABILITY"):
        crt = "SCL"
        df_source = load_data_from_snowflake(f"STAGING_STORE.CR_{crt}_STAGING")
        df_summary = load_data_from_snowflake(f"STAGING_STORE.CALC_CR_{crt}_STAGING")

        for component in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
            component_visualization(df_source,component)

        criterion_visualization(df_summary, crt)        

    if st.button("⬅️ BACK"):
        st.session_state['page'] = 'home'        

#######################################
# SCENARIOUS SIMULATIONS
#######################################

from simulation_scenarios import (
    generate_cr_env_data_rapid_decline, generate_cr_sac_data_rapid_decline, generate_cr_tfe_data_rapid_decline, 
    generate_cr_sfy_data_rapid_decline, generate_cr_reg_data_rapid_decline, generate_cr_qmf_data_rapid_decline,
    generate_cr_ecv_data_rapid_decline, generate_cr_usb_data_rapid_decline, generate_cr_rlb_data_rapid_decline,
    generate_cr_inf_data_rapid_decline, generate_cr_scl_data_rapid_decline,
    generate_cr_env_decline_over_time_data, generate_cr_sac_decline_over_time_data,
    generate_cr_tfe_decline_over_time_data, generate_cr_sfy_decline_over_time_data,
    generate_cr_reg_decline_over_time_data, generate_cr_qmf_decline_over_time_data,
    generate_cr_ecv_decline_over_time_data, generate_cr_usb_decline_over_time_data,
    generate_cr_rlb_decline_over_time_data, generate_cr_inf_decline_over_time_data,
    generate_cr_scl_decline_over_time_data,
    generate_cr_env_rapid_growth_data, generate_cr_sac_rapid_growth_data,
    generate_cr_tfe_rapid_growth_data, generate_cr_sfy_rapid_growth_data,
    generate_cr_reg_rapid_growth_data, generate_cr_qmf_rapid_growth_data,
    generate_cr_ecv_rapid_growth_data, generate_cr_usb_rapid_growth_data,
    generate_cr_rlb_rapid_growth_data, generate_cr_inf_rapid_growth_data,
    generate_cr_scl_rapid_growth_data,
    generate_cr_env_sustainable_growth_data, generate_cr_sac_sustainable_growth_data,
    generate_cr_tfe_sustainable_growth_data, generate_cr_sfy_sustainable_growth_data,
    generate_cr_reg_sustainable_growth_data, generate_cr_qmf_sustainable_growth_data,
    generate_cr_ecv_sustainable_growth_data, generate_cr_usb_sustainable_growth_data,
    generate_cr_rlb_sustainable_growth_data, generate_cr_inf_sustainable_growth_data,
    generate_cr_scl_sustainable_growth_data        
)

#######################################
# RAPID DECLINE SCENARIO
#######################################

def generate_rapid_decline_scenario(time_periods):

    st.write("Executed rapid decline scenario for simulation model.")

    cr_env_df = generate_cr_env_data_rapid_decline(time_periods)
    cr_sac_df = generate_cr_sac_data_rapid_decline(time_periods)
    cr_tfe_df = generate_cr_tfe_data_rapid_decline(time_periods)
    cr_sfy_df = generate_cr_sfy_data_rapid_decline(time_periods)
    cr_reg_df = generate_cr_reg_data_rapid_decline(time_periods)
    cr_qmf_df = generate_cr_qmf_data_rapid_decline(time_periods)
    cr_ecv_df = generate_cr_ecv_data_rapid_decline(time_periods)
    cr_usb_df = generate_cr_usb_data_rapid_decline(time_periods)
    cr_rlb_df = generate_cr_rlb_data_rapid_decline(time_periods)
    cr_inf_df = generate_cr_inf_data_rapid_decline(time_periods)
    cr_scl_df = generate_cr_scl_data_rapid_decline(time_periods)

    scenarios_calculation_to_snowlake(cr_env_df, cr_sac_df, cr_tfe_df, cr_sfy_df, cr_reg_df, cr_qmf_df, cr_ecv_df, cr_usb_df, cr_rlb_df, cr_inf_df, cr_scl_df)

#######################################
# DECLINE OVERTIME SCENARIO
#######################################

def generate_decline_over_time_scenario(time_periods):

    st.write("Executed decline over time scenario for simulation model.")

    cr_env_df = generate_cr_env_decline_over_time_data(time_periods)
    cr_sac_df = generate_cr_sac_decline_over_time_data(time_periods)
    cr_tfe_df = generate_cr_tfe_decline_over_time_data(time_periods)
    cr_sfy_df = generate_cr_sfy_decline_over_time_data(time_periods)
    cr_reg_df = generate_cr_reg_decline_over_time_data(time_periods)
    cr_qmf_df = generate_cr_qmf_decline_over_time_data(time_periods)
    cr_ecv_df = generate_cr_ecv_decline_over_time_data(time_periods)
    cr_usb_df = generate_cr_usb_decline_over_time_data(time_periods)
    cr_rlb_df = generate_cr_rlb_decline_over_time_data(time_periods)
    cr_inf_df = generate_cr_inf_decline_over_time_data(time_periods)
    cr_scl_df = generate_cr_scl_decline_over_time_data(time_periods)

    scenarios_calculation_to_snowlake(cr_env_df, cr_sac_df, cr_tfe_df, cr_sfy_df, cr_reg_df, cr_qmf_df, cr_ecv_df, cr_usb_df, cr_rlb_df, cr_inf_df, cr_scl_df)

#######################################
# DECLINE OVERTIME SCENARIO
#######################################

def generate_rapid_growth_scenario(time_periods):

    st.write("Executed rapid growth scenario for simulation model.")    

    cr_env_df = generate_cr_env_rapid_growth_data(time_periods)
    cr_sac_df = generate_cr_sac_rapid_growth_data(time_periods)
    cr_tfe_df = generate_cr_tfe_rapid_growth_data(time_periods)
    cr_sfy_df = generate_cr_sfy_rapid_growth_data(time_periods)
    cr_reg_df = generate_cr_reg_rapid_growth_data(time_periods)
    cr_qmf_df = generate_cr_qmf_rapid_growth_data(time_periods)
    cr_ecv_df = generate_cr_ecv_rapid_growth_data(time_periods)
    cr_usb_df = generate_cr_usb_rapid_growth_data(time_periods)
    cr_rlb_df = generate_cr_rlb_rapid_growth_data(time_periods)
    cr_inf_df = generate_cr_inf_rapid_growth_data(time_periods)
    cr_scl_df = generate_cr_scl_rapid_growth_data(time_periods)

    scenarios_calculation_to_snowlake(cr_env_df, cr_sac_df, cr_tfe_df, cr_sfy_df, cr_reg_df, cr_qmf_df, cr_ecv_df, cr_usb_df, cr_rlb_df, cr_inf_df, cr_scl_df)      

#######################################
# SUSTAINABLE GROWTH SCENARIO
#######################################

def generate_sustainable_growth_scenario(time_periods):

    st.write("Executed sustainable growth scenario for simulation model.")   

    cr_env_df = generate_cr_env_sustainable_growth_data(time_periods)
    cr_sac_df = generate_cr_sac_sustainable_growth_data(time_periods)
    cr_tfe_df = generate_cr_tfe_sustainable_growth_data(time_periods)
    cr_sfy_df = generate_cr_sfy_sustainable_growth_data(time_periods)
    cr_reg_df = generate_cr_reg_sustainable_growth_data(time_periods)
    cr_qmf_df = generate_cr_qmf_sustainable_growth_data(time_periods)
    cr_ecv_df = generate_cr_ecv_sustainable_growth_data(time_periods)
    cr_usb_df = generate_cr_usb_sustainable_growth_data(time_periods)
    cr_rlb_df = generate_cr_rlb_sustainable_growth_data(time_periods)
    cr_inf_df = generate_cr_inf_sustainable_growth_data(time_periods)
    cr_scl_df = generate_cr_scl_sustainable_growth_data(time_periods)

    scenarios_calculation_to_snowlake(cr_env_df, cr_sac_df, cr_tfe_df, cr_sfy_df, cr_reg_df, cr_qmf_df, cr_ecv_df, cr_usb_df, cr_rlb_df, cr_inf_df, cr_scl_df)    

#######################################
# REUSABLE METHODS FOR SCENARIOS
#######################################

def scenarios_calculation_to_snowlake(cr_env_df, cr_sac_df, cr_tfe_df, cr_sfy_df, cr_reg_df, cr_qmf_df, cr_ecv_df, cr_usb_df, cr_rlb_df, cr_inf_df, cr_scl_df):

    save_data_to_snowflake(cr_env_df, "FUSION_STORE.CR_ENV_SOURCE")
    save_data_to_snowflake(cr_sac_df, "FUSION_STORE.CR_SAC_SOURCE")
    save_data_to_snowflake(cr_tfe_df, "FUSION_STORE.CR_TFE_SOURCE")
    save_data_to_snowflake(cr_sfy_df, "FUSION_STORE.CR_SFY_SOURCE")
    save_data_to_snowflake(cr_reg_df, "FUSION_STORE.CR_REG_SOURCE")
    save_data_to_snowflake(cr_qmf_df, "FUSION_STORE.CR_QMF_SOURCE")
    save_data_to_snowflake(cr_ecv_df, "FUSION_STORE.CR_ECV_SOURCE")
    save_data_to_snowflake(cr_usb_df, "FUSION_STORE.CR_USB_SOURCE")
    save_data_to_snowflake(cr_rlb_df, "FUSION_STORE.CR_RLB_SOURCE")
    save_data_to_snowflake(cr_inf_df, "FUSION_STORE.CR_INF_SOURCE")
    save_data_to_snowflake(cr_scl_df, "FUSION_STORE.CR_SCL_SOURCE")

    fusion_to_staging_migration("FUSION_STORE.CR_ENV_SOURCE", "STAGING_STORE.CR_ENV_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_SAC_SOURCE", "STAGING_STORE.CR_SAC_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_TFE_SOURCE", "STAGING_STORE.CR_TFE_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_SFY_SOURCE", "STAGING_STORE.CR_SFY_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_REG_SOURCE", "STAGING_STORE.CR_REG_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_QMF_SOURCE", "STAGING_STORE.CR_QMF_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_ECV_SOURCE", "STAGING_STORE.CR_ECV_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_USB_SOURCE", "STAGING_STORE.CR_USB_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_RLB_SOURCE", "STAGING_STORE.CR_RLB_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_INF_SOURCE", "STAGING_STORE.CR_INF_STAGING")
    fusion_to_staging_migration("FUSION_STORE.CR_SCL_SOURCE", "STAGING_STORE.CR_SCL_STAGING")

    scenario_df_env = calculate_cr_env()
    st.write(f"Criterion scenario_df_env data loaded.")
    st.dataframe(scenario_df_env.head())
    save_data_to_snowflake(scenario_df_env, "STAGING_STORE.CALC_CR_ENV_STAGING")   

    scenario_df_sac = calculate_cr_sac()
    st.write(f"Criterion scenario_df_sac data loaded.")
    st.dataframe(scenario_df_sac.head())
    save_data_to_snowflake(scenario_df_sac, "STAGING_STORE.CALC_CR_SAC_STAGING")  

    scenario_df_tfe = calculate_cr_tfe()
    st.write(f"Criterion scenario_df_tfe data loaded.")
    st.dataframe(scenario_df_tfe.head())
    save_data_to_snowflake(scenario_df_tfe, "STAGING_STORE.CALC_CR_TFE_STAGING")  

    scenario_df_sfy = calculate_cr_sfy()
    st.write(f"Criterion scenario_df_sfy data loaded.")
    st.dataframe(scenario_df_sfy.head())
    save_data_to_snowflake(scenario_df_sfy, "STAGING_STORE.CALC_CR_SFY_STAGING") 

    scenario_df_reg = calculate_cr_reg()
    st.write(f"Criterion scenario_df_reg data loaded.")
    st.dataframe(scenario_df_reg.head())
    save_data_to_snowflake(scenario_df_reg, "STAGING_STORE.CALC_CR_REG_STAGING")  

    scenario_df_qmf = calculate_cr_qmf()
    st.write(f"Criterion scenario_df_qmf data loaded.")
    st.dataframe(scenario_df_qmf.head())
    save_data_to_snowflake(scenario_df_qmf, "STAGING_STORE.CALC_CR_QMF_STAGING")  

    scenario_df_ecv = calculate_cr_ecv()
    st.write(f"Criterion scenario_df_ecv data loaded.")
    st.dataframe(scenario_df_ecv.head())
    save_data_to_snowflake(scenario_df_ecv, "STAGING_STORE.CALC_CR_ECV_STAGING") 

    scenario_df_usb = calculate_cr_usb()
    st.write(f"Criterion scenario_df_usb data loaded.")
    st.dataframe(scenario_df_usb.head())
    save_data_to_snowflake(scenario_df_usb, "STAGING_STORE.CALC_CR_USB_STAGING")  

    scenario_df_rlb = calculate_cr_rlb()
    st.write(f"Criterion scenario_df_rlb data loaded.")
    st.dataframe(scenario_df_rlb.head())
    save_data_to_snowflake(scenario_df_rlb, "STAGING_STORE.CALC_CR_RLB_STAGING")        
    
    scenario_df_inf = calculate_cr_inf()
    st.write(f"Criterion scenario_df_inf data loaded.")
    st.dataframe(scenario_df_inf.head())
    save_data_to_snowflake(scenario_df_inf, "STAGING_STORE.CALC_CR_INF_STAGING")   

    scenario_df_scl = calculate_cr_scl()
    st.write(f"Criterion scenario_df_scl data loaded.")
    st.dataframe(scenario_df_scl.head())
    save_data_to_snowflake(scenario_df_scl, "STAGING_STORE.CALC_CR_SCL_STAGING")   

    st.success("Scenario data generated and saved.")

def render_scenarios_simulation_page():
    st.title("Simulation Scenarios 🌍")

    st.write("Please select time periods for simulation (seconds).")
    # Handling custom time periods logic
    time_period_raw = st.text_input('Time period:', value='100')
    time_periods = int(time_period_raw)   

    col1, col2 = st.columns(2)
    with col1:
        if st.button("RAPID DECLINE 📉"):
            generate_rapid_decline_scenario(time_periods)

        if st.button("SUSTAINABLE GROWTH 🌱"):
            generate_sustainable_growth_scenario(time_periods)
    
    with col2:
        if st.button("DECLINE OVER TIME ⏳"):
            generate_decline_over_time_scenario(time_periods)

        if st.button("RAPID GROWTH 🚀"):
            generate_rapid_growth_scenario(time_periods)

    with col1:
        if st.button("UPDATE SUCCESS FACTORS TABLE 💃"):
            df = populate_hpl_sd_crs()
            st.write(f"Criterion data preview.")
            st.dataframe(df.head())
            save_data_to_snowflake(df, "ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE")
            st.write(f"Table population completed. Please proceed to visualization tab")            

    if st.button("⬅️ BACK"):
        st.session_state['page'] = 'home' 

#######################################
# EGTL EXPERIMENT PAGE
#######################################             

def render_experiment_page():
    st.title("EGTL EXPERIMENT ⚗️")

    # Handling custom time periods logic
    time_period_raw = st.text_input('Time period:', value='100')
    time_periods = int(time_period_raw)    

    model = st.radio(
        "Select GenAI model for experiment:",
        options=["gpt-3.5-turbo", "gpt-4", ],
        index=0
    )       
 
    if st.button("GENERATE DIRTY DATA FOR SCALABILITY 🧪"):
        raw_df = populate_calc_cr_scl_staging(time_periods)              
        save_data_to_snowflake(raw_df, "STAGING_STORE.CALC_CR_SCL_STAGING")

    if st.button("APPLY EXPLORATIVE ANALYSIS TO ETL USING GEN AI 🧩"):
        cleaned_df = normalize_cr_scl_data(model)            
        save_data_to_snowflake(cleaned_df, "STAGING_STORE.CALC_CR_SCL_STAGING")        

    if st.button("⬅️ BACK"):
        st.session_state['page'] = 'home' 

#######################################
# UTILITY PAGE
#######################################             

def render_utility_page():
    st.title("UTILITIES ")

    if st.button("BACKUP DATA 📦"):
        backup_fusion_store()
        backup_staging_store()        
        backup_alliance_store()

    if st.button("⬅️ BACK"):
        st.session_state['page'] = 'home'  

#######################################
# APPLICATION NAVIGATION
#######################################

if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    render_homepage()
elif st.session_state['page'] == 'upload_data':
    render_upload_data_page()
elif st.session_state['page'] == 'visualizations':
    render_visualizations_page()
elif st.session_state['page'] == 'scenarious':
    render_scenarios_simulation_page()
elif st.session_state['page'] == 'experiment':
    render_experiment_page()    
elif st.session_state['page'] == 'utility':
    render_utility_page()    
