-- ALLIANCE STORE CREATION;

CREATE SCHEMA IF NOT EXISTS ALLIANCE_STORE;

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE (
    TIME INT,
    CR_ENV FLOAT,
    CR_SAC FLOAT,
    CR_TFE FLOAT,
    CR_SFY FLOAT,
    CR_REG FLOAT,
    CR_QMF FLOAT,
    CR_ECV FLOAT,
    CR_USB FLOAT,
    CR_RLB FLOAT,
    CR_INF FLOAT,
    CR_SCL FLOAT
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.PROJECT_STATUS;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.PROJECT_STATUS (
    HISTORY_DATE VARCHAR,
    PROJECT_STATUS VARCHAR,
    REPORTER VARCHAR    
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUANTATIVE_DATA_EXPERIMENT;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUANTATIVE_DATA_EXPERIMENT (
    ID INT,
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,   
    INPUT_DF_VOLUME FLOAT, 
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT,
    OUTPUT_DF_VOLUME FLOAT,    
    CORRECTNESS VARCHAR,
    ERROR_ENCOUNTERED BOOLEAN,
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUALITATIVE_DATA_EXPERIMENT;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUALITATIVE_DATA_EXPERIMENT (
    ID INT,
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,   
    INPUT_DF_VOLUME FLOAT, 
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT,  
    LOADED_SCENARIO VARCHAR,
    STATUS_RESULT VARCHAR,  
    ERROR_ENCOUNTERED BOOLEAN,
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_FUSION_STORE_EXPERIMENT;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_FUSION_STORE_EXPERIMENT (
    ID INT,
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,   
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT,
    OUTPUT_DF_VOLUME FLOAT,
    LOAD_TO_STAGING_TIME FLOAT,    
    CORRECTNESS VARCHAR,
    ERROR_ENCOUNTERED BOOLEAN,
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_EXTRACT_DATA_EXPERIMENT;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_EXTRACT_DATA_EXPERIMENT (
    ID INT,
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,    
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT,
    LOAD_TO_STAGING_TIME FLOAT,   
    CORRECTNESS VARCHAR,  
    ERROR_ENCOUNTERED BOOLEAN,
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HYPERLOOP_SPECIFICATION_ALLIANCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HYPERLOOP_SPECIFICATION_ALLIANCE (
    PARAMETER VARCHAR,
    SPECIFICATION VARCHAR
);

--
-- BACKUP TABLES SCRIPT
--

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HPL_SD_CRS_ALLIANCE_BCK (
    TIME INT,
    CR_ENV FLOAT,
    CR_SAC FLOAT,
    CR_TFE FLOAT,
    CR_SFY FLOAT,
    CR_REG FLOAT,
    CR_QMF FLOAT,
    CR_ECV FLOAT,
    CR_USB FLOAT,
    CR_RLB FLOAT,
    CR_INF FLOAT,
    CR_SCL FLOAT
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.PROJECT_STATUS_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.PROJECT_STATUS_BCK (
    HISTORY_DATE VARCHAR,
    PROJECT_STATUS VARCHAR,
    REPORTER VARCHAR    
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUANTATIVE_DATA_EXPERIMENT_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUANTATIVE_DATA_EXPERIMENT_BCK (
    ID INT,    
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,   
    INPUT_DF_VOLUME FLOAT, 
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT,
    OUTPUT_DF_VOLUME FLOAT,    
    CORRECTNESS VARCHAR,
    ERROR_ENCOUNTERED BOOLEAN,    
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_FUSION_STORE_EXPERIMENT_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_FUSION_STORE_EXPERIMENT_BCK (
    ID INT,
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,   
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT,
    OUTPUT_DF_VOLUME FLOAT,
    LOAD_TO_STAGING_TIME FLOAT,    
    CORRECTNESS VARCHAR,
    ERROR_ENCOUNTERED BOOLEAN,
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUALITATIVE_DATA_EXPERIMENT_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_QUALITATIVE_DATA_EXPERIMENT_BCK (
    ID INT,
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,   
    INPUT_DF_VOLUME FLOAT, 
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT,  
    LOADED_SCENARIO VARCHAR,
    STATUS_RESULT VARCHAR,  
    ERROR_ENCOUNTERED BOOLEAN,
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_EXTRACT_DATA_EXPERIMENT_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.EGTL_EXTRACT_DATA_EXPERIMENT_BCK (
    ID INT,
    MODEL VARCHAR,    
    EXPERIMENT_START_DATE VARCHAR,
    EXPERIMENT_END_DATE VARCHAR,
    MODEL_WORK_TIME FLOAT,
    SAVE_DATA_TO_SNOWFLAKE_TIME FLOAT,
    EXPERIMENT_TIME_TOTAL FLOAT,
    ROWS_PROCESSED INT,   
    PROMPT_VOLUME FLOAT,
    OUTPUT_VOLUME FLOAT, 
    LOAD_TO_STAGING_TIME FLOAT,  
    CORRECTNESS VARCHAR,  
    ERROR_ENCOUNTERED BOOLEAN,
    ERROR_TYPE VARCHAR,
    ERROR_MESSAGE VARCHAR   
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HYPERLOOP_SPECIFICATION_ALLIANCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.ALLIANCE_STORE.HYPERLOOP_SPECIFICATION_ALLIANCE_BCK (
    PARAMETER VARCHAR,
    SPECIFICATION VARCHAR
);