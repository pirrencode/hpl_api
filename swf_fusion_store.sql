-- FUSION STORE CREATION

CREATE SCHEMA IF NOT EXISTS FUSION_STORE;

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ENV_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ENV_SOURCE (
    TIME INT,
    ENERGY_CONSUMED FLOAT,  -- I
    DISTANCE FLOAT,          -- S
    LOAD_WEIGHT FLOAT,       -- L
    CO2_EMISSIONS FLOAT,     -- K
    MATERIAL_SUSTAINABILITY FLOAT, -- M
    ENV_IMPACT_SCORE FLOAT   -- Y
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SAC_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SAC_SOURCE (
    TIME INT,
    POSITIVE_FEEDBACK FLOAT,  -- A
    NEGATIVE_FEEDBACK FLOAT   -- B
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_TFE_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_TFE_SOURCE (
    TIME INT,
    CURRENT_TRL FLOAT,        -- T
    TARGET_TRL FLOAT,         -- P
    ENG_CHALLENGES_RESOLVED FLOAT, -- L
    TARGET_ENG_CHALLENGES FLOAT    -- C
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SFY_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SFY_SOURCE (
    TIME INT,
    RISK_SCORE FLOAT,         -- Rn
    MIN_RISK_SCORE FLOAT,     -- MinRn
    MAX_RISK_SCORE FLOAT      -- MaxRn
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_REG_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_REG_SOURCE (
    TIME INT,
    ETHICAL_COMPLIANCE FLOAT, -- EC
    LEGAL_COMPLIANCE FLOAT,   -- LC
    LAND_USAGE_COMPLIANCE FLOAT, -- LU
    INT_LAW_COMPLIANCE FLOAT, -- IL
    TRL_COMPLIANCE FLOAT      -- TRL
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_QMF_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_QMF_SOURCE (
    TIME INT,
    MAGLEV_LEVITATION BOOLEAN,
    AMBIENT_INTELLIGENCE BOOLEAN,
    GENERATIVE_AI BOOLEAN,
    AI_MACHINE_LEARNING BOOLEAN,
    DIGITAL_TWINS BOOLEAN,
    FIVE_G BOOLEAN,
    QUANTUM_COMPUTING BOOLEAN,
    AUGMENTED_REALITY BOOLEAN,
    VIRTUAL_REALITY BOOLEAN,
    PRINTING_AT_SCALE BOOLEAN,
    BLOCKCHAIN BOOLEAN,
    SELF_DRIVING_AUTONOMOUS_VEHICLES BOOLEAN,
    TOTAL_DISRUPTIVE_TECH INT
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ECV_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ECV_SOURCE (
    TIME INT,
    REVENUE FLOAT,             -- Rev
    OPEX FLOAT,                -- OpEx
    CAPEX FLOAT,               -- CapEx
    DISCOUNT_RATE FLOAT,       -- r
    PROJECT_LIFETIME INT       -- t
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_USB_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_USB_SOURCE (
    TIME INT,
    PRODUCTION_OUTPUT FLOAT,   -- p
    USER_EXP_RATIO FLOAT,      -- e
    ACCESSIBILITY_AGEING FLOAT -- a
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_RLB_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_RLB_SOURCE (
    TIME INT,
    DURABILITY FLOAT,          -- d
    DIGITAL_RELIABILITY FLOAT, -- c
    WEATHER_DISASTER_RESILIENCE FLOAT, -- w
    POLLUTION_PRODUCED FLOAT   -- u
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_INF_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_INF_SOURCE (
    TIME INT,
    COMMON_INFRA_FEATURES FLOAT,  -- C
    CONSTRUCTION_BARRIERS FLOAT,  -- E
    INTERMODAL_CONNECTIONS FLOAT, -- M
    INFRA_ADAPTABILITY_FEATURES FLOAT -- A
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SCL_SOURCE;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SCL_SOURCE (
    TIME INT,
    RESOURCE_MILEAGE FLOAT,    -- L1
    PLANNED_VOLUME FLOAT,      -- Q
    ADJUSTMENT_COEF_1 FLOAT,   -- K1
    ADJUSTMENT_COEF_2 FLOAT,   -- K2
    ADJUSTMENT_COEF_3 FLOAT    -- K3
);


DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CALC_CR_SCL_FUSION;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CALC_CR_SCL_FUSION (
    TIME INT,
    CR_SCL FLOAT
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_SPECIFICATION_FUSION;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_SPECIFICATION_FUSION (
    PARAMETER VARCHAR,
    SPECIFICATION VARCHAR
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_ADVANCEMENTS_FUSION;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_ADVANCEMENTS_FUSION (
    ACTUALITY VARCHAR,
    RELATED_HYPERLOOP_VENDOR VARCHAR,
    ADVANCEMENT VARCHAR
);


CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.INPUT_DATA_FUSION (
    TIME INT,
    -- Columns from CR_ENV_SOURCE
    ENERGY_CONSUMED FLOAT,
    DISTANCE FLOAT,
    LOAD_WEIGHT FLOAT,
    CO2_EMISSIONS FLOAT,
    MATERIAL_SUSTAINABILITY FLOAT,
    ENV_IMPACT_SCORE FLOAT,
    -- Columns from CR_SAC_SOURCE
    POSITIVE_FEEDBACK FLOAT,
    NEGATIVE_FEEDBACK FLOAT,
    -- Columns from CR_TFE_SOURCE
    CURRENT_TRL FLOAT,
    TARGET_TRL FLOAT,
    ENG_CHALLENGES_RESOLVED FLOAT,
    TARGET_ENG_CHALLENGES FLOAT,
    -- Columns from CR_SFY_SOURCE
    RISK_SCORE FLOAT,
    MIN_RISK_SCORE FLOAT,
    MAX_RISK_SCORE FLOAT,
    -- Columns from CR_REG_SOURCE
    ETHICAL_COMPLIANCE FLOAT,
    LEGAL_COMPLIANCE FLOAT,
    LAND_USAGE_COMPLIANCE FLOAT,
    INT_LAW_COMPLIANCE FLOAT,
    TRL_COMPLIANCE FLOAT,
    -- Columns from CR_QMF_SOURCE
    MAGLEV_LEVITATION BOOLEAN,
    AMBIENT_INTELLIGENCE BOOLEAN,
    GENERATIVE_AI BOOLEAN,
    AI_MACHINE_LEARNING BOOLEAN,
    DIGITAL_TWINS BOOLEAN,
    FIVE_G BOOLEAN,
    QUANTUM_COMPUTING BOOLEAN,
    AUGMENTED_REALITY BOOLEAN,
    VIRTUAL_REALITY BOOLEAN,
    PRINTING_AT_SCALE BOOLEAN,
    BLOCKCHAIN BOOLEAN,
    SELF_DRIVING_AUTONOMOUS_VEHICLES BOOLEAN,
    TOTAL_DISRUPTIVE_TECH INT,
    -- Columns from CR_ECV_SOURCE
    REVENUE FLOAT,
    OPEX FLOAT,
    CAPEX FLOAT,
    DISCOUNT_RATE FLOAT,
    PROJECT_LIFETIME INT,
    -- Columns from CR_USB_SOURCE
    PRODUCTION_OUTPUT FLOAT,
    USER_EXP_RATIO FLOAT,
    ACCESSIBILITY_AGEING FLOAT,
    -- Columns from CR_RLB_SOURCE
    DURABILITY FLOAT,
    DIGITAL_RELIABILITY FLOAT,
    WEATHER_DISASTER_RESILIENCE FLOAT,
    POLLUTION_PRODUCED FLOAT,
    -- Columns from CR_INF_SOURCE
    COMMON_INFRA_FEATURES FLOAT,
    CONSTRUCTION_BARRIERS FLOAT,
    INTERMODAL_CONNECTIONS FLOAT,
    INFRA_ADAPTABILITY_FEATURES FLOAT,
    -- Columns from CR_SCL_SOURCE
    RESOURCE_MILEAGE FLOAT,
    PLANNED_VOLUME FLOAT,
    ADJUSTMENT_COEF_1 FLOAT,
    ADJUSTMENT_COEF_2 FLOAT,
    ADJUSTMENT_COEF_3 FLOAT
);

CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.INPUT_DATA_SAMPLE (
    INPUT_PARAMETERS VARCHAR,
    TIME_1 FLOAT
);

--
-- BACKUP TABLES SCRIPTS
--

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ENV_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ENV_SOURCE_BCK (
    TIME INT,
    ENERGY_CONSUMED FLOAT,  -- I
    DISTANCE FLOAT,          -- S
    LOAD_WEIGHT FLOAT,       -- L
    CO2_EMISSIONS FLOAT,     -- K
    MATERIAL_SUSTAINABILITY FLOAT, -- M
    ENV_IMPACT_SCORE FLOAT   -- Y
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SAC_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SAC_SOURCE_BCK (
    TIME INT,
    POSITIVE_FEEDBACK FLOAT,  -- A
    NEGATIVE_FEEDBACK FLOAT   -- B
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_TFE_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_TFE_SOURCE_BCK (
    TIME INT,
    CURRENT_TRL FLOAT,        -- T
    TARGET_TRL FLOAT,         -- P
    ENG_CHALLENGES_RESOLVED FLOAT, -- L
    TARGET_ENG_CHALLENGES FLOAT    -- C
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SFY_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SFY_SOURCE_BCK (
    TIME INT,
    RISK_SCORE FLOAT,         -- Rn
    MIN_RISK_SCORE FLOAT,     -- MinRn
    MAX_RISK_SCORE FLOAT      -- MaxRn
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_REG_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_REG_SOURCE_BCK (
    TIME INT,
    ETHICAL_COMPLIANCE FLOAT, -- EC
    LEGAL_COMPLIANCE FLOAT,   -- LC
    LAND_USAGE_COMPLIANCE FLOAT, -- LU
    INT_LAW_COMPLIANCE FLOAT, -- IL
    TRL_COMPLIANCE FLOAT      -- TRL
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_QMF_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_QMF_SOURCE_BCK (
    TIME INT,
    MAGLEV_LEVITATION BOOLEAN,
    AMBIENT_INTELLIGENCE BOOLEAN,
    GENERATIVE_AI BOOLEAN,
    AI_MACHINE_LEARNING BOOLEAN,
    DIGITAL_TWINS BOOLEAN,
    FIVE_G BOOLEAN,
    QUANTUM_COMPUTING BOOLEAN,
    AUGMENTED_REALITY BOOLEAN,
    VIRTUAL_REALITY BOOLEAN,
    PRINTING_AT_SCALE BOOLEAN,
    BLOCKCHAIN BOOLEAN,
    SELF_DRIVING_AUTONOMOUS_VEHICLES BOOLEAN,
    TOTAL_DISRUPTIVE_TECH INT
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ECV_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_ECV_SOURCE_BCK (
    TIME INT,
    REVENUE FLOAT,             -- Rev
    OPEX FLOAT,                -- OpEx
    CAPEX FLOAT,               -- CapEx
    DISCOUNT_RATE FLOAT,       -- r
    PROJECT_LIFETIME INT       -- t
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_USB_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_USB_SOURCE_BCK (
    TIME INT,
    PRODUCTION_OUTPUT FLOAT,   -- p
    USER_EXP_RATIO FLOAT,      -- e
    ACCESSIBILITY_AGEING FLOAT -- a
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_RLB_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_RLB_SOURCE_BCK (
    TIME INT,
    DURABILITY FLOAT,          -- d
    DIGITAL_RELIABILITY FLOAT, -- c
    WEATHER_DISASTER_RESILIENCE FLOAT, -- w
    POLLUTION_PRODUCED FLOAT   -- u
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_INF_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_INF_SOURCE_BCK (
    TIME INT,
    COMMON_INFRA_FEATURES FLOAT,  -- C
    CONSTRUCTION_BARRIERS FLOAT,  -- E
    INTERMODAL_CONNECTIONS FLOAT, -- M
    INFRA_ADAPTABILITY_FEATURES FLOAT -- A
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SCL_SOURCE_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CR_SCL_SOURCE_BCK (
    TIME INT,
    RESOURCE_MILEAGE FLOAT,    -- L1
    PLANNED_VOLUME FLOAT,      -- Q
    ADJUSTMENT_COEF_1 FLOAT,   -- K1
    ADJUSTMENT_COEF_2 FLOAT,   -- K2
    ADJUSTMENT_COEF_3 FLOAT    -- K3
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CALC_CR_SCL_FUSION_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.CALC_CR_SCL_FUSION_BCK (
    TIME INT,
    CR_SCL FLOAT
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_SPECIFICATION_FUSION_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_SPECIFICATION_FUSION_BCK (
    PARAMETER VARCHAR,
    SPECIFICATION VARCHAR
);

DROP TABLE IF EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_ADVANCEMENTS_FUSION_BCK;
CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.HYPERLOOP_ADVANCEMENTS_FUSION_BCK (
    ACTUALITY VARCHAR,
    RELATED_HYPERLOOP_VENDOR VARCHAR,
    ADVANCEMENT VARCHAR
);

CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.INPUT_DATA_FUSION_BCK (
    TIME INT,
    -- Columns from CR_ENV_SOURCE
    ENERGY_CONSUMED FLOAT,
    DISTANCE FLOAT,
    LOAD_WEIGHT FLOAT,
    CO2_EMISSIONS FLOAT,
    MATERIAL_SUSTAINABILITY FLOAT,
    ENV_IMPACT_SCORE FLOAT,
    -- Columns from CR_SAC_SOURCE
    POSITIVE_FEEDBACK FLOAT,
    NEGATIVE_FEEDBACK FLOAT,
    -- Columns from CR_TFE_SOURCE
    CURRENT_TRL FLOAT,
    TARGET_TRL FLOAT,
    ENG_CHALLENGES_RESOLVED FLOAT,
    TARGET_ENG_CHALLENGES FLOAT,
    -- Columns from CR_SFY_SOURCE
    RISK_SCORE FLOAT,
    MIN_RISK_SCORE FLOAT,
    MAX_RISK_SCORE FLOAT,
    -- Columns from CR_REG_SOURCE
    ETHICAL_COMPLIANCE FLOAT,
    LEGAL_COMPLIANCE FLOAT,
    LAND_USAGE_COMPLIANCE FLOAT,
    INT_LAW_COMPLIANCE FLOAT,
    TRL_COMPLIANCE FLOAT,
    -- Columns from CR_QMF_SOURCE
    MAGLEV_LEVITATION BOOLEAN,
    AMBIENT_INTELLIGENCE BOOLEAN,
    GENERATIVE_AI BOOLEAN,
    AI_MACHINE_LEARNING BOOLEAN,
    DIGITAL_TWINS BOOLEAN,
    FIVE_G BOOLEAN,
    QUANTUM_COMPUTING BOOLEAN,
    AUGMENTED_REALITY BOOLEAN,
    VIRTUAL_REALITY BOOLEAN,
    PRINTING_AT_SCALE BOOLEAN,
    BLOCKCHAIN BOOLEAN,
    SELF_DRIVING_AUTONOMOUS_VEHICLES BOOLEAN,
    TOTAL_DISRUPTIVE_TECH INT,
    -- Columns from CR_ECV_SOURCE
    REVENUE FLOAT,
    OPEX FLOAT,
    CAPEX FLOAT,
    DISCOUNT_RATE FLOAT,
    PROJECT_LIFETIME INT,
    -- Columns from CR_USB_SOURCE
    PRODUCTION_OUTPUT FLOAT,
    USER_EXP_RATIO FLOAT,
    ACCESSIBILITY_AGEING FLOAT,
    -- Columns from CR_RLB_SOURCE
    DURABILITY FLOAT,
    DIGITAL_RELIABILITY FLOAT,
    WEATHER_DISASTER_RESILIENCE FLOAT,
    POLLUTION_PRODUCED FLOAT,
    -- Columns from CR_INF_SOURCE
    COMMON_INFRA_FEATURES FLOAT,
    CONSTRUCTION_BARRIERS FLOAT,
    INTERMODAL_CONNECTIONS FLOAT,
    INFRA_ADAPTABILITY_FEATURES FLOAT,
    -- Columns from CR_SCL_SOURCE
    RESOURCE_MILEAGE FLOAT,
    PLANNED_VOLUME FLOAT,
    ADJUSTMENT_COEF_1 FLOAT,
    ADJUSTMENT_COEF_2 FLOAT,
    ADJUSTMENT_COEF_3 FLOAT
);

CREATE TABLE IF NOT EXISTS HPL_SYSTEM_DYNAMICS.FUSION_STORE.INPUT_DATA_SAMPLE_BCK (
    INPUT_PARAMETERS VARCHAR,
    TIME_1 FLOAT
);