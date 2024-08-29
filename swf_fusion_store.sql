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
