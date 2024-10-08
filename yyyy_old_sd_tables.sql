CREATE SCHEMA IF NOT EXISTS FUSION_STORE;

CREATE SCHEMA IF NOT EXISTS STAGING_STORE;

CREATE SCHEMA IF NOT EXISTS ALLIANCE_STORE;



CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_ENV_SOURCE (
    TIME INT,
    ENERGY_CONSUMED FLOAT,  -- I
    DISTANCE FLOAT,          -- S
    LOAD_WEIGHT FLOAT,       -- L
    CO2_EMISSIONS FLOAT,     -- K
    MATERIAL_SUSTAINABILITY FLOAT, -- M
    ENV_IMPACT_SCORE FLOAT   -- Y
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_ENV (
    TIME INT,
    CR_ENV FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_SAC_SOURCE (
    TIME INT,
    POSITIVE_FEEDBACK FLOAT,  -- A
    NEGATIVE_FEEDBACK FLOAT   -- B
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_SAC (
    TIME INT,
    CR_SAC FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_TFE_SOURCE (
    TIME INT,
    CURRENT_TRL FLOAT,        -- T
    TARGET_TRL FLOAT,         -- P
    ENG_CHALLENGES_RESOLVED FLOAT, -- L
    TARGET_ENG_CHALLENGES FLOAT    -- C
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_TFE (
    TIME INT,
    CR_TFE FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_SFY_SOURCE (
    TIME INT,
    RISK_SCORE FLOAT,         -- Rn
    MIN_RISK_SCORE FLOAT,     -- MinRn
    MAX_RISK_SCORE FLOAT      -- MaxRn
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_SFY (
    TIME INT,
    CR_SFY FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_REG_SOURCE (
    TIME INT,
    ETHICAL_COMPLIANCE FLOAT, -- EC
    LEGAL_COMPLIANCE FLOAT,   -- LC
    LAND_USAGE_COMPLIANCE FLOAT, -- LU
    INT_LAW_COMPLIANCE FLOAT, -- IL
    TRL_COMPLIANCE FLOAT      -- TRL
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_REG (
    TIME INT,
    CR_REG FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_QMF_SOURCE (
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

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_QMF (
    TIME INT,
    CR_QMF FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_ECV_SOURCE (
    TIME INT,
    REVENUE FLOAT,             -- Rev
    OPEX FLOAT,                -- OpEx
    CAPEX FLOAT,               -- CapEx
    DISCOUNT_RATE FLOAT,       -- r
    PROJECT_LIFETIME INT       -- t
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_ECV (
    TIME INT,
    CR_ECV FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_USB_SOURCE (
    TIME INT,
    PRODUCTION_OUTPUT FLOAT,   -- p
    USER_EXP_RATIO FLOAT,      -- e
    ACCESSIBILITY_AGEING FLOAT -- a
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_USB (
    TIME INT,
    CR_USB FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_RLB_SOURCE (
    TIME INT,
    DURABILITY FLOAT,          -- d
    DIGITAL_RELIABILITY FLOAT, -- c
    WEATHER_DISASTER_RESILIENCE FLOAT, -- w
    POLLUTION_PRODUCED FLOAT   -- u
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_RLB (
    TIME INT,
    CR_RLB FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_INF_SOURCE (
    TIME INT,
    COMMON_INFRA_FEATURES FLOAT,  -- C
    CONSTRUCTION_BARRIERS FLOAT,  -- E
    INTERMODAL_CONNECTIONS FLOAT, -- M
    INFRA_ADAPTABILITY_FEATURES FLOAT -- A
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CALC_CR_INF (
    TIME INT,
    CR_INF FLOAT
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.CR_SCL_SOURCE (
    TIME INT,
    RESOURCE_MILEAGE FLOAT,    -- L1
    PLANNED_VOLUME FLOAT,      -- Q
    ADJUSTMENT_COEF_1 FLOAT,   -- K1
    ADJUSTMENT_COEF_2 FLOAT,   -- K2
    ADJUSTMENT_COEF_3 FLOAT    -- K3
);

CREATE TABLE HPL_SYSTEM_DYNAMICS.SYSTEM_DYNAMICS.HPL_SD_CRS (
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
