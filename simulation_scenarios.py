import pandas as pd
import numpy as np
from scipy.stats import beta

####################################
# USER INPUT SCENARIO
####################################

def generate_cr_env_using_input_data(time_periods, lower_range, upper_range):
    """Generate CR_ENV data based on user input ranges."""
    return np.random.uniform(lower_range, upper_range, time_periods)

####################################
# RAPID DECLINE SCENARIO
####################################

def generate_cr_env_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.round(np.linspace(58, 20, time_periods), 3),
        "DISTANCE": np.round(np.linspace(500, 100, time_periods), 3),
        "LOAD_WEIGHT": np.round(np.linspace(11300, 3000, time_periods), 3),
        "CO2_EMISSIONS": np.round(np.linspace(100, 25, time_periods), 3),
        "MATERIAL_SUSTAINABILITY": np.round(np.linspace(0.8, 0.21, time_periods), 3),
        "ENV_IMPACT_SCORE": np.round(np.linspace(0.7, 0.18, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sac_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.round(np.linspace(100, 20, time_periods), 3),
        "NEGATIVE_FEEDBACK": np.round(np.linspace(100, 800, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_tfe_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.round(np.linspace(5, 4, time_periods), 3),
        "TARGET_TRL": np.round(np.linspace(9, 6, time_periods), 3),
        "ENG_CHALLENGES_RESOLVED": np.round(np.linspace(10, 3, time_periods), 3),
        "TARGET_ENG_CHALLENGES": np.round(np.linspace(10.5, 15, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sfy_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.round(np.linspace(0.85, 0.18, time_periods), 3),
        "MIN_RISK_SCORE": np.round(np.full(time_periods, 0.19), 3),
        "MAX_RISK_SCORE": np.round(np.full(time_periods, 0.92), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RISK_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        pert_mean = beta_pert_random(row["RISK_SCORE_lower"], row["RISK_SCORE_upper"], row["RISK_SCORE"], n)
        result_data["RISK_SCORE"].append(pert_mean)
        result_data["MIN_RISK_SCORE"].append(row["MIN_RISK_SCORE"])
        result_data["MAX_RISK_SCORE"].append(row["MAX_RISK_SCORE"])
        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_reg_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.round(np.linspace(0.8, 0.25, time_periods), 3),
        "INT_LAW_COMPLIANCE": np.round(np.linspace(0.85, 0.2, time_periods), 3),  
        "LAND_USAGE_COMPLIANCE": np.round(np.linspace(0.95, 0.3, time_periods), 3),
        "LEGAL_COMPLIANCE": np.round(np.linspace(0.5, 0.3, time_periods), 3),
        "TRL_COMPLIANCE": np.round(np.linspace(0.5, 0.4, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_qmf_data_rapid_decline(time_periods):

    tech_columns = [
        'MAGLEV_LEVITATION', 'AMBIENT_INTELLIGENCE', 'GENERATIVE_AI',
        'AI_MACHINE_LEARNING', 'DIGITAL_TWINS', 'FIVE_G', 
        'QUANTUM_COMPUTING', 'AUGMENTED_REALITY', 'VIRTUAL_REALITY',
        'PRINTING_AT_SCALE', 'BLOCKCHAIN', 'SELF_DRIVING_AUTONOMOUS_VEHICLES'
    ]
    
    time = np.arange(time_periods)
    data = {
        "TIME": time
    }

    for t in time:
        num_true = np.random.randint(4, 9)
        
        true_indices = np.random.choice(tech_columns, size=num_true, replace=False)
        
        row_data = {tech: False for tech in tech_columns}
        
        for tech in true_indices:
            row_data[tech] = True
        
        row_data["TOTAL_DISRUPTIVE_TECH"] = num_true
        
        for tech, value in row_data.items():
            if tech not in data:
                data[tech] = []
            data[tech].append(value)

    return pd.DataFrame(data)

def generate_cr_ecv_data_rapid_decline(time_periods, diff, n):
    
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.round(np.linspace(700000, 694000, time_periods), 3),
        "OPEX": np.round(np.linspace(50000, 51100, time_periods), 3),
        "CAPEX": np.round(np.full(time_periods, 14000000), 3),
        "DISCOUNT_RATE": np.round(np.linspace(0.05, 0.1, time_periods), 3),
        "PROJECT_LIFETIME": np.round(np.full(time_periods, 980), 3),
    }
    df = pd.DataFrame(data)
    
    for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_usb_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.round(np.linspace(1.0, 0.4, time_periods), 3),
        "USER_EXP_RATIO": np.round(np.linspace(1.0, 0.2, time_periods), 3),
        "ACCESSIBILITY_AGEING": np.round(np.linspace(1.0, 0.3, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_rlb_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.round(np.linspace(0.95, 0.5, time_periods), 3),
        "DIGITAL_RELIABILITY": np.round(np.linspace(0.95, 0.4, time_periods), 3),
        "WEATHER_DISASTER_RESILIENCE": np.round(np.linspace(0.95, 0.6, time_periods), 3),
        "POLLUTION_PRODUCED": np.round(np.linspace(0.3, 0.95, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_inf_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.round(np.linspace(0.95, 0.4, time_periods), 3),
        "CONSTRUCTION_BARRIERS": np.round(np.linspace(0.5, 0.95, time_periods), 3),
        "INTERMODAL_CONNECTIONS": np.round(np.linspace(0.95, 0.2, time_periods), 3),
        "INFRA_ADAPTABILITY_FEATURES": np.round(np.linspace(0.95, 0.6, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_scl_data_rapid_decline(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.round(np.linspace(0.95, 0.3, time_periods), 3),
        "PLANNED_VOLUME": np.round(np.linspace(0.95, 0.4, time_periods), 3),
        "ADJUSTMENT_COEF_1": np.round(np.linspace(0.95, 0.2, time_periods), 3),
        "ADJUSTMENT_COEF_2": np.round(np.linspace(0.95, 0.25, time_periods), 3),
        "ADJUSTMENT_COEF_3": np.round(np.linspace(0.7, 0.3, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

####################################
# DECLINE OVER TIME SCENARIO
####################################

def generate_cr_env_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.round(np.linspace(100, 75, time_periods), 3),
        "DISTANCE": np.round(np.linspace(20, 17, time_periods), 3),
        "LOAD_WEIGHT": np.round(np.linspace(8000, 6500, time_periods), 3),
        "CO2_EMISSIONS": np.round(np.linspace(75, 60, time_periods), 3),
        "MATERIAL_SUSTAINABILITY": np.round(np.linspace(0.8, 0.67, time_periods), 3),
        "ENV_IMPACT_SCORE": np.round(np.linspace(0.7, 0.61, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sac_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.round(np.linspace(100, 80, time_periods), 3),
        "NEGATIVE_FEEDBACK": np.round(np.linspace(100, 120, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_tfe_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.round(np.linspace(9, 7, time_periods), 3),
        "TARGET_TRL": np.round(np.linspace(9, 8, time_periods), 3),
        "ENG_CHALLENGES_RESOLVED": np.round(np.linspace(7, 6, time_periods), 3),
        "TARGET_ENG_CHALLENGES": np.round(np.linspace(10, 11, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sfy_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.round(np.linspace(0.9, 0.7, time_periods), 3),
        "MIN_RISK_SCORE": np.round(np.full(time_periods, 0.3), 3),
        "MAX_RISK_SCORE": np.round(np.full(time_periods, 0.9), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RISK_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        pert_mean = beta_pert_random(row["RISK_SCORE_lower"], row["RISK_SCORE_upper"], row["RISK_SCORE"], n)
        result_data["RISK_SCORE"].append(pert_mean)
        result_data["MIN_RISK_SCORE"].append(row["MIN_RISK_SCORE"])
        result_data["MAX_RISK_SCORE"].append(row["MAX_RISK_SCORE"])
        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_reg_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.round(np.linspace(0.91, 0.82, time_periods), 3),
        "INT_LAW_COMPLIANCE": np.round(np.linspace(0.88, 0.81, time_periods), 3), 
        "LAND_USAGE_COMPLIANCE": np.round(np.linspace(0.87, 0.77, time_periods), 3),
        "LEGAL_COMPLIANCE": np.round(np.linspace(0.86, 0.76, time_periods), 3),
        "TRL_COMPLIANCE": np.round(np.linspace(0.7, 0.65, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_qmf_decline_over_time_data(time_periods):

    tech_columns = [
        'MAGLEV_LEVITATION', 'AMBIENT_INTELLIGENCE', 'GENERATIVE_AI',
        'AI_MACHINE_LEARNING', 'DIGITAL_TWINS', 'FIVE_G', 
        'QUANTUM_COMPUTING', 'AUGMENTED_REALITY', 'VIRTUAL_REALITY',
        'PRINTING_AT_SCALE', 'BLOCKCHAIN', 'SELF_DRIVING_AUTONOMOUS_VEHICLES'
    ]
    
    time = np.arange(time_periods)
    data = {
        "TIME": time
    }

    for t in time:
        num_true = np.random.randint(8, 11)
        
        true_indices = np.random.choice(tech_columns, size=num_true, replace=False)
        
        row_data = {tech: False for tech in tech_columns}
        
        for tech in true_indices:
            row_data[tech] = True
        
        row_data["TOTAL_DISRUPTIVE_TECH"] = num_true
        
        for tech, value in row_data.items():
            if tech not in data:
                data[tech] = []
            data[tech].append(value)

    return pd.DataFrame(data)

def generate_cr_ecv_decline_over_time_data(time_periods, diff, n):
    
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.round(np.linspace(200000, 190000, time_periods), 3),
        "OPEX": np.round(np.linspace(6000, 6150, time_periods), 3),
        "CAPEX": np.round(np.full(time_periods, 5400000), 3),
        "DISCOUNT_RATE": np.round(np.linspace(0.055, 0.06, time_periods), 3),
        "PROJECT_LIFETIME": np.round(np.full(time_periods, 600), 3),
    }
    df = pd.DataFrame(data)
    
    for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_usb_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.round(np.linspace(0.93, 0.6, time_periods), 3),
        "USER_EXP_RATIO": np.round(np.linspace(0.93, 0.6, time_periods), 3),
        "ACCESSIBILITY_AGEING": np.round(np.linspace(0.8, 0.5, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_rlb_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.round(np.linspace(0.92, 0.6, time_periods), 3),
        "DIGITAL_RELIABILITY": np.round(np.linspace(0.93, 0.5, time_periods), 3),
        "WEATHER_DISASTER_RESILIENCE": np.round(np.linspace(0.91, 0.7, time_periods), 3),
        "POLLUTION_PRODUCED": np.round(np.linspace(0.7, 1.3, time_periods), 3)
    }
    df = pd.DataFrame(data)

    for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_inf_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.round(np.linspace(0.95, 0.65, time_periods), 3),
        "CONSTRUCTION_BARRIERS": np.round(np.linspace(0.5, 1.2, time_periods), 3),
        "INTERMODAL_CONNECTIONS": np.round(np.linspace(0.91, 0.4, time_periods), 3),
        "INFRA_ADAPTABILITY_FEATURES": np.round(np.linspace(0.92, 0.75, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_scl_decline_over_time_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.round(np.linspace(0.93, 0.5, time_periods), 3),
        "PLANNED_VOLUME": np.round(np.linspace(0.93, 0.6, time_periods), 3),
        "ADJUSTMENT_COEF_1": np.round(np.linspace(0.91, 0.7, time_periods), 3),
        "ADJUSTMENT_COEF_2": np.round(np.linspace(0.92, 0.45, time_periods), 3),
        "ADJUSTMENT_COEF_3": np.round(np.linspace(0.7, 0.38, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

##############################################
# RAPID GROWTH SCENARIO
##############################################

def generate_cr_env_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.round(np.linspace(35, 15, time_periods), 3),
        "DISTANCE": np.round(np.linspace(300, 550, time_periods), 3),
        "LOAD_WEIGHT": np.round(np.linspace(15000, 19000, time_periods), 3),
        "CO2_EMISSIONS": np.round(np.linspace(50, 20, time_periods), 3),
        "MATERIAL_SUSTAINABILITY": np.round(np.linspace(0.8, 0.90, time_periods), 3),
        "ENV_IMPACT_SCORE": np.round(np.linspace(0.75, 0.95, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sac_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.round(np.linspace(100, 105, time_periods), 3),
        "NEGATIVE_FEEDBACK": np.round(np.linspace(210, 107, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_tfe_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.round(np.linspace(6, 8, time_periods), 3),
        "TARGET_TRL": np.round(np.linspace(7, 9, time_periods), 3),
        "ENG_CHALLENGES_RESOLVED": np.round(np.linspace(8, 13, time_periods), 3),
        "TARGET_ENG_CHALLENGES": np.round(np.linspace(17, 14, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sfy_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.round(np.linspace(0.65, 0.95, time_periods), 3),
        "MIN_RISK_SCORE": np.round(np.full(time_periods, 0.3), 3),
        "MAX_RISK_SCORE": np.round(np.full(time_periods, 0.9), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RISK_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        pert_mean = beta_pert_random(row["RISK_SCORE_lower"], row["RISK_SCORE_upper"], row["RISK_SCORE"], n)
        result_data["RISK_SCORE"].append(pert_mean)
        result_data["MIN_RISK_SCORE"].append(row["MIN_RISK_SCORE"])
        result_data["MAX_RISK_SCORE"].append(row["MAX_RISK_SCORE"])
        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_reg_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.round(np.linspace(0.65, 0.95, time_periods), 3),
        "INT_LAW_COMPLIANCE": np.round(np.linspace(0.5, 0.93, time_periods), 3),
        "LAND_USAGE_COMPLIANCE": np.round(np.linspace(0.75, 0.98, time_periods), 3),
        "LEGAL_COMPLIANCE": np.round(np.linspace(0.65, 0.94, time_periods), 3),
        "TRL_COMPLIANCE": np.round(np.linspace(0.7, 0.9, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_qmf_rapid_growth_data(time_periods):
    tech_columns = [
        'MAGLEV_LEVITATION', 'AMBIENT_INTELLIGENCE', 'GENERATIVE_AI',
        'AI_MACHINE_LEARNING', 'DIGITAL_TWINS', 'FIVE_G', 
        'QUANTUM_COMPUTING', 'AUGMENTED_REALITY', 'VIRTUAL_REALITY',
        'PRINTING_AT_SCALE', 'BLOCKCHAIN', 'SELF_DRIVING_AUTONOMOUS_VEHICLES'
    ]
    
    time = np.arange(time_periods)
    data = {
        "TIME": time
    }

    for t in time:
        num_true = np.random.randint(5, 12)
        
        true_indices = np.random.choice(tech_columns, size=num_true, replace=False)
        
        row_data = {tech: False for tech in tech_columns}
        
        for tech in true_indices:
            row_data[tech] = True
        
        row_data["TOTAL_DISRUPTIVE_TECH"] = num_true
        
        for tech, value in row_data.items():
            if tech not in data:
                data[tech] = []
            data[tech].append(value)

    return pd.DataFrame(data)

def generate_cr_ecv_rapid_growth_data(time_periods, diff, n):
    
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.round(np.linspace(350000, 355000, time_periods), 3),
        "OPEX": np.round(np.linspace(300000, 295000, time_periods), 3),
        "CAPEX": np.round(np.full(time_periods, 1200000), 3),
        "DISCOUNT_RATE": np.round(np.linspace(0.06, 0.055, time_periods), 3),
        "PROJECT_LIFETIME": np.round(np.full(time_periods, 480), 3),
    }
    df = pd.DataFrame(data)
    
    for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_usb_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.round(np.linspace(0.4, 0.95, time_periods), 3),
        "USER_EXP_RATIO": np.round(np.linspace(0.5, 0.95, time_periods), 3),
        "ACCESSIBILITY_AGEING": np.round(np.linspace(0.56, 0.95, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_rlb_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.round(np.linspace(0.4, 0.7, time_periods), 3),
        "DIGITAL_RELIABILITY": np.round(np.linspace(0.35, 0.75, time_periods), 3),
        "WEATHER_DISASTER_RESILIENCE": np.round(np.linspace(0.8, 0.85, time_periods), 3),
        "POLLUTION_PRODUCED": np.round(np.linspace(0.6, 0.3, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_inf_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.round(np.linspace(0.4, 0.5, time_periods), 3),
        "CONSTRUCTION_BARRIERS": np.round(np.linspace(0.6, 0.4, time_periods), 3),
        "INTERMODAL_CONNECTIONS": np.round(np.linspace(0.4, 0.75, time_periods), 3),
        "INFRA_ADAPTABILITY_FEATURES": np.round(np.linspace(0.2, 0.65, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_scl_rapid_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.round(np.linspace(0.3, 0.75, time_periods), 3),
        "PLANNED_VOLUME": np.round(np.linspace(0.2, 0.75, time_periods), 3),
        "ADJUSTMENT_COEF_1": np.round(np.linspace(0.45, 0.75, time_periods), 3),
        "ADJUSTMENT_COEF_2": np.round(np.linspace(0.45, 0.75, time_periods), 3),
        "ADJUSTMENT_COEF_3": np.round(np.linspace(0.45, 0.75, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

##########################################
# SUSTAINABLE GROWTH SCENARIO
##########################################

def generate_cr_env_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.round(np.linspace(100, 110, time_periods), 3),
        "DISTANCE": np.round(np.linspace(1000, 1050, time_periods), 3),
        "LOAD_WEIGHT": np.round(np.linspace(8500, 8950, time_periods), 3),
        "CO2_EMISSIONS": np.round(np.linspace(500, 300, time_periods), 3),
        "MATERIAL_SUSTAINABILITY": np.round(np.linspace(0.81, 0.86, time_periods), 3),
        "ENV_IMPACT_SCORE": np.round(np.linspace(0.83, 0.95, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", "ENV_IMPACT_SCORE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sac_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.round(np.linspace(135, 155, time_periods), 3),
        "NEGATIVE_FEEDBACK": np.round(np.linspace(90, 75, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["POSITIVE_FEEDBACK", "NEGATIVE_FEEDBACK"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_tfe_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.round(np.linspace(8, 8.5, time_periods), 3),
        "TARGET_TRL": np.round(np.linspace(8.5, 8.8, time_periods), 3),
        "ENG_CHALLENGES_RESOLVED": np.round(np.linspace(9, 10, time_periods), 3),
        "TARGET_ENG_CHALLENGES": np.round(np.linspace(13, 12, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["CURRENT_TRL", "TARGET_TRL", "ENG_CHALLENGES_RESOLVED", "TARGET_ENG_CHALLENGES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_sfy_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.round(np.linspace(0.55, 0.65, time_periods), 3),
        "MIN_RISK_SCORE": np.round(np.full(time_periods, 0.3), 3),
        "MAX_RISK_SCORE": np.round(np.full(time_periods, 0.9), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RISK_SCORE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        pert_mean = beta_pert_random(row["RISK_SCORE_lower"], row["RISK_SCORE_upper"], row["RISK_SCORE"], n)
        result_data["RISK_SCORE"].append(pert_mean)
        result_data["MIN_RISK_SCORE"].append(row["MIN_RISK_SCORE"])
        result_data["MAX_RISK_SCORE"].append(row["MAX_RISK_SCORE"])
        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_reg_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.round(np.linspace(0.65, 0.7, time_periods), 3),
        "INT_LAW_COMPLIANCE": np.round(np.linspace(0.55, 0.6, time_periods), 3),
        "LAND_USAGE_COMPLIANCE": np.round(np.linspace(0.7, 0.74, time_periods), 3),
        "LEGAL_COMPLIANCE": np.round(np.linspace(0.6, 0.65, time_periods), 3),
        "TRL_COMPLIANCE": np.round(np.linspace(0.62, 0.66, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["ETHICAL_COMPLIANCE", "INT_LAW_COMPLIANCE", "LAND_USAGE_COMPLIANCE", "LEGAL_COMPLIANCE", "TRL_COMPLIANCE"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_qmf_sustainable_growth_data(time_periods):
    tech_columns = [
        'MAGLEV_LEVITATION', 'AMBIENT_INTELLIGENCE', 'GENERATIVE_AI',
        'AI_MACHINE_LEARNING', 'DIGITAL_TWINS', 'FIVE_G', 
        'QUANTUM_COMPUTING', 'AUGMENTED_REALITY', 'VIRTUAL_REALITY',
        'PRINTING_AT_SCALE', 'BLOCKCHAIN', 'SELF_DRIVING_AUTONOMOUS_VEHICLES'
    ]
    
    time = np.arange(time_periods)
    data = {
        "TIME": time
    }

    for t in time:
        num_true = np.random.randint(9, 11)
        
        true_indices = np.random.choice(tech_columns, size=num_true, replace=False)
        
        row_data = {tech: False for tech in tech_columns}
        
        for tech in true_indices:
            row_data[tech] = True
        
        row_data["TOTAL_DISRUPTIVE_TECH"] = num_true
        
        for tech, value in row_data.items():
            if tech not in data:
                data[tech] = []
            data[tech].append(value)

    return pd.DataFrame(data)

def generate_cr_ecv_sustainable_growth_data(time_periods, diff, n):
    
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.round(np.linspace(40000, 40300, time_periods), 3),
        "OPEX": np.round(np.linspace(30000, 29700, time_periods), 3),
        "CAPEX": np.round(np.full(time_periods, 225000), 3),
        "DISCOUNT_RATE": np.round(np.linspace(0.07, 0.068, time_periods), 3),
        "PROJECT_LIFETIME": np.round(np.full(time_periods, 700), 3),
    }
    df = pd.DataFrame(data)
    
    for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["REVENUE", "OPEX", "CAPEX", "DISCOUNT_RATE", "PROJECT_LIFETIME"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_usb_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.round(np.linspace(0.7, 0.8, time_periods), 3),
        "USER_EXP_RATIO": np.round(np.linspace(0.82, 0.87, time_periods), 3),
        "ACCESSIBILITY_AGEING": np.round(np.linspace(0.6, 0.7, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["PRODUCTION_OUTPUT", "USER_EXP_RATIO", "ACCESSIBILITY_AGEING"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_rlb_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.round(np.linspace(0.9, 0.95, time_periods), 3),
        "DIGITAL_RELIABILITY": np.round(np.linspace(0.8, 0.85, time_periods), 3),
        "WEATHER_DISASTER_RESILIENCE": np.round(np.linspace(0.62, 0.56, time_periods), 3),
        "POLLUTION_PRODUCED": np.round(np.linspace(0.55, 0.5, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["DURABILITY", "DIGITAL_RELIABILITY", "WEATHER_DISASTER_RESILIENCE", "POLLUTION_PRODUCED"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_inf_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.round(np.linspace(0.3, 0.4, time_periods), 3),
        "CONSTRUCTION_BARRIERS": np.round(np.linspace(0.55, 0.45, time_periods), 3),
        "INTERMODAL_CONNECTIONS": np.round(np.linspace(0.71, 0.72, time_periods), 3),
        "INFRA_ADAPTABILITY_FEATURES": np.round(np.linspace(0.8, 0.95, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["COMMON_INFRA_FEATURES", "CONSTRUCTION_BARRIERS", "INTERMODAL_CONNECTIONS", "INFRA_ADAPTABILITY_FEATURES"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)

def generate_cr_scl_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.round(np.linspace(0.78, 0.8, time_periods), 3),
        "PLANNED_VOLUME": np.round(np.linspace(0.8, 0.75, time_periods), 3),
        "ADJUSTMENT_COEF_1": np.round(np.linspace(0.75, 0.7, time_periods), 3),
        "ADJUSTMENT_COEF_2": np.round(np.linspace(0.7, 0.72, time_periods), 3),
        "ADJUSTMENT_COEF_3": np.round(np.linspace(0.6, 0.65, time_periods), 3),
    }
    return pd.DataFrame(data)

def generate_cr_scl_sustainable_growth_data(time_periods, diff, n):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.round(np.linspace(0.78, 0.8, time_periods), 3),
        "PLANNED_VOLUME": np.round(np.linspace(0.8, 0.75, time_periods), 3),
        "ADJUSTMENT_COEF_1": np.round(np.linspace(0.75, 0.7, time_periods), 3),
        "ADJUSTMENT_COEF_2": np.round(np.linspace(0.7, 0.72, time_periods), 3),
        "ADJUSTMENT_COEF_3": np.round(np.linspace(0.6, 0.65, time_periods), 3),
    }
    df = pd.DataFrame(data)

    for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
        df[f"{col}_lower"] = df[col] * (1 - diff)
        df[f"{col}_upper"] = df[col] * (1 + diff)

    def beta_pert_random(min_val, max_val, mode, n, lmbda=4):
        """Monte Carlo Beta-PERT simulation."""
        alpha = 1 + lmbda * (mode - min_val) / (max_val - min_val)
        beta_param = 1 + lmbda * (max_val - mode) / (max_val - min_val)
        samples = beta.rvs(alpha, beta_param, size=n, scale=(max_val - min_val)) + min_val
        return np.mean(samples)

    result_data = {col: [] for col in df.columns if not col.endswith(("_lower", "_upper"))}

    for _, row in df.iterrows():
        for col in ["RESOURCE_MILEAGE", "PLANNED_VOLUME", "ADJUSTMENT_COEF_1", "ADJUSTMENT_COEF_2", "ADJUSTMENT_COEF_3"]:
            pert_mean = beta_pert_random(row[f"{col}_lower"], row[f"{col}_upper"], row[col], n)
            result_data[col].append(pert_mean)

        result_data["TIME"].append(row["TIME"])

    return pd.DataFrame(result_data)
