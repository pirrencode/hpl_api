import pandas as pd
import numpy as np

####################################
# USER INPUT SCENARIO
####################################

def generate_cr_env_using_input_data(time_periods, lower_range, upper_range):
    """Generate CR_ENV data based on user input ranges."""
    return np.random.uniform(lower_range, upper_range, time_periods)

####################################
# RAPID DECLINE SCENARIO
####################################

def generate_cr_env_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.linspace(100, 20, time_periods),
        "DISTANCE": np.linspace(1000, 200, time_periods),
        "LOAD_WEIGHT": np.linspace(15000, 3000, time_periods),
        "CO2_EMISSIONS": np.linspace(500, 100, time_periods),
        "MATERIAL_SUSTAINABILITY": np.linspace(0.8, 0.21, time_periods),
        "ENV_IMPACT_SCORE": np.linspace(0.7, 0.18, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sac_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.linspace(100, 20, time_periods),
        "NEGATIVE_FEEDBACK": np.linspace(100, 800, time_periods),
    }
    return pd.DataFrame(data)

def generate_cr_tfe_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.linspace(9, 6, time_periods),
        "TARGET_TRL": np.linspace(9, 7, time_periods),
        "ENG_CHALLENGES_RESOLVED": np.linspace(10, 3, time_periods),
        "TARGET_ENG_CHALLENGES": np.linspace(5, 15, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sfy_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.linspace(0.85, 0.18, time_periods),
        "MIN_RISK_SCORE": np.full(time_periods, 0.19),
        "MAX_RISK_SCORE": np.full(time_periods, 0.92)
    }
    return pd.DataFrame(data)

def generate_cr_reg_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.linspace(0.9, 0.3, time_periods),
        "INT_LAW_COMPLIANCE": np.linspace(0.9, 0.3, time_periods),  
        "LAND_USAGE_COMPLIANCE": np.linspace(0.9, 0.3, time_periods),
        "LEGAL_COMPLIANCE": np.linspace(0.9, 0.3, time_periods),
        "TRL_COMPLIANCE": np.linspace(0.6, 0.5, time_periods)
    }
    return pd.DataFrame(data)

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

def generate_cr_ecv_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.linspace(500000, 100000, time_periods),
        "OPEX": np.linspace(200000, 500000, time_periods),
        "CAPEX": np.full(time_periods, 1000000),
        "DISCOUNT_RATE": np.linspace(0.05, 0.1, time_periods),
        "PROJECT_LIFETIME": np.full(time_periods, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_usb_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.0, 0.4, time_periods),
        "USER_EXP_RATIO": np.linspace(1.0, 0.2, time_periods),
        "ACCESSIBILITY_AGEING": np.linspace(1.0, 0.3, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_rlb_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.5, 0.5, time_periods),
        "DIGITAL_RELIABILITY": np.linspace(1.3, 0.4, time_periods),
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.2, 0.6, time_periods),
        "POLLUTION_PRODUCED": np.linspace(0.7, 1.8, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_inf_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.5, 0.5, time_periods),
        "CONSTRUCTION_BARRIERS": np.linspace(0.5, 1.5, time_periods),
        "INTERMODAL_CONNECTIONS": np.linspace(1.5, 0.5, time_periods),
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.5, 0.5, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_scl_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.5, 0.5, time_periods),
        "PLANNED_VOLUME": np.linspace(1.5, 0.5, time_periods),
        "ADJUSTMENT_COEF_1": np.linspace(1.5, 0.5, time_periods),
        "ADJUSTMENT_COEF_2": np.linspace(1.0, 0.25, time_periods),
        "ADJUSTMENT_COEF_3": np.linspace(0.7, 0.3, time_periods)
    }
    return pd.DataFrame(data)

####################################
# DECLINE OVER TIME SCENARIO
####################################

def generate_cr_env_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.linspace(100, 75, time_periods),
        "DISTANCE": np.linspace(1000, 810, time_periods),
        "LOAD_WEIGHT": np.linspace(15000, 12300, time_periods),
        "CO2_EMISSIONS": np.linspace(500, 400, time_periods),
        "MATERIAL_SUSTAINABILITY": np.linspace(0.8, 0.67, time_periods),
        "ENV_IMPACT_SCORE": np.linspace(0.7, 0.61, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sac_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.linspace(100, 80, time_periods),
        "NEGATIVE_FEEDBACK": np.linspace(100, 120, time_periods),
    }
    return pd.DataFrame(data)

def generate_cr_tfe_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.linspace(9, 7, time_periods),
        "TARGET_TRL": np.linspace(9, 8, time_periods),
        "ENG_CHALLENGES_RESOLVED": np.linspace(10, 8, time_periods),
        "TARGET_ENG_CHALLENGES": np.linspace(5, 7, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sfy_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.linspace(0.9, 0.7, time_periods),
        "MIN_RISK_SCORE": np.full(time_periods, 0.3),
        "MAX_RISK_SCORE": np.full(time_periods, 0.9)
    }
    return pd.DataFrame(data)

def generate_cr_reg_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.linspace(0.91, 0.82, time_periods),
        "INT_LAW_COMPLIANCE": np.linspace(0.88, 0.81, time_periods),  
        "LAND_USAGE_COMPLIANCE": np.linspace(0.87, 0.77, time_periods),
        "LEGAL_COMPLIANCE": np.linspace(0.86, 0.76, time_periods),
        "TRL_COMPLIANCE": np.linspace(0.7, 0.65, time_periods)
    }
    return pd.DataFrame(data)

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

def generate_cr_ecv_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.linspace(500000, 400000, time_periods),
        "OPEX": np.linspace(200000, 300000, time_periods),
        "CAPEX": np.full(time_periods, 1000000),
        "DISCOUNT_RATE": np.linspace(0.05, 0.04, time_periods),
        "PROJECT_LIFETIME": np.full(time_periods, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_usb_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.5, 0.9, time_periods),
        "USER_EXP_RATIO": np.linspace(1.3, 0.8, time_periods),
        "ACCESSIBILITY_AGEING": np.linspace(1.0, 0.8, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_rlb_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.5, 1.3, time_periods),
        "DIGITAL_RELIABILITY": np.linspace(1.5, 1.2, time_periods),
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.5, 1.4, time_periods),
        "POLLUTION_PRODUCED": np.linspace(0.5, 0.6, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_inf_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.5, 1.2, time_periods),
        "CONSTRUCTION_BARRIERS": np.linspace(0.5, 0.6, time_periods),
        "INTERMODAL_CONNECTIONS": np.linspace(1.5, 1.2, time_periods),
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.5, 1.2, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_scl_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.5, 1.2, time_periods),
        "PLANNED_VOLUME": np.linspace(1.5, 1.2, time_periods),
        "ADJUSTMENT_COEF_1": np.linspace(1.5, 1.4, time_periods),
        "ADJUSTMENT_COEF_2": np.linspace(0.5, 0.3, time_periods),
        "ADJUSTMENT_COEF_3": np.linspace(2.5, 2.3, time_periods)
    }
    return pd.DataFrame(data)

##############################################
# RAPID GROWTH SCENARIO
##############################################

def generate_cr_env_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.linspace(100, 150, time_periods),
        "DISTANCE": np.linspace(1000, 1500, time_periods),
        "LOAD_WEIGHT": np.linspace(1150, 1400, time_periods),
        "CO2_EMISSIONS": np.linspace(500, 200, time_periods),
        "MATERIAL_SUSTAINABILITY": np.linspace(0.8, 0.90, time_periods),
        "ENV_IMPACT_SCORE": np.linspace(0.75, 0.95, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sac_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.linspace(1000, 1500, time_periods),
        "NEGATIVE_FEEDBACK": np.linspace(200, 100, time_periods),
    }
    return pd.DataFrame(data)

def generate_cr_tfe_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.linspace(6, 8, time_periods),
        "TARGET_TRL": np.linspace(7, 9, time_periods),
        "ENG_CHALLENGES_RESOLVED": np.linspace(8, 12, time_periods),
        "TARGET_ENG_CHALLENGES": np.linspace(5, 2, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sfy_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.linspace(0.65, 0.95, time_periods),
        "MIN_RISK_SCORE": np.full(time_periods, 0.3),
        "MAX_RISK_SCORE": np.full(time_periods, 0.9)
    }
    return pd.DataFrame(data)

def generate_cr_reg_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.linspace(0.65, 0.95, time_periods),
        "INT_LAW_COMPLIANCE": np.linspace(0.5, 0.93, time_periods),  
        "LAND_USAGE_COMPLIANCE": np.linspace(0.75, 0.98, time_periods),
        "LEGAL_COMPLIANCE": np.linspace(0.65, 0.94, time_periods),
        "TRL_COMPLIANCE": np.linspace(0.7, 0.9, time_periods)
    }
    return pd.DataFrame(data)

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

def generate_cr_ecv_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.linspace(600000, 1000000, time_periods),
        "OPEX": np.linspace(300000, 150000, time_periods),
        "CAPEX": np.full(time_periods, 1000000),
        "DISCOUNT_RATE": np.linspace(0.06, 0.03, time_periods),
        "PROJECT_LIFETIME": np.full(time_periods, 25)
    }
    return pd.DataFrame(data)

def generate_cr_usb_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(0.4, 1.8, time_periods),
        "USER_EXP_RATIO": np.linspace(0.5, 1.6, time_periods),
        "ACCESSIBILITY_AGEING": np.linspace(0.56, 1.7, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_rlb_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.2, 1.8, time_periods),
        "DIGITAL_RELIABILITY": np.linspace(1.1, 1.5, time_periods),
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.25, 1.9, time_periods),
        "POLLUTION_PRODUCED": np.linspace(0.6, 0.3, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_inf_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.2, 1.8, time_periods),
        "CONSTRUCTION_BARRIERS": np.linspace(0.6, 0.4, time_periods),
        "INTERMODAL_CONNECTIONS": np.linspace(1.4, 1.7, time_periods),
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.2, 1.58, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_scl_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.2, 1.8, time_periods),
        "PLANNED_VOLUME": np.linspace(1.2, 1.8, time_periods),
        "ADJUSTMENT_COEF_1": np.linspace(1.2, 1.7, time_periods),
        "ADJUSTMENT_COEF_2": np.linspace(1.2, 1.6, time_periods),
        "ADJUSTMENT_COEF_3": np.linspace(1.2, 1.9, time_periods)
    }
    return pd.DataFrame(data)

##########################################
# SUSTAINABLE GROWTH SCENARIO
##########################################

def generate_cr_env_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ENERGY_CONSUMED": np.linspace(100, 110, time_periods),
        "DISTANCE": np.linspace(1000, 1050, time_periods),
        "LOAD_WEIGHT": np.linspace(8500, 8950, time_periods),
        "CO2_EMISSIONS": np.linspace(500, 300, time_periods),
        "MATERIAL_SUSTAINABILITY": np.linspace(0.81, 0.86, time_periods),
        "ENV_IMPACT_SCORE": np.linspace(0.83, 0.95, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sac_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "POSITIVE_FEEDBACK": np.linspace(100, 115, time_periods),
        "NEGATIVE_FEEDBACK": np.linspace(100, 80, time_periods),
    }
    return pd.DataFrame(data)

def generate_cr_tfe_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "CURRENT_TRL": np.linspace(8, 8.8, time_periods),
        "TARGET_TRL": np.linspace(8.5, 9, time_periods),
        "ENG_CHALLENGES_RESOLVED": np.linspace(9, 11, time_periods),
        "TARGET_ENG_CHALLENGES": np.linspace(4, 3, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_sfy_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RISK_SCORE": np.linspace(0.55, 0.65, time_periods),
        "MIN_RISK_SCORE": np.full(time_periods, 0.3),
        "MAX_RISK_SCORE": np.full(time_periods, 0.9)
    }
    return pd.DataFrame(data)

def generate_cr_reg_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.linspace(0.65, 0.7, time_periods),
        "INT_LAW_COMPLIANCE": np.linspace(0.55, 0.6, time_periods),  
        "LAND_USAGE_COMPLIANCE": np.linspace(0.7, 0.74, time_periods),
        "LEGAL_COMPLIANCE": np.linspace(0.6, 0.65, time_periods),
        "TRL_COMPLIANCE": np.linspace(0.62, 0.66, time_periods)
    }
    return pd.DataFrame(data)

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

def generate_cr_ecv_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "REVENUE": np.linspace(400000, 700000, time_periods),
        "OPEX": np.linspace(300000, 200000, time_periods),
        "CAPEX": np.full(time_periods, 1000000),
        "DISCOUNT_RATE": np.linspace(0.06, 0.045, time_periods),
        "PROJECT_LIFETIME": np.full(time_periods, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_usb_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.4, 1.6, time_periods),
        "USER_EXP_RATIO": np.linspace(1.0, 1.3, time_periods),
        "ACCESSIBILITY_AGEING": np.linspace(0.8, 1.0, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_rlb_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.3, 1.65, time_periods),
        "DIGITAL_RELIABILITY": np.linspace(1.2, 1.5, time_periods),
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.3, 1.5, time_periods),
        "POLLUTION_PRODUCED": np.linspace(0.55, 0.5, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_inf_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.3, 1.4, time_periods),
        "CONSTRUCTION_BARRIERS": np.linspace(0.55, 0.45, time_periods),
        "INTERMODAL_CONNECTIONS": np.linspace(1.3, 1.6, time_periods),
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.3, 1.8, time_periods)
    }
    return pd.DataFrame(data)

def generate_cr_scl_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.3, 1.4, time_periods),
        "PLANNED_VOLUME": np.linspace(1.4, 1.55, time_periods),
        "ADJUSTMENT_COEF_1": np.linspace(1.3, 1.4, time_periods),
        "ADJUSTMENT_COEF_2": np.linspace(1.1, 1.5, time_periods),
        "ADJUSTMENT_COEF_3": np.linspace(1.2, 1.3, time_periods)
    }
    return pd.DataFrame(data)
