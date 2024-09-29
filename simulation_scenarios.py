import pandas as pd
import numpy as np

####################################
# RAPID DECLINE SCENARIO
####################################

def generate_cr_env_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "E": np.linspace(1000, 200, time_periods),  # Energy consumed decreases sharply
        "D": np.linspace(1000, 200, time_periods),  # Distance decreases sharply
        "L": np.linspace(1000, 200, time_periods),  # Load decreases sharply
        "C": np.linspace(500, 100, time_periods),   # CO2 emissions decrease sharply
        "M": np.linspace(0.8, 0.2, time_periods),   # Material sustainability decreases
        "Y": np.linspace(0.8, 0.2, time_periods)    # Environmental impact worsens
    }
    return pd.DataFrame(data)

def generate_cr_sac_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "A": np.linspace(1000, 200, time_periods),  # Positive feedback decreases sharply
        "B": np.linspace(200, 800, time_periods),   # Negative feedback increases
    }
    return pd.DataFrame(data)

def generate_cr_tfe_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "T": np.linspace(9, 5, time_periods),  # TRL decreases
        "P": np.linspace(9, 5, time_periods),  # Target TRL decreases
        "L": np.linspace(10, 3, time_periods), # Engineering challenges resolved decreases
        "C": np.linspace(5, 15, time_periods)  # Challenges increase
    }
    return pd.DataFrame(data)

def generate_cr_sfy_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "R": np.linspace(0.9, 0.2, time_periods),  # Risk score decreases
        "MinR": np.full(time_periods, 0.2),        # Minimum risk remains low
        "MaxR": np.full(time_periods, 0.9)         # Maximum risk remains high
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
        "TRL_COMPLIANCE": np.linspace(0.5, 0.5, time_periods)
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
        "REVENUE": np.linspace(500000, 100000, time_periods),   # Revenue decreases sharply
        "OPEX": np.linspace(200000, 500000, time_periods),      # Operational expenditure increases
        "CAPEX": np.full(time_periods, 1000000),                # Capital expenditure remains constant
        "DISCOUNT_RATE": np.linspace(0.05, 0.1, time_periods),  # Discount rate increases
        "PROJECT_LIFETIME": np.full(time_periods, 25)           # Project lifetime remains constant
    }
    return pd.DataFrame(data)

def generate_cr_usb_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.5, 0.5, time_periods),  # Production output decreases
        "USER_EXP_RATIO": np.linspace(1.5, 0.5, time_periods),      # User experience ratio decreases
        "ACCESSIBILITY_AGEING": np.linspace(1.5, 0.5, time_periods) # Accessibility by aging population decreases
    }
    return pd.DataFrame(data)

def generate_cr_rlb_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.5, 0.5, time_periods),            # Durability decreases
        "DIGITAL_RELIABILITY": np.linspace(1.5, 0.5, time_periods),   # Digital reliability decreases
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.5, 0.5, time_periods), # Weather and disaster resilience decreases
        "POLLUTION_PRODUCED": np.linspace(0.5, 1.5, time_periods)     # Pollution produced increases
    }
    return pd.DataFrame(data)

def generate_cr_inf_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.5, 0.5, time_periods),  # Common infrastructure features decrease
        "CONSTRUCTION_BARRIERS": np.linspace(0.5, 1.5, time_periods),  # Barriers to construction increase
        "INTERMODAL_CONNECTIONS": np.linspace(1.5, 0.5, time_periods), # Intermodal connections decrease
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.5, 0.5, time_periods) # Adaptability decreases
    }
    return pd.DataFrame(data)

def generate_cr_scl_data_rapid_decline(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.5, 0.5, time_periods),    # Resource mileage decreases
        "PLANNED_VOLUME": np.linspace(1.5, 0.5, time_periods),      # Planned volume decreases
        "ADJUSTMENT_COEF_1": np.linspace(1.5, 0.5, time_periods),   # Adjustment coefficient decreases
        "ADJUSTMENT_COEF_2": np.linspace(1.5, 0.5, time_periods),   # Adjustment coefficient decreases
        "ADJUSTMENT_COEF_3": np.linspace(1.5, 0.5, time_periods)    # Adjustment coefficient decreases
    }
    return pd.DataFrame(data)

####################################
# DECLINE OVER TIME SCENARIO
####################################

def generate_cr_env_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "E": np.linspace(1000, 800, time_periods),  # Energy consumed decreases gradually
        "D": np.linspace(1000, 800, time_periods),  # Distance decreases gradually
        "L": np.linspace(1000, 800, time_periods),  # Load decreases gradually
        "C": np.linspace(500, 400, time_periods),   # CO2 emissions decrease gradually
        "M": np.linspace(0.8, 0.7, time_periods),   # Material sustainability slightly decreases
        "Y": np.linspace(0.8, 0.7, time_periods)    # Environmental impact slightly worsens
    }
    return pd.DataFrame(data)

def generate_cr_sac_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "A": np.linspace(1000, 800, time_periods),  # Positive feedback decreases gradually
        "B": np.linspace(200, 300, time_periods),   # Negative feedback increases slightly
    }
    return pd.DataFrame(data)

def generate_cr_tfe_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "T": np.linspace(9, 8, time_periods),  # TRL decreases slightly
        "P": np.linspace(9, 8, time_periods),  # Target TRL decreases slightly
        "L": np.linspace(10, 8, time_periods), # Engineering challenges resolved decreases
        "C": np.linspace(5, 7, time_periods)   # Challenges increase slightly
    }
    return pd.DataFrame(data)

def generate_cr_sfy_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "R": np.linspace(0.9, 0.7, time_periods),  # Risk score decreases slightly
        "MinR": np.full(time_periods, 0.2),        # Minimum risk remains low
        "MaxR": np.full(time_periods, 0.9)         # Maximum risk remains high
    }
    return pd.DataFrame(data)

def generate_cr_reg_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.linspace(0.9, 0.8, time_periods),
        "INT_LAW_COMPLIANCE": np.linspace(0.9, 0.8, time_periods),  
        "LAND_USAGE_COMPLIANCE": np.linspace(0.9, 0.8, time_periods),
        "LEGAL_COMPLIANCE": np.linspace(0.9, 0.8, time_periods),
        "TRL_COMPLIANCE": np.linspace(0.5, 0.5, time_periods)
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
        "REVENUE": np.linspace(500000, 350000, time_periods),
        "OPEX": np.linspace(200000, 350000, time_periods),
        "CAPEX": np.full(time_periods, 1000000),
        "DISCOUNT_RATE": np.linspace(0.04, 0.055, time_periods),
        "PROJECT_LIFETIME": np.full(time_periods, 25)
    }
    return pd.DataFrame(data)

def generate_cr_usb_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.5, 1.2, time_periods),  # Production output decreases slightly
        "USER_EXP_RATIO": np.linspace(1.5, 1.2, time_periods),      # User experience ratio decreases slightly
        "ACCESSIBILITY_AGEING": np.linspace(1.5, 1.2, time_periods) # Accessibility by aging population decreases slightly
    }
    return pd.DataFrame(data)

def generate_cr_rlb_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.5, 1.2, time_periods),            # Durability decreases gradually
        "DIGITAL_RELIABILITY": np.linspace(1.5, 1.2, time_periods),   # Digital reliability decreases gradually
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.5, 1.2, time_periods), # Weather and disaster resilience decreases gradually
        "POLLUTION_PRODUCED": np.linspace(0.5, 0.6, time_periods)     # Pollution produced increases slightly
    }
    return pd.DataFrame(data)

def generate_cr_inf_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.5, 1.2, time_periods),  # Common infrastructure features decrease slightly
        "CONSTRUCTION_BARRIERS": np.linspace(0.5, 0.6, time_periods),  # Barriers to construction increase slightly
        "INTERMODAL_CONNECTIONS": np.linspace(1.5, 1.2, time_periods), # Intermodal connections decrease slightly
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.5, 1.2, time_periods) # Adaptability decreases slightly
    }
    return pd.DataFrame(data)

def generate_cr_scl_decline_over_time_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.5, 1.2, time_periods),    # Resource mileage decreases slightly
        "PLANNED_VOLUME": np.linspace(1.5, 1.2, time_periods),      # Planned volume decreases slightly
        "ADJUSTMENT_COEF_1": np.linspace(1.5, 1.2, time_periods),   # Adjustment coefficient decreases slightly
        "ADJUSTMENT_COEF_2": np.linspace(1.5, 1.2, time_periods),   # Adjustment coefficient decreases slightly
        "ADJUSTMENT_COEF_3": np.linspace(1.5, 1.2, time_periods)    # Adjustment coefficient decreases slightly
    }
    return pd.DataFrame(data)

##############################################
# RAPID GROWTH SCENARIO
##############################################

def generate_cr_env_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "E": np.linspace(1000, 1500, time_periods),  # Energy consumed increases (more efficient energy use)
        "D": np.linspace(1000, 1500, time_periods),  # Distance increases
        "L": np.linspace(1000, 1500, time_periods),  # Load increases
        "C": np.linspace(500, 200, time_periods),    # CO2 emissions decrease rapidly
        "M": np.linspace(0.8, 0.95, time_periods),   # Material sustainability increases
        "Y": np.linspace(0.8, 0.95, time_periods)    # Environmental impact improves
    }
    return pd.DataFrame(data)

def generate_cr_sac_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "A": np.linspace(1000, 1500, time_periods),  # Positive feedback increases rapidly
        "B": np.linspace(200, 100, time_periods),    # Negative feedback decreases
    }
    return pd.DataFrame(data)

def generate_cr_tfe_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "T": np.linspace(7, 10, time_periods),  # TRL increases rapidly
        "P": np.linspace(7, 10, time_periods),  # Target TRL increases rapidly
        "L": np.linspace(8, 12, time_periods),  # Engineering challenges resolved increases
        "C": np.linspace(5, 2, time_periods)    # Challenges decrease rapidly
    }
    return pd.DataFrame(data)

def generate_cr_sfy_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "R": np.linspace(0.7, 0.95, time_periods),  # Risk score decreases rapidly (improved safety)
        "MinR": np.full(time_periods, 0.2),         # Minimum risk remains low
        "MaxR": np.full(time_periods, 0.9)          # Maximum risk remains high
    }
    return pd.DataFrame(data)

def generate_cr_reg_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.linspace(0.7, 0.95, time_periods),
        "INT_LAW_COMPLIANCE": np.linspace(0.7, 0.95, time_periods),  
        "LAND_USAGE_COMPLIANCE": np.linspace(0.7, 0.95, time_periods),
        "LEGAL_COMPLIANCE": np.linspace(0.7, 0.95, time_periods),
        "TRL_COMPLIANCE": np.linspace(0.7, 0.95, time_periods)
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
        "REVENUE": np.linspace(500000, 1000000, time_periods),   # Revenue increases rapidly
        "OPEX": np.linspace(300000, 200000, time_periods),       # Operational expenditure decreases rapidly
        "CAPEX": np.full(time_periods, 1000000),                 # Capital expenditure remains constant
        "DISCOUNT_RATE": np.linspace(0.06, 0.05, time_periods),  # Discount rate decreases slightly
        "PROJECT_LIFETIME": np.full(time_periods, 25)            # Project lifetime remains constant
    }
    return pd.DataFrame(data)

def generate_cr_usb_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.2, 1.8, time_periods),  # Production output increases rapidly
        "USER_EXP_RATIO": np.linspace(1.2, 1.8, time_periods),      # User experience ratio increases rapidly
        "ACCESSIBILITY_AGEING": np.linspace(1.2, 1.8, time_periods) # Accessibility by aging population increases rapidly
    }
    return pd.DataFrame(data)

def generate_cr_rlb_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.2, 1.8, time_periods),             # Durability increases rapidly
        "DIGITAL_RELIABILITY": np.linspace(1.2, 1.8, time_periods),    # Digital reliability increases rapidly
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.2, 1.8, time_periods), # Weather and disaster resilience increases rapidly
        "POLLUTION_PRODUCED": np.linspace(0.6, 0.4, time_periods)      # Pollution produced decreases rapidly
    }
    return pd.DataFrame(data)

def generate_cr_inf_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.2, 1.8, time_periods),  # Common infrastructure features increase rapidly
        "CONSTRUCTION_BARRIERS": np.linspace(0.6, 0.4, time_periods),  # Barriers to construction decrease rapidly
        "INTERMODAL_CONNECTIONS": np.linspace(1.2, 1.8, time_periods), # Intermodal connections increase rapidly
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.2, 1.8, time_periods) # Adaptability increases rapidly
    }
    return pd.DataFrame(data)

def generate_cr_scl_rapid_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.2, 1.8, time_periods),    # Resource mileage increases rapidly
        "PLANNED_VOLUME": np.linspace(1.2, 1.8, time_periods),      # Planned volume increases rapidly
        "ADJUSTMENT_COEF_1": np.linspace(1.2, 1.8, time_periods),   # Adjustment coefficient increases rapidly
        "ADJUSTMENT_COEF_2": np.linspace(1.2, 1.8, time_periods),   # Adjustment coefficient increases rapidly
        "ADJUSTMENT_COEF_3": np.linspace(1.2, 1.8, time_periods)    # Adjustment coefficient increases rapidly
    }
    return pd.DataFrame(data)

##########################################
# SUSTAINABLE GROWTH SCENARIO
##########################################

def generate_cr_env_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "E": np.linspace(1000, 1200, time_periods),  # Energy consumed increases slightly (sustainable energy use)
        "D": np.linspace(1000, 1200, time_periods),  # Distance increases steadily
        "L": np.linspace(1000, 1200, time_periods),  # Load increases steadily
        "C": np.linspace(500, 300, time_periods),    # CO2 emissions decrease steadily
        "M": np.linspace(0.8, 0.9, time_periods),    # Material sustainability improves steadily
        "Y": np.linspace(0.8, 0.9, time_periods)     # Environmental impact improves steadily
    }
    return pd.DataFrame(data)

def generate_cr_sac_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "A": np.linspace(1000, 1200, time_periods),  # Positive feedback increases steadily
        "B": np.linspace(200, 150, time_periods),    # Negative feedback decreases steadily
    }
    return pd.DataFrame(data)

def generate_cr_tfe_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "T": np.linspace(8, 9, time_periods),  # TRL increases steadily
        "P": np.linspace(8, 9, time_periods),  # Target TRL increases steadily
        "L": np.linspace(9, 11, time_periods), # Engineering challenges resolved improves steadily
        "C": np.linspace(4, 3, time_periods)   # Challenges decrease steadily
    }
    return pd.DataFrame(data)

def generate_cr_sfy_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "R": np.linspace(0.75, 0.85, time_periods),  # Risk score decreases steadily (improved safety)
        "MinR": np.full(time_periods, 0.2),          # Minimum risk remains low
        "MaxR": np.full(time_periods, 0.9)           # Maximum risk remains high
    }
    return pd.DataFrame(data)

def generate_cr_reg_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "ETHICAL_COMPLIANCE": np.linspace(0.6, 0.7, time_periods),
        "INT_LAW_COMPLIANCE": np.linspace(0.6, 0.7, time_periods),  
        "LAND_USAGE_COMPLIANCE": np.linspace(0.6, 0.7, time_periods),
        "LEGAL_COMPLIANCE": np.linspace(0.6, 0.7, time_periods),
        "TRL_COMPLIANCE": np.linspace(0.6, 0.7, time_periods)
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
        "REVENUE": np.linspace(500000, 700000, time_periods),   # Revenue increases steadily
        "OPEX": np.linspace(300000, 250000, time_periods),      # Operational expenditure decreases steadily
        "CAPEX": np.full(time_periods, 1000000),                # Capital expenditure remains constant
        "DISCOUNT_RATE": np.linspace(0.06, 0.055, time_periods), # Discount rate decreases slightly
        "PROJECT_LIFETIME": np.full(time_periods, 25)           # Project lifetime remains constant
    }
    return pd.DataFrame(data)

def generate_cr_usb_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.3, 1.6, time_periods),  # Production output increases steadily
        "USER_EXP_RATIO": np.linspace(1.3, 1.6, time_periods),      # User experience ratio increases steadily
        "ACCESSIBILITY_AGEING": np.linspace(1.3, 1.6, time_periods) # Accessibility by aging population increases steadily
    }
    return pd.DataFrame(data)

def generate_cr_rlb_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.3, 1.6, time_periods),             # Durability increases steadily
        "DIGITAL_RELIABILITY": np.linspace(1.3, 1.6, time_periods),    # Digital reliability increases steadily
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.3, 1.6, time_periods), # Weather and disaster resilience increases steadily
        "POLLUTION_PRODUCED": np.linspace(0.55, 0.45, time_periods)    # Pollution produced decreases steadily
    }
    return pd.DataFrame(data)

def generate_cr_inf_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.3, 1.6, time_periods),  # Common infrastructure features increase steadily
        "CONSTRUCTION_BARRIERS": np.linspace(0.55, 0.45, time_periods),# Barriers to construction decrease steadily
        "INTERMODAL_CONNECTIONS": np.linspace(1.3, 1.6, time_periods), # Intermodal connections increase steadily
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.3, 1.6, time_periods) # Adaptability increases steadily
    }
    return pd.DataFrame(data)

def generate_cr_scl_sustainable_growth_data(time_periods):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "RESOURCE_MILEAGE": np.linspace(1.3, 1.6, time_periods),    # Resource mileage increases steadily
        "PLANNED_VOLUME": np.linspace(1.3, 1.6, time_periods),      # Planned volume increases steadily
        "ADJUSTMENT_COEF_1": np.linspace(1.3, 1.6, time_periods),   # Adjustment coefficient increases steadily
        "ADJUSTMENT_COEF_2": np.linspace(1.3, 1.6, time_periods),   # Adjustment coefficient increases steadily
        "ADJUSTMENT_COEF_3": np.linspace(1.3, 1.6, time_periods)    # Adjustment coefficient increases steadily
    }
    return pd.DataFrame(data)
