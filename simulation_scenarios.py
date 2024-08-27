import pandas as pd
import numpy as np

def generate_cr_env_data(time_periods=100):
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

def generate_cr_sac_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "A": np.linspace(1000, 200, time_periods),  # Positive feedback decreases sharply
        "B": np.linspace(200, 800, time_periods),   # Negative feedback increases
    }
    return pd.DataFrame(data)

def generate_cr_tfe_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "T": np.linspace(9, 5, time_periods),  # TRL decreases
        "P": np.linspace(9, 5, time_periods),  # Target TRL decreases
        "L": np.linspace(10, 3, time_periods), # Engineering challenges resolved decreases
        "C": np.linspace(5, 15, time_periods)  # Challenges increase
    }
    return pd.DataFrame(data)

def generate_cr_sfy_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "R": np.linspace(0.9, 0.2, time_periods),  # Risk score decreases
        "MinR": np.full(time_periods, 0.2),        # Minimum risk remains low
        "MaxR": np.full(time_periods, 0.9)         # Maximum risk remains high
    }
    return pd.DataFrame(data)

def generate_cr_reg_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "EC": np.linspace(0.9, 0.3, time_periods),  # Ethical compliance decreases
        "LC": np.linspace(0.9, 0.3, time_periods),  # Legal norms compliance decreases
        "LU": np.linspace(0.9, 0.3, time_periods),  # Land usage compliance decreases
        "IL": np.linspace(0.9, 0.3, time_periods),  # International laws compliance decreases
    }
    return pd.DataFrame(data)

def generate_cr_qmf_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "S": np.full(time_periods, 0),            # Disruptive technology use remains low
        "D": np.linspace(10, 3, time_periods)     # Number of disruptive technologies decreases
    }
    return pd.DataFrame(data)

def generate_cr_ecv_data(time_periods=100):
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

def generate_cr_usb_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "PRODUCTION_OUTPUT": np.linspace(1.5, 0.5, time_periods),  # Production output decreases
        "USER_EXP_RATIO": np.linspace(1.5, 0.5, time_periods),      # User experience ratio decreases
        "ACCESSIBILITY_AGEING": np.linspace(1.5, 0.5, time_periods) # Accessibility by aging population decreases
    }
    return pd.DataFrame(data)

def generate_cr_rlb_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "DURABILITY": np.linspace(1.5, 0.5, time_periods),            # Durability decreases
        "DIGITAL_RELIABILITY": np.linspace(1.5, 0.5, time_periods),   # Digital reliability decreases
        "WEATHER_DISASTER_RESILIENCE": np.linspace(1.5, 0.5, time_periods), # Weather and disaster resilience decreases
        "POLLUTION_PRODUCED": np.linspace(0.5, 1.5, time_periods)     # Pollution produced increases
    }
    return pd.DataFrame(data)

def generate_cr_inf_data(time_periods=100):
    time = np.arange(time_periods)
    data = {
        "TIME": time,
        "COMMON_INFRA_FEATURES": np.linspace(1.5, 0.5, time_periods),  # Common infrastructure features decrease
        "CONSTRUCTION_BARRIERS": np.linspace(0.5, 1.5, time_periods),  # Barriers to construction increase
        "INTERMODAL_CONNECTIONS": np.linspace(1.5, 0.5, time_periods), # Intermodal connections decrease
        "INFRA_ADAPTABILITY_FEATURES": np.linspace(1.5, 0.5, time_periods) # Adaptability decreases
    }
    return pd.DataFrame(data)

def generate_cr_scl_data(time_periods=100):
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
