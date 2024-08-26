import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def generate_safety_data(time_periods=100):
    np.random.seed(42)  # Seed for reproducibility
    
    # Simulate time periods
    time = np.arange(0, time_periods)

    # Generate risk scores between 0 and 1
    risk_scores = np.random.rand(time_periods, 5)  # Assuming 5 risk components

    # Compute the min and max risk scores across all components
    min_risks = risk_scores.min(axis=1)
    max_risks = risk_scores.max(axis=1)

    # Create DataFrame according to the new schema without CR_SFY
    df = pd.DataFrame({
        "TIME": time,
        "RISK_SCORE": risk_scores.mean(axis=1),  # Average risk score for each time period
        "MIN_RISK_SCORE": min_risks,
        "MAX_RISK_SCORE": max_risks
    })

    return df

def calculate_cr_sfy():
    session = Session.builder.configs(get_snowflake_connection_params()).create()

    # Load data from CR_SFY_SOURCE table
    df = session.table("CR_SFY_SOURCE").to_pandas()

    # Calculate CR_SFY for each time period
    epsilon = 1e-6  # Avoid division by zero
    cr_sfy = np.array([1 / np.sum((df.iloc[t]["RISK_SCORE"] - df.iloc[t]["MIN_RISK_SCORE"]) / 
                                  (df.iloc[t]["MAX_RISK_SCORE"] - df.iloc[t]["MIN_RISK_SCORE"] + epsilon))
                       for t in range(len(df))])

    # Create DataFrame with TIME and CR_SFY
    df_cr_sfy = pd.DataFrame({
        "TIME": df["TIME"],
        "CR_SFY": cr_sfy
    })

    # Save the calculated CR_SFY to CALC_CR_SF table in Snowflake
    session.write_pandas(df_cr_sfy, "CALC_CR_SF", mode="overwrite")
    session.close()

    return df_cr_sfy

def generate_environmental_impact_data(time_periods=100):
    np.random.seed(42)
    time = np.arange(0, time_periods)
    energy_consumed = np.random.rand(time_periods)
    distance = np.random.rand(time_periods)
    load_weight = np.random.rand(time_periods)
    co2_emissions = np.random.rand(time_periods)
    material_sustainability = np.random.rand(time_periods)
    env_impact_score = np.random.rand(time_periods)

    # Compute Environmental Impact criterion based on formula
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    criterion_values = (
        w1 * energy_consumed / (distance * load_weight) +
        w2 * co2_emissions / (distance * load_weight) +
        w3 * material_sustainability +
        w4 * env_impact_score
    )

    df = pd.DataFrame({
        "TIME": time,
        "ENERGY_CONSUMED": energy_consumed,
        "DISTANCE": distance,
        "LOAD_WEIGHT": load_weight,
        "CO2_EMISSIONS": co2_emissions,
        "MATERIAL_SUSTAINABILITY": material_sustainability,
        "ENV_IMPACT_SCORE": env_impact_score,
        "CR_ENV": criterion_values
    })

    return df
