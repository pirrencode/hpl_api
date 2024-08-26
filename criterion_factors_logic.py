import numpy as np
import pandas as pd

def generate_safety_data(time_steps=100):
    """
    Generates Safety criterion data.
    
    Args:
    - time_steps (int): The number of time steps to generate data for.
    
    Returns:
    - DataFrame: A Pandas DataFrame containing the generated data.
    """
    risk_scores = np.random.rand(time_steps, 5)  # Simulate risk scores for 5 components
    min_risks = risk_scores.min(axis=0)
    max_risks = risk_scores.max(axis=0)

    # Calculate Safety Criterion
    safety_criterion = np.zeros(time_steps)
    for t in range(time_steps):
        safety_criterion[t] = 1 / np.sum((risk_scores[t, :] - min_risks) / (max_risks - min_risks))

    # Create a DataFrame
    df = pd.DataFrame({
        "TIME": np.arange(time_steps),
        "RISK_SCORE_COMPONENT_1": risk_scores[:, 0],
        "RISK_SCORE_COMPONENT_2": risk_scores[:, 1],
        "RISK_SCORE_COMPONENT_3": risk_scores[:, 2],
        "RISK_SCORE_COMPONENT_4": risk_scores[:, 3],
        "RISK_SCORE_COMPONENT_5": risk_scores[:, 4],
        "SAFETY_CRITERION": safety_criterion
    })
    
    return df

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
