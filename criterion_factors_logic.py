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
    time = np.arange(0, time_periods)
    carbon_footprint = np.random.uniform(0, 1, time_periods)  # Example logic for carbon footprint
    air_quality_impact = np.random.uniform(0, 1, time_periods)  # Example logic for air quality impact
    water_consumption = np.random.uniform(0, 1, time_periods)  # Example logic for water consumption
    biodiversity_loss = np.random.uniform(0, 1, time_periods)  # Example logic for biodiversity loss
    
    environmental_impact_score = 1 - np.mean([carbon_footprint, air_quality_impact, water_consumption, biodiversity_loss], axis=0)

    data = {
        "TIME": time,
        "CARBON_FOOTPRINT": carbon_footprint,
        "AIR_QUALITY_IMPACT": air_quality_impact,
        "WATER_CONSUMPTION": water_consumption,
        "BIODIVERSITY_LOSS": biodiversity_loss,
        "ENVIRONMENTAL_IMPACT_SCORE": environmental_impact_score
    }
    return pd.DataFrame(data)
