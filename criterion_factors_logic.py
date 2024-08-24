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
