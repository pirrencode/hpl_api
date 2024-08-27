import numpy as np
import pandas as pd
import streamlit as st
import random

def generate_safety_data(time_periods=100):

    # Simulate time periods
    time = np.arange(0, time_periods)

    # Generate risk scores between 0 and 1
    risk_scores = np.random.rand(time_periods, 5)

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

    st.write("Generated dataset:")
    st.dataframe(df.head())

    return df

def generate_environmental_impact_data(time_periods=100):

    time = np.arange(0, time_periods)
    energy_consumed = np.random.rand(time_periods)
    distance = np.random.rand(time_periods)
    load_weight = np.random.rand(time_periods)
    co2_emissions = np.random.rand(time_periods)
    material_sustainability = np.random.rand(time_periods)
    env_impact_score = np.random.rand(time_periods)

    # Compute Environmental Impact criterion based on formula
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    # criterion_values = (
    #     w1 * energy_consumed / (distance * load_weight) +
    #     w2 * co2_emissions / (distance * load_weight) +
    #     w3 * material_sustainability +
    #     w4 * env_impact_score
    # )

    df = pd.DataFrame({
        "TIME": time,
        "ENERGY_CONSUMED": energy_consumed,
        "DISTANCE": distance,
        "LOAD_WEIGHT": load_weight,
        "CO2_EMISSIONS": co2_emissions,
        "MATERIAL_SUSTAINABILITY": material_sustainability,
        "ENV_IMPACT_SCORE": env_impact_score,
        # "CR_ENV": criterion_values
    })

    st.write("Generated dataset:")
    st.dataframe(df.head())

    return df

def generate_social_acceptance_data(time_periods=100):
    time = np.arange(time_periods)
    positive_feedback = np.random.rand(time_periods) * 100
    negative_feedback = np.random.rand(time_periods) * 50

    data = pd.DataFrame({
        'TIME': time,
        'POSITIVE_FEEDBACK': positive_feedback,
        'NEGATIVE_FEEDBACK': negative_feedback
    })

    return data

def generate_technical_feasibility_data(time_periods=100):
    time = np.arange(time_periods)
    current_trl = np.full(time_periods, 6)  # Stable CURRENT_TRL at 6
    target_trl = np.random.uniform(6, 11, time_periods)
    eng_challenges_resolved = np.random.rand(time_periods)  # In the range [0, 1]
    target_eng_challenges = np.random.rand(time_periods)  # In the range [0, 1]

    data = pd.DataFrame({
        'TIME': time,
        'CURRENT_TRL': current_trl,
        'TARGET_TRL': target_trl,
        'ENG_CHALLENGES_RESOLVED': eng_challenges_resolved,
        'TARGET_ENG_CHALLENGES': target_eng_challenges
    })

    return data

def generate_regulatory_approval_data(time_periods=100):
    time = np.arange(time_periods)
    ethical_compliance = np.random.rand(time_periods)
    legal_compliance = np.random.rand(time_periods)
    land_usage_compliance = np.random.rand(time_periods)
    int_law_compliance = np.random.rand(time_periods)
    trl_compliance = np.random.rand(time_periods)

    data = pd.DataFrame({
        'TIME': time,
        'ETHICAL_COMPLIANCE': ethical_compliance,
        'LEGAL_COMPLIANCE': legal_compliance,
        'LAND_USAGE_COMPLIANCE': land_usage_compliance,
        'INT_LAW_COMPLIANCE': int_law_compliance,
        'TRL_COMPLIANCE': trl_compliance
    })

    return data

def generate_quantum_factor_data(time_periods=100):
    time = np.arange(time_periods)

    # Generate random boolean values for each technology
    maglev_levitation = np.random.choice([True, False], time_periods)
    ambient_intelligence = np.random.choice([True, False], time_periods)
    generative_ai = np.random.choice([True, False], time_periods)
    ai_machine_learning = np.random.choice([True, False], time_periods)
    digital_twins = np.random.choice([True, False], time_periods)
    five_g = np.random.choice([True, False], time_periods)
    quantum_computing = np.random.choice([True, False], time_periods)
    augmented_reality = np.random.choice([True, False], time_periods)
    virtual_reality = np.random.choice([True, False], time_periods)
    printing_at_scale = np.random.choice([True, False], time_periods)
    blockchain = np.random.choice([True, False], time_periods)
    self_driving_autonomous_vehicles = np.random.choice([True, False], time_periods)

    # Calculate TOTAL_DISRUPTIVE_TECH based on the sum of TRUE values
    total_disruptive_tech = (maglev_levitation.astype(int) + ambient_intelligence.astype(int) +
                             generative_ai.astype(int) + ai_machine_learning.astype(int) +
                             digital_twins.astype(int) + five_g.astype(int) +
                             quantum_computing.astype(int) + augmented_reality.astype(int) +
                             virtual_reality.astype(int) + printing_at_scale.astype(int) +
                             blockchain.astype(int) + self_driving_autonomous_vehicles.astype(int))

    data = pd.DataFrame({
        'TIME': time,
        'MAGLEV_LEVITATION': maglev_levitation,
        'AMBIENT_INTELLIGENCE': ambient_intelligence,
        'GENERATIVE_AI': generative_ai,
        'AI_MACHINE_LEARNING': ai_machine_learning,
        'DIGITAL_TWINS': digital_twins,
        'FIVE_G': five_g,
        'QUANTUM_COMPUTING': quantum_computing,
        'AUGMENTED_REALITY': augmented_reality,
        'VIRTUAL_REALITY': virtual_reality,
        'PRINTING_AT_SCALE': printing_at_scale,
        'BLOCKCHAIN': blockchain,
        'SELF_DRIVING_AUTONOMOUS_VEHICLES': self_driving_autonomous_vehicles,
        'TOTAL_DISRUPTIVE_TECH': total_disruptive_tech
    })

    return data

def generate_economic_viability_data(time_periods=100):

    data = {
        "TIME": list(range(1, time_periods + 1)),
        "REVENUE": [round(random.uniform(100000, 500000), 2) for _ in range(time_periods)],
        "OPEX": [round(random.uniform(50000, 200000), 2) for _ in range(time_periods)],
        "CAPEX": [round(random.uniform(1000000, 5000000), 2) for _ in range(time_periods)],
        "DISCOUNT_RATE": [round(random.uniform(0.01, 0.1), 2) for _ in range(time_periods)],
        "PROJECT_LIFETIME": [random.randint(10, 30) for _ in range(time_periods)]
    }
    

    df = pd.DataFrame(data)
    
    return df

def generate_usability_data(time_periods=100):
    data = {
        "TIME": list(range(1, time_periods + 1)),
        "PRODUCTION_OUTPUT": [round(random.uniform(0.5, 1.5), 2) for _ in range(time_periods)],  # p: values between 0.5 and 1.5
        "USER_EXP_RATIO": [round(random.uniform(0.5, 1.5), 2) for _ in range(time_periods)],     # e: values between 0.5 and 1.5
        "ACCESSIBILITY_AGEING": [round(random.uniform(0.5, 1.5), 2) for _ in range(time_periods)] # a: values between 0.5 and 1.5
    }
    
    df = pd.DataFrame(data)
    
    return df