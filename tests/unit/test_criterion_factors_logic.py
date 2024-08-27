import pytest
import pandas as pd
import numpy as np
from criterion_factors_logic import generate_safety_data, generate_environmental_impact_data

def test_generate_safety_data():
    df = generate_safety_data(time_periods=100)
    
    assert isinstance(df, pd.DataFrame)
    
    expected_columns = ["TIME", "RISK_SCORE", "MIN_RISK_SCORE", "MAX_RISK_SCORE"]
    assert list(df.columns) == expected_columns
    
    assert len(df) == 100
    
    assert (df["TIME"] == np.arange(100)).all()
    
    assert df["RISK_SCORE"].between(0, 1).all()
    assert df["MIN_RISK_SCORE"].between(0, 1).all()
    assert df["MAX_RISK_SCORE"].between(0, 1).all()

def test_generate_environmental_impact_data():
    df = generate_environmental_impact_data(time_periods=100)
    
    assert isinstance(df, pd.DataFrame)
    
    expected_columns = [
        "TIME", "ENERGY_CONSUMED", "DISTANCE", "LOAD_WEIGHT", 
        "CO2_EMISSIONS", "MATERIAL_SUSTAINABILITY", 
        "ENV_IMPACT_SCORE", "CR_ENV"
    ]
    assert list(df.columns) == expected_columns
    
    assert len(df) == 100
    
    assert (df["TIME"] == np.arange(100)).all()
    
    for column in expected_columns[1:]:
        assert df[column].between(0, 1).all()
    
    assert df["CR_ENV"].notnull().all()