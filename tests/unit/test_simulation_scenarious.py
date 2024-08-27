import pytest
from simulation_scenarios import run_simulation

def test_run_simulation():
    result = run_simulation(num_iterations=10, scenario="base_case")
    
    assert result is not None
    
    assert isinstance(result, dict)
    
    expected_keys = ["total_cost", "risk_score", "duration"]

    assert all(key in result for key in expected_keys)
    assert result["total_cost"] >= 0
    assert 0 <= result["risk_score"] <= 1
    assert result["duration"] > 0

from simulation_scenarios import generate_scenario

def test_generate_scenario():
    scenario = generate_scenario(type="optimistic")
    
    assert scenario is not None 
    assert isinstance(scenario, dict)
    expected_keys = ["parameters", "assumptions"]
    assert all(key in scenario for key in expected_keys)
    assert scenario["parameters"]["growth_rate"] > 0
    assert scenario["assumptions"]["market_stability"] in [True, False]

from simulation_scenarios import calculate_outcome

def test_calculate_outcome():

    scenario = {"parameters": {"growth_rate": 0.05}, "assumptions": {"market_stability": True}}
    outcome = calculate_outcome(scenario)
    
    assert outcome is not None
    assert isinstance(outcome, dict)
    assert "profit" in outcome
    assert outcome["profit"] > 0
    assert "risk" in outcome
    assert 0 <= outcome["risk"] <= 1

from simulation_scenarios import evaluate_risks

def test_evaluate_risks():
    risks = evaluate_risks(factors={"market_volatility": 0.2, "competition": 0.5})

    assert risks is not None
    assert isinstance(risks, float)
    assert 0 <= risks <= 1
