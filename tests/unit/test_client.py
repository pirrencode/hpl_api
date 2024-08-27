import os
import unittest
from unittest.mock import patch
from app import get_snowflake_connection_params

def get_snowflake_connection_params():
    return {
        'user': os.getenv('SNOWFLAKE_USER'),
        'password': os.getenv('SNOWFLAKE_PASSWORD'),
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    }

class TestGetSnowflakeConnectionParams(unittest.TestCase):

    @patch('app.os.getenv')
    def test_get_snowflake_connection_params(self, mock_getenv):
        # Define the mock return values for environment variables
        mock_getenv.side_effect = lambda key: {
            'SNOWFLAKE_USER': 'test_user',
            'SNOWFLAKE_PASSWORD': 'test_password',
            'SNOWFLAKE_ACCOUNT': 'test_account',
            'SNOWFLAKE_WAREHOUSE': 'test_warehouse',
            'SNOWFLAKE_DATABASE': 'test_database',
            'SNOWFLAKE_SCHEMA': 'test_schema',
        }.get(key)

        # Call the function
        params = get_snowflake_connection_params()

        # Assert the returned dictionary matches the expected values
        expected_params = {
            'user': 'test_user',
            'password': 'test_password',
            'account': 'test_account',
            'warehouse': 'test_warehouse',
            'database': 'test_database',
            'schema': 'test_schema',
        }
        self.assertEqual(params, expected_params)

from app import calculate_cr_env

class TestCalculateCrEnv(unittest.TestCase):

    def test_calculate_cr_env(self):
        # Test case with balanced inputs
        result = calculate_cr_env(energy_consumed=0.5, co2_emissions=0.5, material_sustainability=0.5)
        expected_result = (0.5 * 0.5) + (0.5 * 0.5) + 0.5
        self.assertEqual(result, expected_result)

        # Test case with zero impact factors
        result = calculate_cr_env(energy_consumed=0, co2_emissions=0, material_sustainability=0)
        expected_result = 0
        self.assertEqual(result, expected_result)

        # Test case with maximum inputs
        result = calculate_cr_env(energy_consumed=1, co2_emissions=1, material_sustainability=1)
        expected_result = (1 * 0.5) + (1 * 0.5) + 1
        self.assertEqual(result, expected_result)

        # Test case with different impact_factor
        result = calculate_cr_env(energy_consumed=0.7, co2_emissions=0.3, material_sustainability=0.9, impact_factor=0.7)
        expected_result = (0.7 * 0.7) + (0.3 * 0.7) + 0.9
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()    