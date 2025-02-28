import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Add the workspace directory to the Python path
sys.path.append('/workspace')

from utils import (
    validate_prompt_format,
    find_prompt_usage_in_code,
    get_prompt_from_dynamodb,
    update_prompt
)

class TestPromptValidation(unittest.TestCase):

    @patch('utils.get_test_parameters')
    def test_validate_prompt_format_success(self, mock_get_test_parameters):
        # Setup
        mock_get_test_parameters.return_value = ['param1', 'param2', 'param3']

        # Test with valid prompt
        valid_prompt = "This is a valid prompt with {param1} and {param2}."
        is_valid, error = validate_prompt_format(valid_prompt)

        # Assert
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    @patch('utils.get_test_parameters')
    def test_validate_prompt_format_unknown_variable(self, mock_get_test_parameters):
        # Setup
        mock_get_test_parameters.return_value = ['param1', 'param2', 'param3']

        # Test with unknown variable
        invalid_prompt = "This prompt has an {unknown_param}."
        is_valid, error = validate_prompt_format(invalid_prompt)

        # Assert
        self.assertFalse(is_valid)
        self.assertIn("Unknown variables in prompt", error)
        self.assertIn("unknown_param", error)

    @patch('utils.get_test_parameters')
    @patch('utils.find_prompt_usage_in_code')
    def test_validate_prompt_format_missing_required_param(self, mock_find_usage, mock_get_test_parameters):
        # Setup
        mock_get_test_parameters.return_value = ['param1', 'param2', 'param3']
        mock_find_usage.return_value = ('test_prompt', ['param1', 'param2', 'param3'])

        # Test with missing required parameter
        incomplete_prompt = "This prompt only has {param1}."
        is_valid, error = validate_prompt_format(incomplete_prompt)

        # Assert
        self.assertFalse(is_valid)
        self.assertIn("Missing required parameters in prompt", error)
        self.assertTrue("param2" in error or "param3" in error)

    @patch('utils.get_test_parameters')
    def test_validate_prompt_format_invalid_format(self, mock_get_test_parameters):
        # Setup
        mock_get_test_parameters.return_value = ['param1', 'param2', 'param3']

        # Test with invalid format (unclosed bracket)
        invalid_format_prompt = "This prompt has an {unclosed bracket."
        is_valid, error = validate_prompt_format(invalid_format_prompt)

        # Assert
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)

    @patch('glob.glob')
    @patch('builtins.open')
    def test_find_prompt_usage_in_code(self, mock_open, mock_glob):
        # Setup
        mock_glob.return_value = ['/workspace/test_file.py']

        # Mock file content with prompt usage
        file_content = """
        def test_function():
            system_message = get_prompt_from_dynamodb('test_prompt', {
                'param1': 'value1',
                'param2': 'value2'
            })
        """

        # Setup mock file open
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = file_content
        mock_open.return_value = mock_file

        # Test
        result = find_prompt_usage_in_code('test_prompt')

        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'test_prompt')
        self.assertEqual(set(result[1]), {'param1', 'param2'})

    @patch('utils.get_dynamodb_table')
    def test_get_prompt_from_dynamodb_with_substitutions(self, mock_get_table):
        # Setup mock table and response
        mock_table = MagicMock()
        mock_table.query.return_value = {
            'Items': [{'content': 'Hello {name}, welcome to {service}!'}]
        }
        mock_get_table.return_value = mock_table

        # Test with valid substitutions
        result = get_prompt_from_dynamodb('test_prompt', {
            'name': 'John',
            'service': 'AWS'
        })

        # Assert
        self.assertEqual(result, 'Hello John, welcome to AWS!')

    @patch('utils.get_dynamodb_table')
    def test_get_prompt_from_dynamodb_missing_substitution(self, mock_get_table):
        # Setup mock table and response
        mock_table = MagicMock()
        mock_table.query.return_value = {
            'Items': [{'content': 'Hello {name}, welcome to {service}!'}]
        }
        mock_get_table.return_value = mock_table

        # Test with missing substitution
        with self.assertRaises(ValueError) as context:
            get_prompt_from_dynamodb('test_prompt', {'name': 'John'})

        # Assert
        self.assertIn("Missing substitution key", str(context.exception))
        self.assertIn("service", str(context.exception))

    @patch('utils.validate_prompt_format')
    @patch('utils.get_dynamodb_table')
    def test_update_prompt_validation_error(self, mock_get_table, mock_validate):
        # Setup
        mock_table = MagicMock()
        mock_table.query.return_value = {
            'Items': [{'version': 1, 'is_object': False, 'createdAt': '2023-01-01'}]
        }
        mock_get_table.return_value = mock_table

        # Mock validation failure
        mock_validate.return_value = (False, "Test validation error")

        # Test
        result = update_prompt('test_prompt', 'Invalid prompt content')

        # Assert
        self.assertFalse(result)
        mock_validate.assert_called_once()
        # Verify the table's put_item was not called
        mock_table.put_item.assert_not_called()

class TestLambdaPromptRetry(unittest.TestCase):
    @patch('utils.update_prompt')
    @patch('utils.validate_prompt_format')
    @patch('lambda_function.run_completion_with_fallback')
    def test_lambda_prompt_retry(self, mock_run_completion, mock_validate, mock_update_prompt):
        # This would be a more complex test for the lambda_function retry logic
        # but we'll leave it as a placeholder for now
        pass

if __name__ == '__main__':
    unittest.main()
