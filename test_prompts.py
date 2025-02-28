import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json
from utils import update_prompt, get_prompt_from_dynamodb, validate_prompt_format

class TestPrompts(unittest.TestCase):
    @patch('utils.get_dynamodb_table')
    def test_update_prompt_success(self, mock_get_table):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response for existing prompt
        mock_table.query.return_value = {
            'Items': [{
                'ref': 'test_ref',
                'version': 1,
                'content': 'test content with {param1} and {param2}',
                'is_object': False,
                'createdAt': '2023-01-01T00:00:00'
            }]
        }

        # Test data
        ref = "test_ref"
        content = "new content with {question} and {business_context}"

        # Call the function
        result = update_prompt(ref, content)

        # Verify the result
        self.assertTrue(result)
        mock_table.put_item.assert_called_once()

    @patch('utils.get_dynamodb_table')
    def test_update_prompt_validation_failure(self, mock_get_table):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response for existing prompt
        mock_table.query.return_value = {
            'Items': [{
                'ref': 'test_ref',
                'version': 1,
                'content': 'test content with {param1} and {param2}',
                'is_object': False,
                'createdAt': '2023-01-01T00:00:00'
            }]
        }

        # Test data with invalid format (unknown parameter)
        ref = "test_ref"
        content = "new content with {param1} and {unknown_param}"

        # Call the function
        result = update_prompt(ref, content)

        # Verify the result
        self.assertFalse(result)
        mock_table.put_item.assert_not_called()

    @patch('utils.get_dynamodb_table')
    def test_get_prompt_with_params(self, mock_get_table):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response
        mock_table.query.return_value = {
            'Items': [{
                'ref': 'test_ref',
                'version': 1,
                'content': 'Hello {name}, welcome to {service}!',
                'is_object': False
            }]
        }

        # Test with valid parameters
        params = {'name': 'John', 'service': 'Prompt Manager'}
        result = get_prompt_from_dynamodb('test_ref', params)

        self.assertEqual(result, 'Hello John, welcome to Prompt Manager!')

    @patch('utils.get_dynamodb_table')
    def test_get_prompt_with_missing_params(self, mock_get_table):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response
        mock_table.query.return_value = {
            'Items': [{
                'ref': 'test_ref',
                'version': 1,
                'content': 'Hello {name}, welcome to {service}!',
                'is_object': False
            }]
        }

        # Test with missing parameter
        params = {'name': 'John'}

        with self.assertRaises(ValueError) as context:
            get_prompt_from_dynamodb('test_ref', params)

        self.assertIn("Missing parameters", str(context.exception))

    @patch('utils.get_dynamodb_table')
    def test_get_prompt_with_unused_params(self, mock_get_table):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response
        mock_table.query.return_value = {
            'Items': [{
                'ref': 'test_ref',
                'version': 1,
                'content': 'Hello {name}!',
                'is_object': False
            }]
        }

        # Test with extra unused parameter
        params = {'name': 'John', 'unused': 'Extra'}

        with self.assertRaises(ValueError) as context:
            get_prompt_from_dynamodb('test_ref', params)

        self.assertIn("Unused parameters", str(context.exception))

    def test_validate_prompt_format(self):
        # Valid prompt with known parameters
        valid_prompt = "This is a {question} with {business_context}"
        is_valid, error = validate_prompt_format(valid_prompt)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # Invalid prompt with unknown parameter
        invalid_prompt = "This has an {unknown_param}"
        is_valid, error = validate_prompt_format(invalid_prompt)
        self.assertFalse(is_valid)
        self.assertIn("Missing required parameter", error)

if __name__ == '__main__':
    unittest.main()
