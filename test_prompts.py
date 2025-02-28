import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json
from utils import update_prompt, validate_prompt_format, get_prompt_from_dynamodb

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
                'content': 'old content',
                'is_object': False,
                'createdAt': '2023-01-01T00:00:00'
            }]
        }

        # Test data
        ref = "test_ref"
        content = "new content with {question} parameter"

        # Mock validate_prompt_format to return success
        with patch('utils.validate_prompt_format') as mock_validate:
            mock_validate.return_value = (True, None)

            # Call the function
            result = update_prompt(ref, content)

            # Verify the result
            self.assertTrue(result)
            mock_table.put_item.assert_called_once()
            call_args = mock_table.put_item.call_args[1]
            self.assertEqual(call_args['Item']['ref'], ref)
            self.assertEqual(call_args['Item']['content'], content)
            self.assertEqual(call_args['Item']['version'], 2)  # Version incremented
            self.assertFalse(call_args['Item']['is_object'])

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
                'content': 'old content',
                'is_object': False,
                'createdAt': '2023-01-01T00:00:00'
            }]
        }

        # Test data
        ref = "test_ref"
        content = "new content with {invalid_param} parameter"

        # Mock validate_prompt_format to return failure
        with patch('utils.validate_prompt_format') as mock_validate:
            mock_validate.return_value = (False, "Unknown format variables in prompt: invalid_param")

            # Call the function
            result = update_prompt(ref, content)

            # Verify the result
            self.assertFalse(result)
            mock_table.put_item.assert_not_called()

    @patch('utils.get_dynamodb_table')
    def test_get_prompt_from_dynamodb(self, mock_get_table):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response
        mock_table.query.return_value = {
            'Items': [{
                'ref': 'test_ref',
                'version': 1,
                'content': 'Test prompt with {question} and {business_context}',
                'is_object': False,
                'createdAt': '2023-01-01T00:00:00'
            }]
        }

        # Test data
        ref = "test_ref"
        params = {
            'question': 'What is the meaning of life?',
            'business_context': 'Philosophy company'
        }

        # Call the function
        result = get_prompt_from_dynamodb(ref, params)

        # Verify the result
        expected = 'Test prompt with What is the meaning of life? and Philosophy company'
        self.assertEqual(result, expected)
        mock_table.query.assert_called_once()

    @patch('utils.get_dynamodb_table')
    def test_get_prompt_from_dynamodb_missing_param(self, mock_get_table):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response
        mock_table.query.return_value = {
            'Items': [{
                'ref': 'test_ref',
                'version': 1,
                'content': 'Test prompt with {question} and {business_context}',
                'is_object': False,
                'createdAt': '2023-01-01T00:00:00'
            }]
        }

        # Test data
        ref = "test_ref"
        params = {
            'question': 'What is the meaning of life?'
            # Missing business_context
        }

        # Call the function - it should return empty string on error
        result = get_prompt_from_dynamodb(ref, params)
        self.assertEqual(result, "")

    def test_validate_prompt_format_success(self):
        # Test data
        content = "Test prompt with {question} and {business_context}"

        # Call the function
        is_valid, error_msg = validate_prompt_format(content)

        # Verify the result
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    def test_validate_prompt_format_unknown_variable(self):
        # Test data
        content = "Test prompt with {question} and {unknown_variable}"

        # Call the function
        is_valid, error_msg = validate_prompt_format(content)

        # Verify the result
        self.assertFalse(is_valid)
        self.assertIn("unknown_variable", error_msg)

if __name__ == '__main__':
    unittest.main()
