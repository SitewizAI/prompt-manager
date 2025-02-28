import os
import json
import unittest
from unittest.mock import patch, MagicMock
from lambda_function import lambda_handler, AnalysisResponse, PromptChange

class TestLambdaFunction(unittest.TestCase):
    def setUp(self):
        self.mock_llm_response = {
            "prompt_changes": [
                {
                    "ref": "test-prompt",
                    "content": "Updated content",
                    "reason": "Test reason"
                }
            ]
        }

        self.mock_context = {
            "stream_key": "test-stream",
            "timestamp": "2023-01-01T00:00:00"
        }

    @patch('lambda_function.get_most_recent_stream_key')
    @patch('lambda_function.get_context')
    @patch('lambda_function.run_completion_with_fallback')
    @patch('lambda_function.update_prompt')
    def test_lambda_handler_success(self, mock_update_prompt, mock_run_completion, mock_get_context, mock_get_stream_key):
        # Configure mocks
        mock_get_stream_key.return_value = ("test-stream", "2023-01-01T00:00:00")
        mock_get_context.return_value = "Test context"
        mock_run_completion.return_value = self.mock_llm_response
        mock_update_prompt.return_value = True

        # Test event
        event = {}

        # Call the function
        result = lambda_handler(event, None)

        # Verify results
        self.assertEqual(result['statusCode'], 200)
        body = json.loads(result['body'])
        self.assertEqual(body['message'], 'System analysis complete')
        self.assertEqual(len(body['results']['updated_prompts']), 1)

        # Verify the function was called with retry logic
        mock_run_completion.assert_called_once()
        mock_update_prompt.assert_called_once_with('test-prompt', 'Updated content')

    @patch('lambda_function.get_most_recent_stream_key')
    @patch('lambda_function.get_context')
    @patch('lambda_function.run_completion_with_fallback')
    def test_lambda_handler_with_validation_error(self, mock_run_completion, mock_get_context, mock_get_stream_key):
        # Configure mocks
        mock_get_stream_key.return_value = ("test-stream", "2023-01-01T00:00:00")
        mock_get_context.return_value = "Test context"

        # First call raises ValueError, second call succeeds
        mock_run_completion.side_effect = [
            ValueError("Missing parameter in prompt"),
            self.mock_llm_response
        ]

        # Test event
        event = {}

        # Call the function
        result = lambda_handler(event, None)

        # Verify results
        self.assertEqual(result['statusCode'], 200)

        # Verify the function was called twice (retry after error)
        self.assertEqual(mock_run_completion.call_count, 2)

    @patch('lambda_function.get_most_recent_stream_key')
    def test_lambda_handler_no_evaluations(self, mock_get_stream_key):
        # Configure mock to return no stream key
        mock_get_stream_key.return_value = (None, None)

        # Test event
        event = {}

        # Call the function
        result = lambda_handler(event, None)

        # Verify results
        self.assertEqual(result['statusCode'], 500)
        body = json.loads(result['body'])
        self.assertIn('error', body)
        self.assertIn('No evaluations found', body['error'])

if __name__ == '__main__':
    unittest.main()
