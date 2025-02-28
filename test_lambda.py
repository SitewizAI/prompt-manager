import os
import json
import unittest
from unittest.mock import patch, MagicMock
from lambda_function import (
    lambda_handler,
    AnalysisResponse,
    PromptChange
)
from utils import validate_prompt_format, get_prompt_from_dynamodb

class TestLambdaFunction(unittest.TestCase):
    def setUp(self):
        self.mock_github_response = {
            "id": 1,
            "html_url": "https://github.com/test/repo/issues/1",
            "title": "Test Issue",
            "body": "Test Body",
            "number": 1
        }

        self.mock_llm_response = {
            "prompt_changes": [
                {
                    "ref": "test-prompt",
                    "content": "Updated content",
                    "reason": "Test reason"
                }
            ]
        }

    def test_run_completion_with_fallback_with_validation_errors(self):
        with patch("utils.run_completion_with_fallback") as mock_run_completion, \
             patch("utils.initialize_vertex_ai") as mock_init_ai:
            # Skip AI initialization
            mock_init_ai.return_value = None

            # Set up the mock to return a value on the first call
            mock_run_completion.return_value = {"prompt_changes": [{"ref": "test-ref", "content": "test content", "reason": "test reason"}]}

            # Create a test event
            event = {
                "type": "test",
                "additional_instructions": "test instructions"
            }

            # Call the lambda handler
            with patch("lambda_function.get_most_recent_stream_key", return_value=("test-key", "test-timestamp")), \
                 patch("lambda_function.get_context", return_value="test context"), \
                 patch("lambda_function.update_prompt", return_value=True), \
                 patch("lambda_function.token_counter", return_value=100):

                result = lambda_handler(event, None)

                # Verify the result
                self.assertEqual(result["statusCode"], 200)
                body = json.loads(result["body"])
                self.assertIn("updated_prompts", body["results"])
                self.assertEqual(len(body["results"]["updated_prompts"]), 1)

    def test_lambda_handler_with_validation_errors(self):
        with patch("utils.run_completion_with_fallback") as mock_run_completion, \
             patch("utils.initialize_vertex_ai") as mock_init_ai:
            # Skip AI initialization
            mock_init_ai.return_value = None

            # Set up the mock to return different values on subsequent calls
            mock_run_completion.side_effect = [
                {"prompt_changes": [{"ref": "test-ref", "content": "invalid content", "reason": "test reason"}]},
                {"prompt_changes": [{"ref": "test-ref", "content": "valid content", "reason": "test reason"}]}
            ]

            # Create a test event
            event = {
                "type": "test",
                "additional_instructions": "test instructions"
            }

            # Call the lambda handler
            with patch("lambda_function.get_most_recent_stream_key", return_value=("test-key", "test-timestamp")), \
                 patch("lambda_function.get_context", return_value="test context"), \
                 patch("lambda_function.update_prompt", side_effect=[False, True]), \
                 patch("lambda_function.token_counter", return_value=100):

                result = lambda_handler(event, None)

                # Verify the result
                self.assertEqual(result["statusCode"], 200)
                body = json.loads(result["body"])
                self.assertIn("updated_prompts", body["results"])
                self.assertEqual(len(body["results"]["updated_prompts"]), 1)

                # Verify that run_completion_with_fallback was called twice
                self.assertEqual(mock_run_completion.call_count, 2)

                # Verify that the second call included validation errors
                args, kwargs = mock_run_completion.call_args_list[1]
                self.assertIn("validation_errors", kwargs)

    def test_lambda_handler_error(self):
        event = {
            "type": "test"
        }
        with patch("lambda_function.get_most_recent_stream_key", side_effect=Exception("Test error")):
            result = lambda_handler(event, None)
            self.assertEqual(result["statusCode"], 500)
            self.assertIn("error", json.loads(result["body"]))

    def test_lambda_handler_max_retries(self):
        with patch("utils.run_completion_with_fallback") as mock_run_completion, \
             patch("utils.initialize_vertex_ai") as mock_init_ai:
            # Skip AI initialization
            mock_init_ai.return_value = None

            # Set up the mock to always return invalid content
            mock_run_completion.return_value = {"prompt_changes": [{"ref": "test-ref", "content": "invalid content", "reason": "test reason"}]}

            # Create a test event
            event = {
                "type": "test"
            }

            # Call the lambda handler
            with patch("lambda_function.get_most_recent_stream_key", return_value=("test-key", "test-timestamp")), \
                 patch("lambda_function.get_context", return_value="test context"), \
                 patch("lambda_function.update_prompt", return_value=False), \
                 patch("lambda_function.token_counter", return_value=100):  # Always fail validation

                result = lambda_handler(event, None)

                # Verify the result
                self.assertEqual(result["statusCode"], 200)
                body = json.loads(result["body"])
                self.assertIn("errors", body["results"])
                self.assertGreater(len(body["results"]["errors"]), 0)

                # Verify that run_completion_with_fallback was called 3 times (initial + 2 retries)
                self.assertEqual(mock_run_completion.call_count, 3)
