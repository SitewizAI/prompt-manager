import unittest
from unittest.mock import patch, MagicMock
import json
from lambda_function import lambda_handler, analyze_issue_with_llm, create_github_issue

class TestLambdaFunction(unittest.TestCase):
    def setUp(self):
        self.test_event = {
            'issue_content': 'Test issue content',
            'github_token': 'test_token',
            'repo': 'test/repo',
            'title': 'Test Issue'
        }

        self.mock_analysis = {
            'root_cause': 'Test root cause',
            'solution_steps': ['Step 1', 'Step 2'],
            'estimated_effort': 'Low'
        }

        self.mock_github_response = {
            'html_url': 'https://github.com/test/repo/issues/1'
        }

    @patch('lambda_function.analyze_issue_with_llm')
    @patch('lambda_function.create_github_issue')
    def test_lambda_handler_success(self, mock_create_issue, mock_analyze):
        # Set up mocks
        mock_analyze.return_value = self.mock_analysis
        mock_create_issue.return_value = self.mock_github_response

        # Call lambda handler
        response = lambda_handler(self.test_event, None)

        # Verify response
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        self.assertEqual(body['issue_url'], 'https://github.com/test/repo/issues/1')
        self.assertEqual(body['analysis'], self.mock_analysis)

        # Verify mocks were called correctly
        mock_analyze.assert_called_once_with('Test issue content')
        mock_create_issue.assert_called_once()

    def test_lambda_handler_missing_params(self):
        # Test with missing parameters
        incomplete_event = {
            'issue_content': 'Test content'
            # Missing other required params
        }

        response = lambda_handler(incomplete_event, None)

        self.assertEqual(response['statusCode'], 400)
        body = json.loads(response['body'])
        self.assertIn('error', body)
        self.assertIn('Missing required parameters', body['error'])

    @patch('lambda_function.analyze_issue_with_llm')
    def test_lambda_handler_analysis_failure(self, mock_analyze):
        # Mock analysis failure
        mock_analyze.return_value = None

        response = lambda_handler(self.test_event, None)

        self.assertEqual(response['statusCode'], 500)
        body = json.loads(response['body'])
        self.assertIn('error', body)
        self.assertIn('Failed to analyze issue with LLM', body['error'])

    @patch('lambda_function.create_github_issue')
    @patch('lambda_function.analyze_issue_with_llm')
    def test_lambda_handler_github_error(self, mock_analyze, mock_create_issue):
        # Set up mocks
        mock_analyze.return_value = self.mock_analysis
        mock_create_issue.side_effect = Exception('GitHub API error')

        response = lambda_handler(self.test_event, None)

        self.assertEqual(response['statusCode'], 500)
        body = json.loads(response['body'])
        self.assertIn('error', body)
        self.assertIn('GitHub API error', body['error'])

if __name__ == '__main__':
    unittest.main()
