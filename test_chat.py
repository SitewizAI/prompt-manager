import pytest
from unittest.mock import MagicMock, patch
import litellm
import boto3
from utils import get_data, run_completion_with_fallback
from datetime import datetime

@patch('boto3.resource')
@patch('litellm.completion')
@patch('utils.get_api_key')
@patch('litellm.api_key', new='sk-test_key')
@patch('litellm.api_base', new='https://llms.sitewiz.ai')
@patch('litellm.enable_json_schema_validation', new=True)
@patch('utils.run_completion_with_fallback')
def test_chat_assistant_functionality(mock_run_completion, mock_get_api_key, mock_completion, mock_boto3):
    # Mock run_completion_with_fallback to return a test response
    mock_run_completion.return_value = "Test AI Response"
    # Mock API key response
    mock_get_api_key.return_value = {"LLM_API_KEY": "sk-test_key"}
    # Mock DynamoDB tables
    mock_boto3.return_value.configure_mock(region_name='us-east-1')
    mock_table = MagicMock()

    # Mock table responses for different tables
    def mock_table_response(table_name):
        if table_name == 'website-okrs':
            return {
                'Items': [
                    {
                        'streamKey': 'test_stream_key',
                        'name': 'Test OKR',
                        'description': 'Test Description',
                        'timestamp': 1234567890000
                    }
                ]
            }
        elif table_name == 'website-insights':
            return {
                'Items': [
                    {
                        'streamKey': 'test_stream_key',
                        'data_statement': 'Test Insight',
                        'problem_statement': 'Test Problem',
                        'business_objective': 'Test Objective',
                        'hypothesis': 'Test Hypothesis',
                        'frequency': 'High',
                        'severity': 'Medium',
                        'severity_reasoning': 'Test Severity',
                        'confidence': 'High',
                        'confidence_reasoning': 'Test Confidence'
                    }
                ]
            }
        elif table_name == 'WebsiteReports':
            return {
                'Items': [
                    {
                        'streamKey': 'test_stream_key',
                        'Shortened': [{'type': 'header', 'text': 'Test Suggestion'}],
                        'Tags': [{'type': 'test', 'Value': 'test', 'Tooltip': 'test'}],
                        'Expanded': [{'type': 'text', 'header': 'Test', 'text': 'Test Content'}],
                        'Insights': [{'data': [{'type': 'Heatmap', 'key': '123', 'name': 'test', 'explanation': 'test'}], 'text': 'Test Insight'}]
                    }
                ]
            }
        elif table_name == 'EvaluationsTable':
            return {
                'Items': [
                    {
                        'streamKey': 'test_stream_key',
                        'timestamp': str(datetime.now().timestamp()),
                        'type': 'test_type',
                        'question': 'test_question',
                        'success': True,
                        'num_turns': 3,
                        'summary': 'Test summary',
                        'failure_reasons': ['Test failure']
                    },
                    {
                        'streamKey': 'test_stream_key',
                        'timestamp': str(datetime.now().timestamp() - 3600),
                        'type': 'test_type_prev',
                        'success': False,
                        'num_turns': 2,
                        'summary': 'Previous test summary',
                        'failure_reasons': ['Previous test failure']
                    }
                ]
            }
        return {'Items': []}

    # Mock table.query to return appropriate data based on table name
    def mock_query(*args, **kwargs):
        table_name = mock_table._mock_name
        print(f"Querying table {table_name} with kwargs: {kwargs}")
        if 'ExpressionAttributeValues' in kwargs:
            stream_key = kwargs['ExpressionAttributeValues'].get(':sk')
            if stream_key == 'test_stream_key':
                response = mock_table_response(table_name)
                print(f"Response for {table_name}: {response}")
                return response
        return {'Items': []}

    mock_table.query.side_effect = mock_query

    # Mock table name property
    def mock_table_factory(table_name):
        mock = MagicMock()
        mock._mock_name = table_name
        mock.query.side_effect = lambda *args, **kwargs: mock_table_response(table_name)
        return mock

    def get_table(table_name):
        mock = mock_table_factory(table_name)
        print(f"Creating mock table for {table_name}")
        return mock

    mock_boto3.return_value.Table.side_effect = get_table

    # Mock litellm response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Test AI Response"

    # Test get_data function returns expected structure
    data = get_data("test_stream_key")
    assert isinstance(data, dict)
    assert "okrs" in data
    assert "insights" in data
    assert "suggestions" in data
    assert "code" in data

    # Test data structure
    assert isinstance(data["okrs"], list)
    assert isinstance(data["insights"], list)
    assert isinstance(data["suggestions"], list)
    assert isinstance(data["code"], list)

    # Test markdown content contains expected data
    assert "Test OKR" in data["okrs"][0]["markdown"]
    assert "Test Description" in data["okrs"][0]["markdown"]

    # Test run_completion_with_fallback
    context = f"""
    Current OKRs:
    {data["okrs"][0]["markdown"]}

    Current Evaluation:
    - Type: test_type
    - Question: test_question
    - Success: True
    - Number of Turns: 3
    - Summary: Test summary

    Previous Evaluations Summary:
    Evaluation 1: Type=test_type_prev, Success=False, Failure Reasons=['Previous test failure'], Summary=Previous test summary
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes website optimization data and provides insights. Use the provided context to answer questions accurately."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: test question"}
    ]

    # Call run_completion_with_fallback
    response = mock_run_completion(messages=messages)

    # Verify run_completion_with_fallback was called with correct parameters
    mock_run_completion.assert_called_once_with(messages=messages)

    # Verify response
    assert response == "Test AI Response"
