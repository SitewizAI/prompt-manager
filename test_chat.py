import pytest
from unittest.mock import MagicMock, patch
import litellm
import boto3
from datetime import datetime, timezone
from utils import get_data, run_completion_with_fallback
from chat import get_recent_evaluations, get_chat_context, chat_with_context

@patch('boto3.resource')
@patch('utils.run_completion_with_fallback')
@patch('utils.get_api_key')
@patch('litellm.completion')
@patch('litellm.api_key', new="sk-test-key")
@patch('litellm.api_base', new="https://llms.sitewiz.ai")
@patch('litellm.enable_json_schema_validation', new=True)
@patch('litellm.proxy', new=False)
@patch('utils.model_fallback_list', new=["video", "main"])
@patch('litellm.utils.trim_messages')
@patch('litellm.utils.proxy_request', new=False)
@patch('litellm.utils.proxy_server_request', new=False)
def test_chat_functionality(mock_trim_messages, mock_litellm_completion, mock_get_api_key, mock_completion, mock_boto3):
    # Mock API key response
    mock_get_api_key.return_value = {"LLM_API_KEY": "sk-test-key"}

    # Mock litellm completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test AI Response"
    mock_litellm_completion.return_value = mock_response

    # Mock run_completion_with_fallback to return the test response directly
    mock_completion.return_value = "Test AI Response"

    # Mock trim_messages to return the same messages
    mock_trim_messages.side_effect = lambda messages, model: messages

    # Mock initialize_vertex_ai to do nothing
    with patch('utils.initialize_vertex_ai') as mock_init:
        # Mock DynamoDB tables
        mock_boto3.return_value.configure_mock(region_name='us-east-1')
        mock_table = MagicMock()

        # Mock evaluations data
        mock_evaluations = {
            'Items': [
                {
                    'streamKey': 'test_stream_key',
                    'timestamp': 1234567890000,
                    'conversation': ['Test conversation'],
                    'results': {'test': 'result'},
                    'prompt_refs': ['test_prompt'],
                    'failure_reason': 'Test failure',
                    'summary': 'Test summary'
                },
                {
                    'streamKey': 'test_stream_key',
                    'timestamp': 1234567880000,
                    'failure_reason': 'Previous failure',
                    'summary': 'Previous summary'
                }
            ]
        }

        # Mock OKRs data
        mock_okrs = {
            'Items': [
                {
                    'name': 'Test OKR',
                    'description': 'Test Description',
                    'timestamp': 1234567890000
                }
            ]
        }

        # Mock insights data
        mock_insights = {
            'Items': [
                {
                    'data_statement': 'Test insight',
                    'problem_statement': 'Test problem',
                    'business_objective': 'Test objective',
                    'hypothesis': 'Test hypothesis'
                }
            ]
        }

        # Mock suggestions data
        mock_suggestions = {
            'Items': [
                {
                    'Shortened': [{'type': 'header', 'text': 'Test suggestion'}],
                    'Tags': [{'type': 'test', 'Value': 'test', 'Tooltip': 'test'}]
                }
            ]
        }

        # Mock prompts data
        mock_prompts = {
            'Item': {
                'ref': 'test_prompt',
                'content': 'Test prompt content'
            }
        }

        def mock_table_side_effect(table_name):
            mock_table_instance = MagicMock()
            if table_name == 'website-evaluations':
                mock_table_instance.query.return_value = mock_evaluations
            elif table_name == 'website-okrs':
                mock_table_instance.query.return_value = mock_okrs
            elif table_name == 'website-insights':
                mock_table_instance.query.return_value = mock_insights
            elif table_name == 'WebsiteReports':
                mock_table_instance.query.return_value = mock_suggestions
            elif table_name == 'PromptsTable':
                mock_table_instance.get_item.return_value = mock_prompts
            return mock_table_instance

        mock_boto3.return_value.Table.side_effect = mock_table_side_effect

        # Test get_recent_evaluations
        evaluations = get_recent_evaluations("test_stream_key")
        assert len(evaluations) == 2
        assert evaluations[0]['timestamp'] == 1234567890000

        # Test get_chat_context
        context = get_chat_context("test_stream_key")
        assert context is not None
        assert 'most_recent_evaluation' in context
        assert 'previous_evaluations' in context
        assert 'current_data' in context
        assert 'prompts' in context

        # Test chat_with_context
        response = chat_with_context("test_stream_key", "test question")
        assert response == "Test AI Response"

        # Verify run_completion_with_fallback was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[0]
        assert len(call_args[0]) == 2  # messages list
        assert call_args[0][0]["role"] == "system"
        assert call_args[0][1]["role"] == "user"
        assert "test question" in call_args[0][1]["content"]


