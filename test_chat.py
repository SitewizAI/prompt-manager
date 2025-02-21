import pytest
from unittest.mock import MagicMock, patch
import litellm
import boto3
from utils import get_data

@patch('boto3.resource')
@patch('litellm.completion')
def test_chat_assistant_functionality(mock_completion, mock_boto3):
    # Mock DynamoDB tables
    mock_boto3.return_value.configure_mock(region_name='us-east-1')
    mock_table = MagicMock()
    mock_table.query.return_value = {
        'Items': [
            {
                'name': 'Test OKR',
                'description': 'Test Description',
                'timestamp': 1234567890000
            }
        ]
    }
    mock_boto3.return_value.Table.return_value = mock_table

    # Mock litellm response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test AI Response"
    mock_completion.return_value = mock_response

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

    # Test litellm completion call
    context = f"""
    Current OKRs:
    {data["okrs"][0]["markdown"]}
    """

    response = litellm.completion(
        model="litellm_proxy/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes website optimization data and provides insights. Use the provided context to answer questions accurately."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: test question"}
        ]
    )

    # Verify litellm was called with correct parameters
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args[1]
    assert call_args["model"] == "litellm_proxy/gpt-4o"
    assert len(call_args["messages"]) == 2
    assert call_args["messages"][0]["role"] == "system"
    assert call_args["messages"][1]["role"] == "user"

    # Verify response
    assert response.choices[0].message.content == "Test AI Response"
