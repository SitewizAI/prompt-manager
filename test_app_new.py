import pytest
from unittest.mock import MagicMock, patch
from app import get_prompt_content, get_recent_evals
from datetime import datetime

@pytest.fixture
def mock_dynamodb():
    with patch('boto3.resource') as mock_resource:
        mock_table = MagicMock()
        mock_resource.return_value.Table.return_value = mock_table
        yield mock_table

def test_get_prompt_content_success(mock_dynamodb):
    # Mock response data
    mock_response = {
        'Item': {
            'ref': 'test_ref',
            'content': 'Test prompt content',
            'version': '1.0'
        }
    }
    mock_dynamodb.get_item.return_value = mock_response

    # Test the function
    result = get_prompt_content('test_ref')

    # Verify the result
    assert result == mock_response['Item']
    mock_dynamodb.get_item.assert_called_once_with(Key={'ref': 'test_ref'})

def test_get_prompt_content_not_found(mock_dynamodb):
    # Mock empty response
    mock_dynamodb.get_item.return_value = {}

    # Test the function
    result = get_prompt_content('nonexistent_ref')

    # Verify the result
    assert result == {}

def test_get_recent_evals_success(mock_dynamodb):
    # Mock response data
    timestamp = int(datetime.now().timestamp())
    mock_response = {
        'Items': [
            {
                'type': 'test_type',
                'streamKey': 'test_stream',
                'failure_reasons': [],
                'num_turns': 3,
                'success': True,
                'timestamp': timestamp,
                'question': 'test question'
            }
        ]
    }
    mock_dynamodb.query.return_value = mock_response

    # Test the function
    result = get_recent_evals('test_stream', 5)

    # Verify the result
    assert len(result) == 1
    assert result[0]['type'] == 'test_type'
    assert result[0]['stream_key'] == 'test_stream'
    assert result[0]['num_turns'] == 3
    assert result[0]['success'] is True
    assert result[0]['question'] == 'test question'

    # Verify the query parameters
    mock_dynamodb.query.assert_called_once_with(
        KeyConditionExpression='streamKey = :sk',
        ExpressionAttributeValues={':sk': 'test_stream'},
        ScanIndexForward=False,
        Limit=5
    )

def test_get_recent_evals_empty(mock_dynamodb):
    # Mock empty response
    mock_dynamodb.query.return_value = {'Items': []}

    # Test the function
    result = get_recent_evals('test_stream')

    # Verify the result
    assert result == []
