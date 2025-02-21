import pytest
from unittest.mock import MagicMock, patch, call
from app import get_all_prompts, get_all_evaluations

@pytest.fixture
def mock_dynamodb():
    with patch('boto3.resource') as mock_resource:
        mock_table = MagicMock()
        mock_resource.return_value.Table.return_value = mock_table
        yield mock_table

def test_get_all_prompts_success(mock_dynamodb):
    # Mock data
    mock_prompts = [
        {'ref': 'test1', 'content': 'content1', 'version': '1'},
        {'ref': 'test2', 'content': 'content2', 'version': '2'}
    ]

    # Mock the scan response
    mock_dynamodb.scan.return_value = {'Items': mock_prompts}

    # Call the function
    result = get_all_prompts()

    # Verify the results
    assert len(result) == 2
    assert result == mock_prompts
    mock_dynamodb.scan.assert_called_once()

def test_get_all_prompts_pagination(mock_dynamodb):
    # Mock data for pagination
    mock_prompts1 = [{'ref': 'test1', 'content': 'content1'}]
    mock_prompts2 = [{'ref': 'test2', 'content': 'content2'}]

    # Mock responses with pagination
    mock_dynamodb.scan.side_effect = [
        {'Items': mock_prompts1, 'LastEvaluatedKey': 'key1'},
        {'Items': mock_prompts2}
    ]



    # Call the function
    result = get_all_prompts()

    # Verify the results
    assert len(result) == 2
    assert result == mock_prompts1 + mock_prompts2
    assert mock_dynamodb.scan.call_count == 2

    # Verify the scan calls were made with correct arguments
    mock_dynamodb.scan.assert_has_calls([
        call(),  # First call with no arguments
        call(ExclusiveStartKey='key1')  # Second call with ExclusiveStartKey
    ])

def test_get_all_evaluations_success(mock_dynamodb):
    # Mock stream keys scan
    mock_stream_keys = [
        {'streamKey': 'stream1'},
        {'streamKey': 'stream2'}
    ]
    mock_dynamodb.scan.return_value = {'Items': mock_stream_keys}

    # Mock evaluations for each stream
    mock_evals1 = [{'streamKey': 'stream1', 'timestamp': 1234567890}]
    mock_evals2 = [{'streamKey': 'stream2', 'timestamp': 1234567891}]

    mock_dynamodb.query.side_effect = [
        {'Items': mock_evals1},
        {'Items': mock_evals2}
    ]

    # Call the function
    result = get_all_evaluations()

    # Verify the results
    assert len(result) == 2
    assert result == mock_evals1 + mock_evals2
    assert mock_dynamodb.query.call_count == 2

def test_get_all_evaluations_empty(mock_dynamodb):
    # Mock empty stream keys
    mock_dynamodb.scan.return_value = {'Items': []}

    # Call the function
    result = get_all_evaluations()

    # Verify the results
    assert len(result) == 0
    mock_dynamodb.query.assert_not_called()

def test_get_all_evaluations_error(mock_dynamodb):
    # Mock an error
    mock_dynamodb.scan.side_effect = Exception("DynamoDB error")

    # Call the function
    result = get_all_evaluations()

    # Verify the results
    assert len(result) == 0
