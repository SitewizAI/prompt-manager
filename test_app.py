import pytest
from unittest.mock import patch, MagicMock
from app import get_prompts, get_recent_evaluations

@pytest.fixture
def mock_dynamodb():
    with patch('boto3.resource') as mock_resource:
        mock_table = MagicMock()
        mock_resource.return_value.Table.return_value = mock_table
        yield mock_table

def test_get_prompts_success(mock_dynamodb):
    # Mock data
    expected_items = [
        {"ref": "test1", "content": "content1", "version": "1"},
        {"ref": "test2", "content": "content2", "version": "2"}
    ]
    mock_dynamodb.scan.return_value = {"Items": expected_items}

    # Test
    result = get_prompts()

    # Verify
    assert result == expected_items
    mock_dynamodb.scan.assert_called_once()

def test_get_prompts_error(mock_dynamodb):
    # Mock error
    mock_dynamodb.scan.side_effect = Exception("Test error")

    # Test
    result = get_prompts()

    # Verify
    assert result == []
    mock_dynamodb.scan.assert_called_once()

def test_get_recent_evaluations_success(mock_dynamodb):
    # Mock data
    stream_key = "test-key"
    expected_items = [
        {
            "streamKey": stream_key,
            "timestamp": 1234567890,
            "type": "test",
            "success": True,
            "num_turns": 3
        }
    ]
    mock_dynamodb.query.return_value = {"Items": expected_items}

    # Test
    result = get_recent_evaluations(stream_key)

    # Verify
    assert result == expected_items
    mock_dynamodb.query.assert_called_once_with(
        KeyConditionExpression='streamKey = :sk',
        ExpressionAttributeValues={':sk': stream_key},
        ScanIndexForward=False,
        Limit=10
    )

def test_get_recent_evaluations_error(mock_dynamodb):
    # Mock error
    stream_key = "test-key"
    mock_dynamodb.query.side_effect = Exception("Test error")

    # Test
    result = get_recent_evaluations(stream_key)

    # Verify
    assert result == []
    mock_dynamodb.query.assert_called_once()
