import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from analyze_dynamodb_data import get_recent_evaluations, get_prompt_by_ref, format_evaluation

@pytest.fixture
def mock_dynamodb():
    with patch('boto3.resource') as mock_resource:
        mock_table = MagicMock()
        mock_resource.return_value.Table.return_value = mock_table
        yield mock_table

def test_get_recent_evaluations(mock_dynamodb):
    expected_items = [
        {'streamKey': 'test-key', 'timestamp': 1234567890}
    ]
    mock_dynamodb.query.return_value = {'Items': expected_items}

    result = get_recent_evaluations('test-key', 10)

    mock_dynamodb.query.assert_called_once_with(
        KeyConditionExpression='streamKey = :sk',
        ExpressionAttributeValues={':sk': 'test-key'},
        ScanIndexForward=False,
        Limit=10
    )
    assert result == expected_items

def test_get_prompt_by_ref(mock_dynamodb):
    expected_item = {
        'ref': 'test-ref',
        'content': 'test content',
        'version': '1.0'
    }
    mock_dynamodb.get_item.return_value = {'Item': expected_item}

    result = get_prompt_by_ref('test-ref')

    mock_dynamodb.get_item.assert_called_once_with(
        Key={'ref': 'test-ref'}
    )
    assert result == expected_item

def test_format_evaluation():
    eval_data = {
        'timestamp': 1234567890,
        'question': 'test question',
        'type': 'test type',
        'success': True,
        'num_turns': 2,
        'failure_reasons': ['reason1'],
        'summary': 'test summary',
        'prompt_ref': 'test-ref',
        'conversation': 'test conversation'
    }

    with patch('analyze_dynamodb_data.get_prompt_by_ref') as mock_get_prompt:
        mock_get_prompt.return_value = {
            'ref': 'test-ref',
            'content': 'test content',
            'version': '1.0'
        }

        result = format_evaluation(eval_data)

        assert 'test question' in result
        assert 'test type' in result
        assert 'True' in result
        assert 'reason1' in result
        assert 'test summary' in result
        assert 'test-ref' in result
        assert 'test content' in result
        assert '1.0' in result
        assert 'test conversation' in result
