import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from analyze_dynamodb_evals import (
    get_recent_evaluations,
    get_prompt_details,
    format_evaluation,
    save_to_file
)
import json
import os

@pytest.fixture
def mock_dynamodb():
    with patch('boto3.resource') as mock_resource:
        mock_table = MagicMock()
        mock_resource.return_value.Table.return_value = mock_table
        yield mock_table

@pytest.fixture
def sample_evaluation():
    return {
        'timestamp': 1645574400,  # 2022-02-23 00:00:00 UTC
        'question': 'Test question',
        'type': 'test_type',
        'success': True,
        'num_turns': 3,
        'failure_reasons': ['reason1', 'reason2'],
        'summary': 'Test summary',
        'conversation': 'Test conversation',
        'prompt_ref': 'test_ref'
    }

@pytest.fixture
def sample_prompt():
    return {
        'ref': 'test_ref',
        'name': 'Test Prompt',
        'description': 'Test Description'
    }

def test_get_recent_evaluations(mock_dynamodb):
    expected_items = [{'id': '1'}, {'id': '2'}]
    mock_dynamodb.query.return_value = {'Items': expected_items}

    result = get_recent_evaluations('test_stream', 2)

    mock_dynamodb.query.assert_called_once_with(
        KeyConditionExpression='streamKey = :sk',
        ExpressionAttributeValues={':sk': 'test_stream'},
        ScanIndexForward=False,
        Limit=2
    )
    assert result == expected_items

def test_get_prompt_details(mock_dynamodb):
    expected_item = {'ref': 'test_ref', 'content': 'test content'}
    mock_dynamodb.get_item.return_value = {'Item': expected_item}

    result = get_prompt_details('test_ref')

    mock_dynamodb.get_item.assert_called_once_with(
        Key={'ref': 'test_ref'}
    )
    assert result == expected_item

def test_format_evaluation(mock_dynamodb, sample_evaluation, sample_prompt):
    mock_dynamodb.get_item.return_value = {'Item': sample_prompt}

    formatted = format_evaluation(sample_evaluation)

    assert 'Evaluation at 2022-02-23 00:00:00' in formatted
    assert 'Question: Test question' in formatted
    assert 'Type: test_type' in formatted
    assert 'Success: True' in formatted
    assert 'Number of Turns: 3' in formatted
    assert 'reason1' in formatted
    assert 'reason2' in formatted
    assert 'Test summary' in formatted
    assert 'Test conversation' in formatted
    assert 'Test Prompt' in formatted
    assert 'Test Description' in formatted

def test_save_to_file(tmp_path):
    data = [{'test': 'data'}]
    filename = tmp_path / 'test.json'

    save_to_file(data, str(filename))

    assert filename.exists()
    with open(filename) as f:
        saved_data = json.load(f)
    assert saved_data == data

def test_get_recent_evaluations_empty(mock_dynamodb):
    mock_dynamodb.query.return_value = {'Items': []}

    result = get_recent_evaluations('test_stream')

    assert result == []

def test_get_prompt_details_not_found(mock_dynamodb):
    mock_dynamodb.get_item.return_value = {}

    result = get_prompt_details('nonexistent_ref')

    assert result == {}
