import os
import json
import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock the weave module
weave_mock = MagicMock()
with patch.dict('sys.modules', {'weave': weave_mock}):
    from analyze_weave_refs import save_to_s3
    from app import load_from_s3

@pytest.fixture
def mock_s3_client():
    mock_client = MagicMock()
    mock_client.put_object.return_value = {}  # Mock successful upload
    mock_body = MagicMock()
    mock_body.read.return_value = b'{"test": "data"}'
    mock_client.get_object.return_value = {'Body': mock_body}  # Mock successful download
    with patch('boto3.client', return_value=mock_client), \
         patch.dict('os.environ', {
             'AWS_ACCESS_KEY_ID': 'dummy_access_key',
             'AWS_SECRET_ACCESS_KEY': 'dummy_secret_key',
             'AWS_REGION': 'us-east-1'
         }), \
         patch('botocore.client.BaseClient._make_api_call', return_value={}), \
         patch('botocore.credentials.Credentials', return_value=MagicMock()), \
         patch('botocore.auth.SigV4Auth', return_value=MagicMock()), \
         patch('botocore.client.ClientCreator', return_value=MagicMock()), \
         patch('botocore.config.Config', return_value=MagicMock()), \
         patch('botocore.session.Session', return_value=MagicMock()), \
         patch('botocore.client.Config', return_value=MagicMock()), \
         patch('botocore.auth.SIGV4_TIMESTAMP', return_value=MagicMock()), \
         patch('botocore.auth.datetime', return_value=MagicMock()), \
         patch('botocore.auth.datetime.datetime', return_value=MagicMock()), \
         patch('botocore.auth.datetime.datetime.utcnow', return_value=MagicMock()), \
         patch('botocore.auth.datetime.datetime.now', return_value=MagicMock()), \
         patch('botocore.auth.datetime.datetime.now.utcnow', return_value=MagicMock()), \
         patch('botocore.auth.datetime.datetime.now.utcnow.utcnow', return_value=MagicMock()):
        yield mock_client

def test_save_to_s3(mock_s3_client):
    # Test data
    test_data = {"test": "data"}
    filename = "test.json"
    bucket_name = "sitewiz-prompts"

    # Call function
    save_to_s3(test_data, filename, bucket_name)

    # Verify S3 client was called correctly
    mock_s3_client.put_object.assert_called_once_with(
        Bucket=bucket_name,
        Key=filename,
        Body=json.dumps(test_data, indent=2),
        ContentType='application/json'
    )

def test_load_from_s3(mock_s3_client):
    # Test data
    test_data = {"test": "data"}
    filename = "test.json"
    bucket_name = "sitewiz-prompts"

    # Mock S3 response
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(test_data).encode('utf-8')
    mock_s3_client.get_object.return_value = {'Body': mock_body}

    # Call function
    result = load_from_s3(filename, bucket_name)

    # Verify result
    assert result == test_data
    mock_s3_client.get_object.assert_called_once_with(
        Bucket=bucket_name,
        Key=filename
    )

def test_save_to_s3_fallback(mock_s3_client):
    # Make S3 upload fail
    mock_s3_client.put_object.side_effect = Exception("S3 error")

    # Test data
    test_data = {"test": "data"}
    filename = "test.json"

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Call function
    save_to_s3(test_data, filename)

    # Verify file was saved locally
    with open(os.path.join("output", filename)) as f:
        saved_data = json.load(f)
        assert saved_data == test_data

def test_load_from_s3_fallback(mock_s3_client):
    # Make S3 download fail
    mock_s3_client.get_object.side_effect = Exception("S3 error")

    # Test data
    test_data = {"test": "data"}
    filename = "test.json"

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Create local file
    with open(os.path.join("output", filename), 'w') as f:
        json.dump(test_data, f)

    # Call function
    result = load_from_s3(filename)

    # Verify result
    assert result == test_data
