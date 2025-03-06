"""
Tests for the conversation history functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
from decimal import Decimal
import json
import io

from utils.metrics_utils import get_conversation_history
from utils.db_utils import get_boto3_client

class TestConversationHistory(unittest.TestCase):
    """Test cases for the conversation history functionality."""

    @patch('utils.metrics_utils.get_dynamodb_table')
    @patch('utils.metrics_utils.get_boto3_client')
    def test_get_conversation_from_s3(self, mock_get_boto3_client, mock_get_dynamodb_table):
        """Test fetching conversation history from S3."""
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_dynamodb_table.return_value = mock_table

        # Set up DynamoDB response with conversation_key
        mock_table.get_item.return_value = {
            'Item': {
                'type': 'test_type',
                'conversation_key': 'test/conversation/key.json'
            }
        }

        # Mock S3 client and response
        mock_s3_client = MagicMock()
        mock_get_boto3_client.return_value = mock_s3_client

        # Create mock S3 response with conversation content
        mock_conversation_content = json.dumps([
            {"role": "user", "message": "Hello"},
            {"role": "assistant", "message": "Hi there!"}
        ])
        mock_body = io.BytesIO(mock_conversation_content.encode('utf-8'))
        mock_s3_client.get_object.return_value = {
            'Body': mock_body
        }

        # Call the function
        result = get_conversation_history('test_stream', 1234567890.0, 'test_type')

        # Verify DynamoDB was called correctly
        mock_table.get_item.assert_called_once_with(
            Key={
                'streamKey': 'test_stream',
                'timestamp': Decimal('1234567890.0')
            },
            ProjectionExpression='conversation_key,#t',
            ExpressionAttributeNames={
                '#t': 'type'
            }
        )

        # Verify S3 was called correctly
        mock_get_boto3_client.assert_called_with('s3')
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='sitewiz-websites',
            Key='test/conversation/key.json'
        )

        # Verify the result is the conversation content
        self.assertEqual(result, mock_conversation_content)

    @patch('utils.metrics_utils.get_dynamodb_table')
    def test_no_conversation_key(self, mock_get_dynamodb_table):
        """Test handling when no conversation_key is found."""
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_dynamodb_table.return_value = mock_table

        # Set up DynamoDB response without conversation_key
        mock_table.get_item.return_value = {
            'Item': {
                'type': 'test_type'
                # No conversation_key
            }
        }

        # Call the function
        result = get_conversation_history('test_stream', 1234567890.0, 'test_type')

        # Verify result is empty string
        self.assertEqual(result, "")

    @patch('utils.metrics_utils.get_dynamodb_table')
    def test_no_item_found(self, mock_get_dynamodb_table):
        """Test handling when no item is found in DynamoDB."""
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_get_dynamodb_table.return_value = mock_table

        # Set up DynamoDB response with no Item
        mock_table.get_item.return_value = {}

        # Call the function
        result = get_conversation_history('test_stream', 1234567890.0, 'test_type')

        # Verify result is empty string
        self.assertEqual(result, "")

if __name__ == '__main__':
    unittest.main()
