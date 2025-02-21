import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from app import update_prompt

class TestPrompts(unittest.TestCase):
    @patch('boto3.resource')
    def test_update_prompt_success(self, mock_boto3_resource):
        # Mock DynamoDB table and response
        mock_table = MagicMock()
        mock_boto3_resource.return_value.Table.return_value = mock_table
        mock_table.update_item.return_value = {'Attributes': {'content': 'new content'}}

        # Test data
        ref = "test_ref"
        version = "1.0"
        content = "new content"

        # Call the function
        result = update_prompt(ref, version, content)

        # Verify the result
        self.assertTrue(result)
        mock_table.update_item.assert_called_once()
        call_args = mock_table.update_item.call_args[1]
        self.assertEqual(call_args['Key'], {'ref': ref, 'version': version})
        self.assertEqual(call_args['UpdateExpression'], 'SET content = :content, updatedAt = :timestamp')
        self.assertEqual(call_args['ExpressionAttributeValues'][':content'], content)

    @patch('boto3.resource')
    def test_update_prompt_failure(self, mock_boto3_resource):
        # Mock DynamoDB table and make it raise an exception
        mock_table = MagicMock()
        mock_boto3_resource.return_value.Table.return_value = mock_table
        mock_table.update_item.side_effect = Exception("DynamoDB error")

        # Test data
        ref = "test_ref"
        version = "1.0"
        content = "new content"

        # Call the function
        result = update_prompt(ref, version, content)

        # Verify the result
        self.assertFalse(result)
        mock_table.update_item.assert_called_once()

if __name__ == '__main__':
    unittest.main()
