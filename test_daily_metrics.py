import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
from daily_metrics_lambda import aggregate_daily_metrics

class TestDailyMetricsLambda(unittest.TestCase):
    def setUp(self):
        self.mock_table = MagicMock()
        self.mock_dynamodb = MagicMock()
        self.mock_dynamodb.Table.return_value = self.mock_table

        # Set up patch for boto3.resource
        self.patcher = patch('boto3.resource', return_value=self.mock_dynamodb)
        self.mock_boto3 = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_aggregate_daily_metrics(self):
        # Mock data for yesterday's evaluations
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        yesterday_date = yesterday.strftime('%Y-%m-%d')
        start_timestamp = int(yesterday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        end_timestamp = int(yesterday.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp())

        # Mock get_item response for check_existing_metrics
        self.mock_table.get_item.return_value = {}  # No existing metrics

        # Mock query response with sample evaluations for different types
        self.mock_table.query.side_effect = [
            # Response for 'okr' type
            {
                'Items': [
                    {
                        'type': 'okr',
                        'timestamp': Decimal(str(start_timestamp + 3600)),
                        'num_turns': 5,
                        'success': True,
                        'attempts': 2
                    }
                ]
            },
            # Response for 'insights' type
            {
                'Items': [
                    {
                        'type': 'insights',
                        'timestamp': Decimal(str(start_timestamp + 7200)),
                        'num_turns': 3,
                        'success': False,
                        'attempts': 1
                    },
                    {
                        'type': 'insights',
                        'timestamp': Decimal(str(start_timestamp + 10800)),
                        'num_turns': 4,
                        'success': True,
                        'attempts': 1
                    }
                ]
            },
            # Response for 'suggestion' type
            {
                'Items': [
                    {
                        'type': 'suggestion',
                        'timestamp': Decimal(str(start_timestamp + 14400)),
                        'num_turns': 2,
                        'success': True,
                        'attempts': 1
                    }
                ]
            },
            # Response for 'code' type
            {
                'Items': []
            },
            # Response for 'design' type
            {
                'Items': []
            }
        ]

        # Call the function
        result = aggregate_daily_metrics({}, {})

        # Verify the DynamoDB table was queried correctly for each type
        self.assertEqual(self.mock_table.query.call_count, 5)  # One call for each type

        # Check the first call for 'okr' type
        first_call = self.mock_table.query.call_args_list[0]
        self.assertEqual(first_call.kwargs['IndexName'], 'type-timestamp-index')
        self.assertEqual(first_call.kwargs['KeyConditionExpression'], '#type = :type_val AND #ts BETWEEN :start AND :end')
        self.assertEqual(first_call.kwargs['ExpressionAttributeNames'], {'#type': 'type', '#ts': 'timestamp'})
        self.assertEqual(first_call.kwargs['ExpressionAttributeValues'][':type_val'], 'okr')

        # Verify that put_item was called with correct aggregated metrics for each type
        put_item_calls = self.mock_table.put_item.call_args_list
        self.assertEqual(len(put_item_calls), 3)  # One for each type with data (okr, insights, suggestion)

        # Check okr metrics
        okr_call = next(call for call in put_item_calls
                        if call.kwargs['Item']['type'] == 'okr')
        okr_data = okr_call.kwargs['Item']['data']
        self.assertEqual(okr_data['evaluations'], 1)
        self.assertEqual(okr_data['successes'], 1)
        self.assertEqual(okr_data['attempts'], 2)
        self.assertEqual(okr_data['turns'], 5)
        self.assertTrue(okr_data['is_cumulative'])

        # Check insights metrics
        insights_call = next(call for call in put_item_calls
                           if call.kwargs['Item']['type'] == 'insights')
        insights_data = insights_call.kwargs['Item']['data']
        self.assertEqual(insights_data['evaluations'], 2)
        self.assertEqual(insights_data['successes'], 1)
        self.assertEqual(insights_data['attempts'], 2)
        self.assertEqual(insights_data['turns'], 7)
        self.assertTrue(insights_data['is_cumulative'])

        # Check suggestion metrics
        suggestion_call = next(call for call in put_item_calls
                             if call.kwargs['Item']['type'] == 'suggestion')
        suggestion_data = suggestion_call.kwargs['Item']['data']
        self.assertEqual(suggestion_data['evaluations'], 1)
        self.assertEqual(suggestion_data['successes'], 1)
        self.assertEqual(suggestion_data['attempts'], 1)
        self.assertEqual(suggestion_data['turns'], 2)
        self.assertTrue(suggestion_data['is_cumulative'])

        # Check response
        self.assertEqual(result['statusCode'], 200)
        self.assertIn('Daily metrics aggregation completed', result['body'])

if __name__ == '__main__':
    unittest.main()
