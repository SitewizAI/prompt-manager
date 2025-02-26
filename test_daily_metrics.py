import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
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

        # Mock query response with sample evaluations
        self.mock_table.query.return_value = {
            'Items': [
                {
                    'date': yesterday_date,
                    'timestamp': start_timestamp + 3600,
                    'data': {
                        'type': 'suggestion',
                        'turns': 5,
                        'success': True,
                        'attempts': 2
                    }
                },
                {
                    'date': yesterday_date,
                    'timestamp': start_timestamp + 7200,
                    'data': {
                        'type': 'suggestion',
                        'turns': 3,
                        'success': False,
                        'attempts': 1
                    }
                },
                {
                    'date': yesterday_date,
                    'timestamp': start_timestamp + 10800,
                    'data': {
                        'type': 'insights',
                        'turns': 4,
                        'success': True,
                        'attempts': 1
                    }
                }
            ]
        }

        # Call the function
        result = aggregate_daily_metrics({}, {})

        # Verify that the DynamoDB table was queried for each type
        expected_query_params = {
            'KeyConditionExpression': '#date = :date AND #timestamp BETWEEN :start AND :end',
            'ExpressionAttributeNames': {
                '#date': 'date',
                '#timestamp': 'timestamp'
            },
            'ExpressionAttributeValues': {
                ':date': yesterday_date,
                ':start': start_timestamp,
                ':end': end_timestamp
            }
        }

        # We expect 5 queries, one for each type (okr, insights, suggestion, code, design)
        self.assertEqual(self.mock_table.query.call_count, 5)
        for call_args in self.mock_table.query.call_args_list:
            self.assertEqual(call_args.kwargs, expected_query_params)

        # Verify that put_item was called with correct aggregated metrics for each type
        put_item_calls = self.mock_table.put_item.call_args_list
        self.assertEqual(len(put_item_calls), 2)  # One for suggestion, one for insights

        # Check suggestion metrics
        suggestion_call = next(call for call in put_item_calls
                             if call.kwargs['Item']['data']['type'] == 'suggestion')
        suggestion_data = suggestion_call.kwargs['Item']['data']
        self.assertEqual(suggestion_data['evaluations'], 2)
        self.assertEqual(suggestion_data['successes'], 1)
        self.assertEqual(suggestion_data['attempts'], 3)
        self.assertEqual(suggestion_data['turns'], 8)
        self.assertTrue(suggestion_data['is_cumulative'])

        # Check insights metrics
        insights_call = next(call for call in put_item_calls
                           if call.kwargs['Item']['data']['type'] == 'insights')
        insights_data = insights_call.kwargs['Item']['data']
        self.assertEqual(insights_data['evaluations'], 1)
        self.assertEqual(insights_data['successes'], 1)
        self.assertEqual(insights_data['attempts'], 1)
        self.assertEqual(insights_data['turns'], 4)
        self.assertTrue(insights_data['is_cumulative'])

        # Check response
        self.assertEqual(result['statusCode'], 200)
        self.assertIn('Daily metrics aggregation completed', result['body'])

if __name__ == '__main__':
    unittest.main()
