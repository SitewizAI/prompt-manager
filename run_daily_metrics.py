import sys
import json
from datetime import datetime, timedelta
from daily_metrics_lambda import aggregate_daily_metrics

def run_metrics_aggregation(days_back=1):
    """Run the daily metrics aggregation for a specific number of days back."""
    print(f"Running daily metrics aggregation for {days_back} days back...")
    
    # Create event object
    event = {
        'days_back': days_back
    }
    
    # Run the aggregation
    result = aggregate_daily_metrics(event, {})
    
    # Print result
    print(f"Aggregation complete. Status code: {result['statusCode']}")
    if result['statusCode'] == 200:
        body = json.loads(result['body'])
        print(f"Date processed: {body.get('date_processed')}")
        print(f"Metrics stored: {json.dumps(body.get('metrics_stored'), indent=2)}")
    else:
        print(f"Error: {result['body']}")
    
    return result

if __name__ == "__main__":
    # Default to 1 day back if no arguments
    days = 1
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print(f"Invalid days value: {sys.argv[1]}. Using default of 1.")
    
    print(f"Processing data for the past {days} days")
    
    # If days is greater than 1, process each day individually
    if days > 1:
        for i in range(1, days+1):
            print(f"\nProcessing day {i} of {days}...")
            run_metrics_aggregation(i)
    else:
        run_metrics_aggregation(days)
    
    print("\nDone!")
