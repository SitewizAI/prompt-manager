import boto3
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

def get_dynamodb_table(table_name: str):
    """Get DynamoDB table resource."""
    dynamodb = boto3.resource('dynamodb')
    return dynamodb.Table(table_name)

def get_data(stream_key: str) -> Dict[str, Any]:
    """
    Get OKRs, insights and suggestions with markdown representations.
    """
    try:
        # Use resource tables
        okr_table = get_dynamodb_table('website-okrs')
        insight_table = get_dynamodb_table('website-insights')
        suggestion_table = get_dynamodb_table('WebsiteReports')

        # Get all OKRs for the stream key
        okr_response = okr_table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            }
        )
        okrs = okr_response.get('Items', [])

        # Get insights
        insight_response = insight_table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            }
        )
        insights = insight_response.get('Items', [])

        # Get suggestions
        suggestion_response = suggestion_table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            }
        )
        suggestions = suggestion_response.get('Items', [])

        # Process data
        processed_data = {
            "okrs": [],
            "insights": [],
            "suggestions": [],
            "code": []
        }

        # Process OKRs
        for okr in okrs:
            processed_data["okrs"].append({
                "markdown": okr_to_markdown(okr),
                "raw": okr
            })

        # Process insights
        for insight in insights:
            processed_data["insights"].append({
                "markdown": insight_to_markdown(insight),
                "raw": insight
            })

        # Process suggestions
        for suggestion in suggestions:
            suggestion_record = {
                "markdown": suggestion_to_markdown(suggestion),
                "raw": suggestion
            }
            processed_data["suggestions"].append(suggestion_record)

            # Add to code list if it includes a Code field
            if suggestion.get('Code'):
                processed_data["code"].append(suggestion_record)

        return processed_data
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def okr_to_markdown(okr: dict) -> str:
    """Convert an OKR to markdown format."""
    markdown = "# OKR Analysis\n\n"

    # Add name and description
    markdown += f"## Name\n{okr.get('name', '')}\n\n"
    markdown += f"## Description\n{okr.get('description', '')}\n\n"

    # Add timestamp if available
    if 'timestamp' in okr:
        timestamp_int = int(okr.get('timestamp', 0))
        markdown += f"## Last Updated\n{datetime.fromtimestamp(timestamp_int/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add metrics output if available
    if 'output' in okr:
        try:
            output_dict = eval(okr['output'])
            markdown += "## Metrics\n"
            markdown += f"- Metric Name: {output_dict.get('Metric', 'N/A')}\n"
            markdown += f"- Description: {output_dict.get('Description', 'N/A')}\n"
            markdown += f"- Date Range: {output_dict.get('start_date', 'N/A')} to {output_dict.get('end_date', 'N/A')}\n"
            if 'values' in output_dict:
                markdown += "- Values:\n"
                for date, value in output_dict['values']:
                    markdown += f"  - {date}: {value}\n"
        except:
            markdown += f"## Raw Output\n{okr.get('output', 'N/A')}\n"

    # Add reach value if available
    if 'reach_value' in okr:
        markdown += f"\n## Reach\n{okr.get('reach_value', 'N/A')}\n"

    return markdown

def insight_to_markdown(insight: dict) -> str:
    """Convert an insight to markdown format."""
    try:
        markdown = "# Insight Analysis\n\n"

        # Add data statement
        markdown += f"## Data Statement\n{insight.get('data_statement', '')}\n\n"

        # Add other sections
        markdown += f"## Problem Statement\n{insight.get('problem_statement', '')}\n\n"
        markdown += f"## Business Objective\n{insight.get('business_objective', '')}\n\n"
        markdown += f"## Hypothesis\n{insight.get('hypothesis', '')}\n\n"

        # Add metrics
        markdown += "## Metrics\n"
        markdown += f"- Frequency: {insight.get('frequency', 'N/A')}\n"
        markdown += f"- Severity: {insight.get('severity', 'N/A')}\n"
        markdown += f"- Severity reasoning: {insight.get('severity_reasoning', 'N/A')}\n"
        markdown += f"- Confidence: {insight.get('confidence', 'N/A')}\n"
        markdown += f"- Confidence reasoning: {insight.get('confidence_reasoning', 'N/A')}\n"

        return markdown
    except Exception as e:
        print(f"Error converting insight to markdown: {e}")
        return f"Error processing insight. Raw data:\n{json.dumps(insight, indent=4)}"

def suggestion_to_markdown(suggestion: Dict[str, Any]) -> str:
    """Convert a suggestion to markdown format."""
    try:
        markdown = []

        # Add header
        if 'Shortened' in suggestion:
            for shortened in suggestion.get('Shortened', []):
                if shortened.get('type') == 'header':
                    markdown.append(f"## {shortened.get('text', '')}\n")

        # Add tags
        if 'Tags' in suggestion:
            markdown.append("## Tags")
            for tag in suggestion.get('Tags', []):
                markdown.append(f"- **{tag.get('type', '')}:** {tag.get('Value', '')} ({tag.get('Tooltip', '')})")

        # Add expanded content
        if 'Expanded' in suggestion:
            for expanded in suggestion.get('Expanded', []):
                if expanded.get('type') == 'text':
                    markdown.append(f"### {expanded.get('header', '')}\n")
                    markdown.append(expanded.get('text', ''))

        # Add insights
        if 'Insights' in suggestion:
            markdown.append("## Insights")
            for insight in suggestion.get('Insights', []):
                if 'data' in insight:
                    for data_point in insight.get('data', []):
                        if data_point.get('type') == 'Heatmap':
                            markdown.append(f"- **Heatmap (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                        elif data_point.get('type') == 'Session Recording':
                            markdown.append(f"- **Session Recording (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                        else:
                            markdown.append(f"- **{data_point.get('type')} (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                markdown.append(insight.get('text', ''))

        return "\n\n".join(markdown)
    except Exception as e:
        print(f"Error converting suggestion to markdown: {e}")
        return f"Error processing suggestion. Raw data:\n{json.dumps(suggestion, indent=4)}"

def get_prompt_from_dynamodb(ref: str) -> str:
    """Get prompt from DynamoDB PromptsTable by ref."""
    try:
        table = get_dynamodb_table('PromptsTable')
        response = table.get_item(Key={'ref': ref})
        return response['Item']['content']
    except Exception as e:
        print(f"Error getting prompt {ref} from DynamoDB: {e}")
        return ""
