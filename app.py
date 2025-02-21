import streamlit as st
import boto3
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Prompt Manager", layout="wide")
st.title("Prompt Manager")

def get_all_prompts() -> List[Dict[str, Any]]:
    """Fetch all prompts from DynamoDB PromptsTable."""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('PromptsTable')

        prompts = []
        response = table.scan()
        prompts.extend(response.get('Items', []))

        # Handle pagination if there are more items
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            prompts.extend(response.get('Items', []))

        return prompts
    except Exception as e:
        print(f"Error getting prompts: {e}")
        return []

def get_all_evaluations(limit_per_stream: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent evaluations for all stream keys from DynamoDB EvaluationsTable.

    Args:
        limit_per_stream: Maximum number of items to return per stream key
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('EvaluationsTable')

        # First get all unique stream keys
        response = table.scan(
            ProjectionExpression='streamKey',
        )
        stream_keys = {item['streamKey'] for item in response.get('Items', [])}

        # Handle pagination for stream keys
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ProjectionExpression='streamKey',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            stream_keys.update({item['streamKey'] for item in response.get('Items', [])})

        # For each stream key, get recent evaluations
        all_evaluations = []
        for stream_key in stream_keys:
            response = table.query(
                KeyConditionExpression='streamKey = :sk',
                ExpressionAttributeValues={
                    ':sk': stream_key
                },
                ScanIndexForward=False,  # Sort in descending order (most recent first)
                Limit=limit_per_stream
            )
            all_evaluations.extend(response.get('Items', []))

        return all_evaluations
    except Exception as e:
        print(f"Error getting evaluations: {e}")
        return []

# Load data
prompts = get_all_prompts()
recent_evals = get_all_evaluations()

# Tabs for different views
tab1, tab2 = st.tabs(["Prompts", "Recent Evaluations"])

with tab1:
    # Sidebar filters
    st.sidebar.header("Filters")

    # Get unique refs for filtering
    all_refs = list(set([p["ref"] for p in prompts]))
    selected_refs = st.sidebar.multiselect(
        "Filter by refs",
        options=all_refs,
    )

    # Search box
    search_term = st.sidebar.text_input("Search content").lower()

    # Filter data based on selections
    filtered_prompts = prompts
    if selected_refs:
        filtered_prompts = [p for p in filtered_prompts if p["ref"] in selected_refs]

    if search_term:
        filtered_prompts = [p for p in filtered_prompts if (
            search_term in p["content"].lower()
        )]

    # Display prompts
    st.header("Prompts")
    for prompt in filtered_prompts:
        with st.expander(f"{prompt['ref']} (Version: {prompt.get('version', 'N/A')})"):
            st.subheader("Content")
            st.text_area("", prompt["content"], height=200)

with tab2:
    # Display recent evaluations
    st.header("Recent Evaluations")
    for eval in recent_evals:
        timestamp = datetime.fromtimestamp(eval['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        with st.expander(f"Evaluation - {eval.get('type', 'N/A')} ({eval['streamKey']}) - {timestamp}"):
            st.write(f"Question: {eval.get('question', 'N/A')}")
            st.metric("Success", eval.get('success', False))
            st.metric("Number of Turns", eval.get('num_turns', 0))

            if eval.get('failure_reasons'):
                st.subheader("Failure Reasons")
                for reason in eval['failure_reasons']:
                    st.error(reason)

            if eval.get('summary'):
                st.subheader("Summary")
                st.write(eval['summary'])
