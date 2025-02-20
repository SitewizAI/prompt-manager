import streamlit as st
import boto3
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Prompt Manager", layout="wide")
st.title("Prompt Manager")

def get_prompts() -> List[Dict[str, Any]]:
    """Fetch all prompts from DynamoDB PromptsTable."""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('PromptsTable')
        response = table.scan()
        return response.get('Items', [])
    except Exception as e:
        print(f"Error getting prompts: {e}")
        return []

def get_recent_evaluations(stream_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch the most recent evaluations for a given stream key.
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('EvaluationsTable')

        response = table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            },
            ScanIndexForward=False,  # Sort in descending order (most recent first)
            Limit=limit
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"Error getting evaluations: {e}")
        return []

# Load data
STREAM_KEY = "VPFnNcTxE78nD7fMcxfcmnKv2C5coD92vdcYBtdf"
prompts = get_prompts()
recent_evals = get_recent_evaluations(STREAM_KEY, 5)

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
        with st.expander(f"{prompt['ref']} (Version {prompt.get('version', 'N/A')})"):
            st.subheader("Content")
            st.text_area("", prompt["content"], height=200)

with tab2:
    # Display recent evaluations
    st.header("Recent Evaluations")
    for eval in recent_evals:
        timestamp = datetime.fromtimestamp(eval['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        with st.expander(f"Evaluation at {timestamp} - {eval.get('type', 'N/A')}"):
            st.metric("Success", eval.get('success', False))
            st.metric("Number of Turns", eval.get('num_turns', 0))

            if eval.get('failure_reasons'):
                st.subheader("Failure Reasons")
                for reason in eval['failure_reasons']:
                    st.error(reason)

            if eval.get('summary'):
                st.subheader("Summary")
                st.write(eval['summary'])

            if eval.get('question'):
                st.subheader("Question")
                st.write(eval['question'])
