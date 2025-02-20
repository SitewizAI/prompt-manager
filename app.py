import streamlit as st
import json
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime

load_dotenv()

st.set_page_config(page_title="Prompt Manager", layout="wide")
st.title("Prompt Manager")

def get_prompt_content(ref_name: str) -> Dict[str, Any]:
    """Fetch prompt content from DynamoDB."""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('PromptsTable')
        response = table.get_item(Key={'ref': ref_name})
        return response.get('Item', {})
    except Exception as e:
        print(f"Error getting content for {ref_name}: {e}")
        return {}

def get_recent_evals(stream_key: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch recent evaluations from DynamoDB."""
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

        evaluations = []
        for item in response.get('Items', []):
            eval_data = {
                'type': item.get('type', 'N/A'),
                'stream_key': item.get('streamKey', 'N/A'),
                'failure_reasons': item.get('failure_reasons', []),
                'num_turns': item.get('num_turns', 0),
                'success': item.get('success', False),
                'timestamp': item.get('timestamp'),
                'question': item.get('question', 'N/A')
            }
            evaluations.append(eval_data)
        return evaluations
    except Exception as e:
        print(f"Error getting evaluations: {e}")
        return []

def load_json_file(filename):
    filepath = Path("output") / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return []

# Load data
weave_refs = load_json_file("weave_refs.json")
stream_key = "VPFnNcTxE78nD7fMcxfcmnKv2C5coD92vdcYBtdf"  # Example stream key
recent_evals = get_recent_evals(stream_key)

# Tabs for different views
tab1, tab2 = st.tabs(["Prompts", "Recent Evaluations"])

with tab1:
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_files = st.sidebar.multiselect(
        "Filter by files",
        options=list(set([w["file"] for w in weave_refs])),
    )

    # Search box
    search_term = st.sidebar.text_input("Search content").lower()

    # Filter data based on selections
    if selected_files:
        weave_refs = [w for w in weave_refs if w["file"] in selected_files]

    if search_term:
        weave_refs = [w for w in weave_refs if search_term in w["content"].lower()]

    # Display prompts
    st.header("Prompts")
    for ref in weave_refs:
        prompt_data = get_prompt_content(ref["ref_name"])
        with st.expander(f"{ref['file']} - {ref['ref_name']} (Line {ref['line']})"):
            st.subheader("Reference")
            st.code(ref["content"])

            st.subheader("Prompt Content")
            if prompt_data:
                st.text_area("Content", prompt_data.get('content', ''), height=200)
                st.text(f"Version: {prompt_data.get('version', 'N/A')}")
            else:
                st.error("Failed to fetch prompt content")

with tab2:
    # Display recent evaluations
    st.header("Recent Evaluations")
    for eval in recent_evals:
        timestamp = datetime.fromtimestamp(eval['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        with st.expander(f"Evaluation - {eval['type']} at {timestamp}"):
            st.text(f"Question: {eval['question']}")
            st.metric("Success", "Yes" if eval['success'] else "No")
            st.metric("Number of Turns", eval['num_turns'])

            if eval['failure_reasons']:
                st.subheader("Failure Reasons")
                for reason in eval['failure_reasons']:
                    st.error(reason)
