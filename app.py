import streamlit as st
import boto3
import json
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

# Import litellm and utils
import litellm
from utils import get_data, get_prompt_from_dynamodb

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Prompts", "Recent Evaluations", "Chat Assistant"])

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
        timestamp = datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
        with st.expander(f"Evaluation - {eval.get('type', 'N/A')} ({eval['streamKey']}) - {timestamp}"):
            st.write(f"Question: {eval.get('question', 'N/A')}")
            st.metric("Success", eval.get('success', False))
            st.metric("Number of Turns", float(eval.get('num_turns', 0)))

            if eval.get('failure_reasons'):
                st.subheader("Failure Reasons")
                for reason in eval['failure_reasons']:
                    st.error(reason)

            if eval.get('summary'):
                st.subheader("Summary")
                st.write(eval['summary'])

with tab3:
    st.header("Chat Assistant")

    # Stream key input
    stream_key = st.text_input("Enter Stream Key")

    if stream_key:
        # Get all context data
        data = get_data(stream_key)

        if data:
            # Display current data sections
            with st.expander("Current Context Data"):
                if data.get("okrs"):
                    st.subheader("OKRs")
                    for okr in data["okrs"]:
                        st.markdown(okr["markdown"])

                if data.get("insights"):
                    st.subheader("Insights")
                    for insight in data["insights"]:
                        st.markdown(insight["markdown"])

                if data.get("suggestions"):
                    st.subheader("Suggestions")
                    for suggestion in data["suggestions"]:
                        st.markdown(suggestion["markdown"])

                if data.get("code"):
                    st.subheader("Code Suggestions")
                    for code_suggestion in data["code"]:
                        st.markdown(code_suggestion["markdown"])

            # Chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about the data..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Prepare context for the AI
                context = f"""
                Current OKRs:
                {' '.join(okr['markdown'] for okr in data['okrs'])}

                Recent Insights:
                {' '.join(insight['markdown'] for insight in data['insights'])}

                Recent Suggestions:
                {' '.join(suggestion['markdown'] for suggestion in data['suggestions'])}

                Code Suggestions:
                {' '.join(code['markdown'] for code in data['code'])}
                """

                # Get AI response using litellm
                try:
                    # First get normal chat response
                    response = litellm.completion(
                        model="litellm_proxy/gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes website optimization data and provides insights. Use the provided context to answer questions accurately."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                        ]
                    )

                    ai_response = response.choices[0].message.content

                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

                    # Display AI response
                    with st.chat_message("assistant"):
                        st.markdown(ai_response)

                    # Also analyze if this should be a GitHub issue
                    issue_response = litellm.completion(
                        model="litellm_proxy/gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an AI that analyzes user questions and determines if they should be GitHub issues. If yes, provide a structured response with issue details."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnalyze if this should be a GitHub issue and if so, provide details in JSON format with title, description, labels, and priority fields."}
                        ],
                        response_format={
                            "type": "object",
                            "properties": {
                                "should_create_issue": {"type": "boolean"},
                                "issue": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "labels": {"type": "array", "items": {"type": "string"}},
                                        "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                                    }
                                }
                            }
                        }
                    )

                    issue_data = json.loads(issue_response.choices[0].message.content)

                    if issue_data.get("should_create_issue"):
                        with st.chat_message("assistant"):
                            st.info("This question has been identified as a potential GitHub issue:")
                            st.json(issue_data["issue"])

                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")
        else:
            st.error("No data found for the provided stream key")
