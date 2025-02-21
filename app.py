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

def update_prompt(ref: str, version: str, content: str) -> bool:
    """Update a prompt in DynamoDB PromptsTable."""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('PromptsTable')

        response = table.update_item(
            Key={
                'ref': ref,
                'version': version
            },
            UpdateExpression='SET content = :content, updatedAt = :timestamp',
            ExpressionAttributeValues={
                ':content': content,
                ':timestamp': datetime.now().isoformat()
            },
            ReturnValues='UPDATED_NEW'
        )
        return True
    except Exception as e:
        print(f"Error updating prompt: {e}")
        return False

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

def get_stream_evaluations(stream_key: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Fetch recent evaluations for a specific stream key from DynamoDB EvaluationsTable.
    Returns the most recent evaluation and 5 evaluations before it.

    Args:
        stream_key: The stream key to fetch evaluations for
        limit: Maximum number of evaluations to return (default 6 to get current + 5 previous)
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

        evaluations = response.get('Items', [])
        evaluations.sort(key=lambda x: float(x.get('timestamp', 0)), reverse=True)
        return evaluations
    except Exception as e:
        print(f"Error getting evaluations for stream key {stream_key}: {e}")
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
            content = st.text_area("", prompt["content"], height=200, key=f"content_{prompt['ref']}_{prompt.get('version', 'N/A')}")
            if content != prompt["content"]:
                if st.button("Update", key=f"update_{prompt['ref']}_{prompt.get('version', 'N/A')}"):
                    if update_prompt(prompt['ref'], prompt.get('version', 'N/A'), content):
                        st.success("Prompt updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update prompt")

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

    # Stream key input with autofill from recent evaluations
    recent_evals = get_all_evaluations(limit_per_stream=1)  # Get most recent eval for each stream
    stream_keys = [eval['streamKey'] for eval in recent_evals]
    stream_key = st.selectbox("Select Stream Key", options=stream_keys) if stream_keys else st.text_input("Enter Stream Key")

    if stream_key:
        # Get all context data
        data = get_data(stream_key)
        evaluations = get_stream_evaluations(stream_key)  # Get most recent + 5 previous evaluations

        if data and evaluations:
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

            # Display evaluation history
            with st.expander("Evaluation History"):
                if evaluations:
                    current_eval = evaluations[0]  # Most recent evaluation
                    st.subheader("Current Evaluation")
                    st.write(f"Timestamp: {datetime.fromtimestamp(float(current_eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"Type: {current_eval.get('type', 'N/A')}")
                    st.write(f"Question: {current_eval.get('question', 'N/A')}")
                    st.write(f"Success: {current_eval.get('success', False)}")
                    st.write(f"Number of Turns: {current_eval.get('num_turns', 0)}")

                    if len(evaluations) > 1:
                        st.subheader("Previous 5 Evaluations")
                        for eval in evaluations[1:]:
                            with st.expander(f"Evaluation from {datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')}"):
                                st.write(f"Type: {eval.get('type', 'N/A')}")
                                st.write(f"Success: {eval.get('success', False)}")
                                st.write(f"Number of Turns: {eval.get('num_turns', 0)}")
                                if eval.get('failure_reasons'):
                                    st.write("Failure Reasons:")
                                    for reason in eval['failure_reasons']:
                                        st.error(reason)
                                if eval.get('summary'):
                                    st.write("Summary:")
                                    st.write(eval['summary'])

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

                Current Evaluation:
                - Type: {current_eval.get('type', 'N/A')}
                - Question: {current_eval.get('question', 'N/A')}
                - Success: {current_eval.get('success', False)}
                - Number of Turns: {current_eval.get('num_turns', 0)}
                - Summary: {current_eval.get('summary', 'N/A')}

                Previous Evaluations Summary:
                {' '.join(f"Evaluation {i+1}: Type={eval.get('type', 'N/A')}, Success={eval.get('success', False)}, Failure Reasons={eval.get('failure_reasons', [])}, Summary={eval.get('summary', 'N/A')}" for i, eval in enumerate(evaluations[1:]))}
                """

                # Get AI response using run_completion_with_fallback from utils
                try:
                    ai_response = run_completion_with_fallback(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes website optimization data and provides insights. Use the provided context to answer questions accurately."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                        ]
                    )

                    if ai_response:
                        # Add AI response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})

                        # Display AI response
                        with st.chat_message("assistant"):
                            st.markdown(ai_response)
                    else:
                        st.error("Failed to get AI response")
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")
        else:
            st.error("No data or evaluations found for the provided stream key")
