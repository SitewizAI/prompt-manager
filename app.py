import streamlit as st
import boto3
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from utils import run_completion_with_fallback
import requests
import os
import json
import tiktoken
from decimal import Decimal

load_dotenv()

st.set_page_config(page_title="Prompt Manager", layout="wide")
st.title("Prompt Manager")

# Initialize session state for expanders
if "expanders_open" not in st.session_state:
    st.session_state.expanders_open = True

# Add close/open all button
if st.button("Close All" if st.session_state.expanders_open else "Open All"):
    st.session_state.expanders_open = not st.session_state.expanders_open
    st.rerun()

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

def get_github_files(token, repo="SitewizAI/sitewiz", target_path="backend/agents/data_analyst_group"):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    def get_contents(path=""):
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error accessing {path}: {response.status_code}")
            return []

        contents = response.json()
        if not isinstance(contents, list):
            contents = [contents]

        return contents

    def process_contents(path=""):
        contents = get_contents(path)
        python_files = []

        for item in contents:
            full_path = os.path.join(path, item["name"])
            if item["type"] == "file" and item["name"].endswith(".py"):
                python_files.append({
                    "path": full_path,
                    "download_url": item["download_url"]
                })
            elif item["type"] == "dir":
                python_files.extend(process_contents(item["path"]))

        return python_files

    return process_contents(path=target_path)

def get_file_contents(file_info):
    response = requests.get(file_info["download_url"])
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error downloading {file_info['path']}")
        return ""

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))

def convert_decimal(value):
    """Convert Decimal values to float/int for Streamlit metrics."""
    if isinstance(value, Decimal):
        return float(value)
    return value

# Load data
prompts = get_all_prompts()
recent_evals = get_all_evaluations()

# Fetch GitHub files
github_token = os.getenv("GITHUB_TOKEN")
if github_token:
    python_files = get_github_files(github_token)
    file_contents = [get_file_contents(file_info) for file_info in python_files]
else:
    file_contents = []

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
        with st.expander(f"{prompt['ref']} (Version: {prompt.get('version', 'N/A')})", expanded=st.session_state.expanders_open):
            st.subheader("Content")
            content = st.text_area("Prompt Content", prompt["content"], height=200, key=f"content_{prompt['ref']}_{prompt.get('version', 'N/A')}")
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
        with st.expander(f"Evaluation - {eval.get('type', 'N/A')} ({eval['streamKey']}) - {timestamp}", expanded=st.session_state.expanders_open):
            st.write(f"Question: {eval.get('question', 'N/A')}")
            st.metric("Successes", convert_decimal(eval.get('successes', 0)))
            st.metric("Attempts", convert_decimal(eval.get('attempts', 0)))
            st.metric("Number of Turns", convert_decimal(eval.get('num_turns', 0)))

            if eval.get('failure_reasons'):
                st.subheader("Failure Reasons")
                for reason in eval['failure_reasons']:
                    st.error(reason)

            if eval.get('summary'):
                st.subheader("Summary")
                st.write(eval['summary'])

with tab3:
    st.header("Chat Assistant")

    # Initialize chat messages state if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Stream key and evaluation timestamp selection
    recent_evals = get_all_evaluations(limit_per_stream=1)
    stream_keys = [eval['streamKey'] for eval in recent_evals]
    stream_key = st.selectbox("Select Stream Key", options=stream_keys) if stream_keys else st.text_input("Enter Stream Key")

    if stream_key:
        # Get evaluations for timestamp selection
        evaluations = get_stream_evaluations(stream_key)
        if evaluations:
            eval_options = {datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'): eval 
                          for eval in evaluations}
            selected_timestamp = st.selectbox(
                "Select Evaluation Timestamp",
                options=list(eval_options.keys()),
                format_func=lambda x: f"Evaluation from {x}"
            )
            current_eval = eval_options[selected_timestamp]

            # Get data and filtered prompts
            data = get_data(stream_key)
            filtered_prompts = [p for p in prompts if p["ref"] in selected_refs] if selected_refs else prompts

            # Get previous evaluations before context preparation
            prev_evals = [e for e in evaluations if float(e['timestamp']) < float(current_eval['timestamp'])][:5]

            # Display chat messages from session state
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about the data..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Prepare comprehensive context
                context = f"""
                Current Evaluation:
                Timestamp: {selected_timestamp}
                Type: {current_eval.get('type', 'N/A')}
                Successes: {current_eval.get('successes', 0)}
                Failure Reasons: {current_eval.get('failure_reasons', [])}
				Attempts: {current_eval.get('attempts', 0)}
                Conversation History:
                {current_eval.get('conversation', '')}

                Previous Evaluations:
                {' '.join(f'''
                Evaluation from {datetime.fromtimestamp(float(e['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')}:
                - Type: {e.get('type', 'N/A')}
                - Successes: {e.get('successes', 0)}
                - Failure Reasons: {e.get('failure_reasons', [])}
                - Attempts: {e.get('attempts', 0)}
                - Summary: {e.get('summary', 'N/A')}
                ''' for e in prev_evals)}

                Current Prompts:
                {' '.join(f'''
                Prompt {p['ref']}:
                {p['content']}
                ''' for p in filtered_prompts)}

                Current Data:
                OKRs:
                {' '.join(okr['markdown'] for okr in data.get('okrs', []))}

                Insights:
                {' '.join(insight['markdown'] for insight in data.get('insights', []))}

                Suggestions:
                {' '.join(suggestion['markdown'] for suggestion in data.get('suggestions', []))}

                Python Files Content:
                {' '.join(content for content in file_contents)}
                """

                # Count tokens
                token_count = count_tokens(context)
                st.write(f"Token count: {token_count}")
                system_prompt = """You are a helpful website optimization expert assistant assisting in creating an agentic workflow that automates digital experience optimization – from data analysis to insight/suggestion generation to code implementation. Your role is to analyze evaluations and provide recommendations to update the prompts and code files, thereby improving the quality and accuracy of outputs so that each evaluation is successful in a low number of turns. Use the provided context to generate specific, accurate, and traceable recommendations that update the code and prompt structure.

---------------------------------------------------------------------
Types of Suggestions to Provide:

1. Block-Level Prompt Optimization using MIPRO  
   - Techniques to Use:
     • Bootstrapped Demonstration Extraction: Analyze evaluation traces to identify 2–3 high-quality input/output demonstration examples that clarify task patterns.
     • Grounded Instruction Proposal: Create a concise context block that includes:
         - A brief dataset summary (key patterns or rules)
         - A short program summary (outline of processing steps)
         - The selected high-quality demonstration examples
         - A short history snippet of previously proposed instructions with evaluation scores  
       Use this context to generate a new, clear, and unambiguous instruction aligned with task requirements.
     • Simplified Surrogate Evaluation: Heuristically simulate mini-batch evaluation for candidate instructions. Assess each candidate’s clarity, specificity, and integration of demonstration examples; then provide a brief rationale and select the best candidate.
     
   - Prompt Formatting Requirements:
     • Current Instruction: Display the existing prompt exactly as given.
     • Proposed Optimized Instruction: Present the revised prompt incorporating the bootstrapped examples and grounded context in plain language.
     • Key Changes: List 3–5 bullet points summarizing the modifications (e.g., “Added explicit dataset summary”, “Included 2 demonstration examples”, “Specified task rules to reduce ambiguity”).
     • Evaluation Heuristic: Provide a one- to two-sentence explanation of how the new prompt is expected to improve performance (e.g., by enhancing clarity or reducing misinterpretation).

2. Evaluations Optimization (Improving Success Rate and Quality)
   - Techniques to Use:
     • Refine Evaluation Questions: Review and update the evaluation questions to ensure they precisely measure the desired outcomes (e.g., correctness, traceability, and clarity). Adjust confidence thresholds as needed to better differentiate between successful and unsuccessful outputs.
     • Actionable Feedback Generation: For each evaluation failure, generate specific, actionable feedback that identifies the issue (e.g., ambiguous instructions, missing context, or incorrect data integration) and provide concrete suggestions for improvement.
     • Enhanced Evaluation Data Integration: Modify the storing function to ensure that all relevant evaluation details (such as SQL query outputs, execution logs, error messages, and computed metrics) are captured in a structured and traceable manner.
     
   - Output Requirements:
     • Present an updated list of evaluation questions with any new or adjusted confidence thresholds.
     • List clear, bullet-pointed actionable feedback items for common evaluation failure scenarios.
     • Describe specific modifications made to the storing function to improve data traceability and completeness, highlighting how these changes help in extracting useful insights from evaluation outputs.

3. Workflow Topology Optimization (Improving Agent Interactions)
   - Focus on evaluating and refining the interactions between multiple agents (when applicable).
   - Propose adjustments to the sequence and arrangement of agent modules to reduce redundant computation and improve overall coordination.
   - Provide suggestions that clarify the orchestration process (e.g., by introducing parallel processing, debate mechanisms, or reflective feedback loops) that can lead to faster convergence and improved output quality.

4. General Optimizations
   - Scope: Offer recommendations related to:
     • Fixing bugs
     • Improving performance
     • Adding, removing, or updating tools/functions
     • Any other general improvements that enhance system robustness
   - Ensure that all recommendations are specific, actionable, and directly traceable to the provided evaluation data.

---------------------------------------------------------------------
Human Guidelines and Goals:

• Ensure the final output’s data is fully traceable to the database and that the data used is directly reflected in the output.
• The final markdown output must be fully human-readable, contextually coherent, and useful to the business.
• Present smaller, verifiable results with nonzero outputs before constructing more complex queries. The higher the quality of the data, the more segmented and detailed the output should be.
• Avoid using dummy data; the provided data must be used to generate insights.
• Each new OKR, Insight, and Suggestion must offer a novel idea distinct from previous generations.
• Insights should detail problems or opportunities with a high severity/frequency/risk score and include a clear hypothesis for action.
• Suggestions must integrate all available data points, presenting a convincing, well-justified, and impactful story with high reach, impact, and confidence.
• Code generation should implement suggestions in a manner that meets the expectations of a conversion rate optimizer.

---------------------------------------------------------------------
Instructions for Operation:

• Focus Area: When optimizing, limit your scope to the specific areas indicated for each type of suggestion.
   - For Block-Level Prompt Optimization, apply the MIPRO techniques to a single prompt block.
   - For Evaluations Optimization, focus on refining evaluation questions, generating actionable feedback, and enhancing data integration in the storing function.
   - For Workflow Topology and General Optimizations, provide recommendations as applicable based on the evaluation data.
• Clarity and Traceability: Ensure every modification is clearly traceable to the provided data and context.
• Output Format: Structure your final output in clear markdown with sections as specified for each type of suggestion, making it fully human-readable and actionable.

By following these guidelines, you will produce a refined set of recommendations and updated system designs that leverage bootstrapped demonstration extraction, grounded instruction proposal, simplified surrogate evaluation, and enhanced evaluation methodologies to drive improved performance in digital experience optimization.
"""

                try:
                    ai_response = run_completion_with_fallback(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            # Include chat history in the context
                            *st.session_state.messages[:-1],  # Previous messages
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
                        ]
                    )

                    if ai_response:
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        with st.chat_message("assistant"):
                            st.markdown(ai_response)
                    else:
                        st.error("Failed to get AI response")
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")

            # Display sections
            st.subheader("Current Evaluation Details")
            with st.expander("Conversation History", expanded=st.session_state.expanders_open):
                if 'conversation' in current_eval:
                    st.markdown(current_eval['conversation'])
                else:
                    st.write("No conversation history available")

            # Rest of the display sections remain unchanged
            st.subheader("Current Prompts")
            for prompt in filtered_prompts:
                with st.expander(f"Prompt: {prompt['ref']}", expanded=st.session_state.expanders_open):
                    st.code(prompt['content'])

            # Display previous evaluations
            st.subheader("Previous Evaluations")
            prev_evals = [e for e in evaluations if float(e['timestamp']) < float(current_eval['timestamp'])][:5]
            for eval in prev_evals:
                timestamp = datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
                with st.expander(f"Evaluation from {timestamp}", expanded=st.session_state.expanders_open):
                    st.write(f"Type: {eval.get('type', 'N/A')}")
                    st.write(f"Successes: {convert_decimal(eval.get('successes', 0))}")
                    st.write(f"Attempts: {convert_decimal(eval.get('attempts', 0))}")
                    if eval.get('failure_reasons'):
                        st.write("Failure Reasons:")
                        for reason in eval['failure_reasons']:
                            st.error(reason)
                    if eval.get('summary'):
                        st.write("Summary:")
                        st.write(eval['summary'])

            # Display Python files
            st.subheader("Python Files Content")
            for file_info, content in zip(python_files, file_contents):
                with st.expander(f"File: {file_info['path']}", expanded=st.session_state.expanders_open):
                    st.code(content, language='python')

        else:
            st.error("No evaluations found for the selected stream key")
