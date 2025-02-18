import streamlit as st
import json
import os
import weave
import boto3
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
client = weave.init("Agents")

st.set_page_config(page_title="Prompt Manager", layout="wide")
st.title("Prompt Manager")

def get_weave_content(ref_name):
        try:
                content = weave.ref(ref_name).get().content
                return content
        except Exception as e:
                print(f"Error getting content for {ref_name}: {e}")
                return None

def get_recent_evals(num_traces=5) -> List[Dict[str, Any]]:
        try:
                calls = client.get_calls(
                        filter={"op_names": ["weave:///sitewiz/Agents/op/Evaluation.predict_and_score:*"]},
                        sort_by=[{"field": "started_at", "direction": "desc"}],
                )
                num_traces = min(num_traces, len(calls))
                traces = []
                for i in range(num_traces):
                        try:
                                call = calls[i]
                                output = call.output.get("scores", {})
                                trace = {
                                        "failure_reasons": output.get("failure_reasons", []),
                                        "type": call.inputs.get("example", {}).get("options", {}).get("type", "N/A"),
                                        "stream_key": call.inputs.get("example", {}).get("stream_key", "N/A"),
                                        "attempts": output.get("attempts", 0),
                                        "successes": output.get("successes", 0),
                                        "num_turns": output.get("num_turns", 0)
                                }
                                traces.append(trace)
                        except Exception as e:
                                print(e)
                return traces
        except Exception as e:
                print(f"Error getting evaluations: {e}")
                return []

def load_from_s3(filename, bucket_name="sitewiz-prompts"):
        """Load data from S3 bucket."""
        try:
                s3_client = boto3.client('s3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                        region_name=os.getenv('AWS_REGION')
                )

                # Get object from S3
                response = s3_client.get_object(
                        Bucket=bucket_name,
                        Key=filename
                )
                content = response['Body'].read().decode('utf-8')

                # Parse JSON if the content is not a plain text file
                if not filename.endswith('.txt'):
                        return json.loads(content)
                return content
        except Exception as e:
                print(f"Error loading from S3: {e}")
                # Fallback to local file
                filepath = Path("output") / filename
                if filepath.exists():
                        with open(filepath) as f:
                                return json.load(f)
                return []

# Load data from S3
weave_refs = load_from_s3("weave_refs.json")
recent_evals = load_from_s3("recent_evals.json")


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
                weave_refs = [w for w in weave_refs if (
                        search_term in w["content"].lower() or
                        (w["prompt_content"] and search_term in w["prompt_content"].lower())
                )]

        # Display prompts
        st.header("Weave References and Prompts")
        for ref in weave_refs:
                with st.expander(f"{ref['file']} - {ref['ref_name']} (Line {ref['line']})"):
                        st.subheader("Weave Reference")
                        st.code(ref["content"])

                        st.subheader("Prompt Content")
                        if ref["prompt_content"]:
                                st.text_area("", ref["prompt_content"], height=200)
                        else:
                                st.error("Failed to fetch prompt content")

with tab2:
        # Display recent evaluations
        st.header("Recent Evaluations")
        for eval in recent_evals:
                with st.expander(f"Evaluation - {eval['type']} ({eval['stream_key']})"):
                        st.metric("Attempts", eval['attempts'])
                        st.metric("Successes", eval['successes'])
                        st.metric("Number of Turns", eval['num_turns'])

                        if eval['failure_reasons']:
                                st.subheader("Failure Reasons")
                                for reason in eval['failure_reasons']:
                                        st.error(reason)
