import streamlit as st
import json
import os
import weave
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
try:
    client = weave.init("Agents")
except:
    # For testing purposes
    client = None

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

def load_json_file(filename):
        filepath = Path("output") / filename
        if filepath.exists():
                with open(filepath) as f:
                        return json.load(f)
        return []

# Load data
weave_refs = load_json_file("weave_refs.json")
recent_evals = get_recent_evals(5)


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
