import os
import requests
import weave
from dotenv import load_dotenv
import argparse
import ast
import json
import re
from typing import List, Dict, Any

load_dotenv()
client = weave.init("Agents")

def ensure_output_dir():
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_github_files(token, repo="SitewizAI/sitewiz", target_path="."):
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
            print(f"Processing {full_path}")
            if not full_path.startswith(target_path):
                continue

            if item["type"] == "file" and item["name"].endswith(".py"):
                python_files.append({
                    "path": full_path,
                    "download_url": item["download_url"]
                })
            elif item["type"] == "dir":
                python_files.extend(process_contents(item["path"]))

        return python_files

    return process_contents(path=target_path)

def extract_weave_ref(line):
    """Extract the weave reference name from a line of code."""
    match = re.search(r"weave\.ref\('([^']+)'\)", line)
    if match:
        return match.group(1)
    return None

def get_weave_content(ref_name):
    """Get the content of a weave reference."""
    try:
        content = weave.ref(ref_name).get().content
        return content
    except Exception as e:
        print(f"Error getting content for {ref_name}: {e}")
        return None

def analyze_file_content(content):
    weave_refs = []
    lines = content.split("\n")
    
    for i, line in enumerate(lines, 1):
        if "weave.ref" in line:
            ref_name = extract_weave_ref(line)
            if ref_name:
                ref_content = get_weave_content(ref_name)
                weave_refs.append({
                    "line": i,
                    "content": line.strip(),
                    "ref_name": ref_name,
                    "prompt_content": ref_content
                })
    
    return weave_refs

def extract_functions_and_imports(content):
    tree = ast.parse(content)
    extracted_code = ""

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            extracted_code += ast.unparse(node) + "\n\n"

    return extracted_code
    
def save_to_file(data, filename, output_dir):
    filepath = os.path.join(output_dir, filename)
    if isinstance(data, str):
        with open(filepath, "w") as f:
            f.write(data)
    else:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    print(f"Saved {filename}")

def get_recent_evals(num_traces = 5) -> List[Dict[str, Any]]:
    """Get the 5 most recent evaluations from weave."""
    try:
        # Fetch evaluation traces
        calls = client.get_calls(
            filter={"op_names": ["weave:///sitewiz/Agents/op/Evaluation.predict_and_score:*"]},
            sort_by=[{"field": "started_at", "direction": "desc"}],
        )

        # Number of traces to extract
        num_traces = min(10, len(calls))  # Ensure we don't exceed available traces

        # Function to clean conversation data
        def filterConversation(conversation):
            filtered_conversation = []
            for message in conversation:
                # Create new dict with only the fields we want to keep
                filtered_message = {
                    key: value for key, value in message.items()
                    if key not in ["models_usage", "_class_name", "_bases"]
                }
                filtered_conversation.append(filtered_message)
            return filtered_conversation

        # Extract input-output pairs
        traces = []
        for i in range(num_traces):
            try:
                call = calls[i]
                trace = {
                    "input": call.inputs.get("example", "N/A"),
                    "output": call.output.get("scores", {}).get("score", "N/A"),
                }
                if "conversation" in trace["output"]:
                    # trace["output"]["conversation"] = filterConversation(trace["output"].get("conversation", []))
                    trace["output"]["conversation"] = []
                trace["failure_reasons"] = trace["output"].get("failure_reasons", [])
                trace["type"] = trace["input"]["options"]["type"]
                trace["stream_key"] = trace["input"]["stream_key"]
                trace["attempts"] = trace["output"]["attempts"]
                trace["successes"] = trace["output"]["successes"]
                trace["num_turns"] = trace["output"]["num_turns"]
                traces.append(trace)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
        return traces
    except Exception as e:
        print(f"Error getting evaluations: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Analyze GitHub repository for weave.ref commands and prompts.")
    parser.add_argument("--target_path", type=str, default="backend/agents/data_analyst_group", 
                       help="Path to analyze within the repository.")
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable not set")

    output_dir = ensure_output_dir()
    
    
    # Get recent evaluations
    print("Fetching recent evaluations...")
    recent_evals = get_recent_evals(5)
    print(f"Found {len(recent_evals)} recent evaluations")
    
    print("Fetching Python files from repository...")
    python_files = get_github_files(token, target_path=args.target_path)
    print(f"Found {len(python_files)} Python files")

    all_extracted_code = ""
    all_weave_refs = []

    for file_info in python_files:
        file_path = file_info["path"]
        download_url = file_info["download_url"]
        print(f"\nAnalyzing {file_path}...")

        response = requests.get(download_url)
        if response.status_code != 200:
            print(f"Error downloading {file_path}")
            continue

        content = response.text
        
        # Extract code
        extracted_code = extract_functions_and_imports(content)
        all_extracted_code += f"\n\n# File: {file_path}\n\n{extracted_code}"
        
        # Extract weave refs and their content
        weave_refs = analyze_file_content(content)
        if weave_refs:
            all_weave_refs.extend([{**ref, "file": file_path} for ref in weave_refs])

    # Save all outputs
    save_to_file(all_extracted_code, "extracted_code.txt", output_dir)
    save_to_file(all_weave_refs, "weave_refs.json", output_dir)
    save_to_file(recent_evals, "recent_evals.json", output_dir)

if __name__ == "__main__":
    main()
