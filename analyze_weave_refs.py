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


import boto3
import json
import os
import litellm
from litellm import completion
from litellm.utils import trim_messages
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# check if aws credentials are set
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')

model_fallback_list = ["video"]

def get_api_key(secret_name):
    region_name = "us-east-1"
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    return json.loads(get_secret_value_response["SecretString"])


def initialize_vertex_ai():
    """Initialize Vertex AI with service account credentials"""
    AI_KEYS = get_api_key("AI_KEYS")
    litellm.api_key = AI_KEYS["LLM_API_KEY"]
    litellm.api_base = "https://llms.sitewiz.ai"
    litellm.enable_json_schema_validation = True


def run_completion_with_fallback(messages=None, prompt=None, models=model_fallback_list, response_format=None):
    """
    Run completion with fallback to evaluate.
    """
    initialize_vertex_ai()

    if messages is None:
        if prompt is None:
            raise ValueError("Either messages or prompt should be provided.")
        else:
            messages = [{"role": "user", "content": prompt}]

    trimmed_messages = messages
    try:
        trimmed_messages = trim_messages(messages, model)
    except Exception as e:
        pass

    for model in models:
        try:
            if response_format is None:
                response = completion(model=model, messages=trimmed_messages)
                content = response.choices[0].message.content
                return content
            else:
                response = completion(model=model, messages=trimmed_messages, response_format=response_format)
                content = json.loads(response.choices[0].message.content)  
                if isinstance(response_format, BaseModel):
                    response_format.model_validate(content)

                return content
        except Exception as e:
            print(f"Failed to run completion with model {model}. Error: {str(e)}")
    return None



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
    
def save_to_s3(data, filename, bucket_name="sitewiz-prompts"):
    """Save data to S3 bucket."""
    try:
        s3_client = boto3.client('s3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )

        # Convert data to JSON string if it's not already a string
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, indent=2)

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=content,
            ContentType='application/json'
        )
        print(f"Saved {filename} to S3 bucket {bucket_name}")
    except Exception as e:
        print(f"Error saving to S3: {e}")
        # Fallback to local file save
        filepath = os.path.join("output", filename)
        if isinstance(data, str):
            with open(filepath, "w") as f:
                f.write(data)
        else:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        print(f"Saved {filename} locally as fallback")

def analyze_issue_with_llm(issue_content: str) -> Dict[str, Any]:
    """Analyze an issue using LLM to understand how to fix it."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that analyzes GitHub issues and provides structured solutions."},
        {"role": "user", "content": f"Please analyze this issue and provide a structured response with root cause and solution steps:\n\n{issue_content}"}
    ]

    response_format = {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string"},
            "solution_steps": {"type": "array", "items": {"type": "string"}},
            "estimated_effort": {"type": "string"}
        }
    }

    result = run_completion_with_fallback(
        messages=messages,
        models=["gpt-4"],
        response_format=response_format
    )

    return result

def create_github_issue(token: str, repo: str, title: str, body: str) -> Dict[str, Any]:
    """Create a new GitHub issue."""
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 201:
        raise Exception(f"Failed to create issue: {response.text}")

    return response.json()

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

    # Save all outputs to S3
    save_to_s3(all_extracted_code, "extracted_code.txt")
    save_to_s3(all_weave_refs, "weave_refs.json")
    save_to_s3(recent_evals, "recent_evals.json")

if __name__ == "__main__":
    main()
