import boto3
import json
from datetime import datetime, timezone
from decimal import Decimal
import boto3
import json
import os
import litellm
from litellm import completion
from litellm.utils import trim_messages
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, List
import requests

load_dotenv()

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

# check if aws credentials are set
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')

model_fallback_list = ["video", "main"]

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


SYSTEM_PROMPT = """You are a helpful website optimization expert assistant assisting in creating an agentic workflow that automates digital experience optimization – from data analysis to insight/suggestion generation to code implementation. Your role is to analyze evaluations and provide recommendations to update the prompts and code files, thereby improving the quality and accuracy of outputs so that each evaluation is successful in a low number of turns. Use the provided context to generate specific, accurate, and traceable recommendations that update the code and prompt structure.

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


def run_completion_with_fallback(messages=None, prompt=None, models=model_fallback_list, response_format=None, temperature=None):
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
                response = completion(model="litellm_proxy/"+model, messages=trimmed_messages, temperature=temperature)
                content = response.choices[0].message.content
                return content
            else:
                
                response = completion(model="litellm_proxy/"+model, messages=trimmed_messages, response_format=response_format, temperature=temperature)
                content = json.loads(response.choices[0].message.content)  
                if isinstance(response_format, BaseModel):
                    response_format.model_validate(content)

                return content
        except Exception as e:
            print(f"Failed to run completion with model {model}. Error: {str(e)}")
    return None



def get_dynamodb_table(table_name: str):
    """Get DynamoDB table resource."""
    dynamodb = boto3.resource('dynamodb')
    return dynamodb.Table(table_name)

def get_data(stream_key: str) -> Dict[str, Any]:
    """
    Get OKRs, insights and suggestions with markdown representations.
    """
    try:
        # Use resource tables
        okr_table = get_dynamodb_table('website-okrs')
        insight_table = get_dynamodb_table('website-insights')
        suggestion_table = get_dynamodb_table('WebsiteReports')

        # Get all OKRs for the stream key
        okr_response = okr_table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            }
        )
        okrs = okr_response.get('Items', [])

        # Get insights
        insight_response = insight_table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            }
        )
        insights = insight_response.get('Items', [])

        # Get suggestions
        suggestion_response = suggestion_table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            }
        )
        suggestions = suggestion_response.get('Items', [])

        # Process data
        processed_data = {
            "okrs": [],
            "insights": [],
            "suggestions": [],
            "code": []
        }

        # Process OKRs
        for okr in okrs:
            processed_data["okrs"].append({
                "markdown": okr_to_markdown(okr),
                "raw": okr
            })

        # Process insights
        for insight in insights:
            processed_data["insights"].append({
                "markdown": insight_to_markdown(insight),
                "raw": insight
            })

        # Process suggestions
        for suggestion in suggestions:
            suggestion_record = {
                "markdown": suggestion_to_markdown(suggestion),
                "raw": suggestion
            }
            processed_data["suggestions"].append(suggestion_record)

            # Add to code list if it includes a Code field
            if suggestion.get('Code'):
                processed_data["code"].append(suggestion_record)

        return processed_data
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def okr_to_markdown(okr: dict) -> str:
    """Convert an OKR to markdown format."""
    markdown = "# OKR Analysis\n\n"

    # Add name and description
    markdown += f"## Name\n{okr.get('name', '')}\n\n"
    markdown += f"## Description\n{okr.get('description', '')}\n\n"

    # Add timestamp if available
    if 'timestamp' in okr:
        timestamp_int = int(okr.get('timestamp', 0))
        markdown += f"## Last Updated\n{datetime.fromtimestamp(timestamp_int/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add metrics output if available
    if 'output' in okr:
        try:
            output_dict = eval(okr['output'])
            markdown += "## Metrics\n"
            markdown += f"- Metric Name: {output_dict.get('Metric', 'N/A')}\n"
            markdown += f"- Description: {output_dict.get('Description', 'N/A')}\n"
            markdown += f"- Date Range: {output_dict.get('start_date', 'N/A')} to {output_dict.get('end_date', 'N/A')}\n"
            if 'values' in output_dict:
                markdown += "- Values:\n"
                for date, value in output_dict['values']:
                    markdown += f"  - {date}: {value}\n"
        except:
            markdown += f"## Raw Output\n{okr.get('output', 'N/A')}\n"

    # Add reach value if available
    if 'reach_value' in okr:
        markdown += f"\n## Reach\n{okr.get('reach_value', 'N/A')}\n"

    return markdown

def insight_to_markdown(insight: dict) -> str:
    """Convert an insight to markdown format."""
    try:
        markdown = "# Insight Analysis\n\n"

        # Add data statement
        markdown += f"## Data Statement\n{insight.get('data_statement', '')}\n\n"

        # Add other sections
        markdown += f"## Problem Statement\n{insight.get('problem_statement', '')}\n\n"
        markdown += f"## Business Objective\n{insight.get('business_objective', '')}\n\n"
        markdown += f"## Hypothesis\n{insight.get('hypothesis', '')}\n\n"

        # Add metrics
        markdown += "## Metrics\n"
        markdown += f"- Frequency: {insight.get('frequency', 'N/A')}\n"
        markdown += f"- Severity: {insight.get('severity', 'N/A')}\n"
        markdown += f"- Severity reasoning: {insight.get('severity_reasoning', 'N/A')}\n"
        markdown += f"- Confidence: {insight.get('confidence', 'N/A')}\n"
        markdown += f"- Confidence reasoning: {insight.get('confidence_reasoning', 'N/A')}\n"

        return markdown
    except Exception as e:
        print(f"Error converting insight to markdown: {e}")
        return f"Error processing insight. Raw data:\n{json.dumps(insight, indent=4)}"

def suggestion_to_markdown(suggestion: Dict[str, Any]) -> str:
    """Convert a suggestion to markdown format."""
    try:
        markdown = []

        # Add header
        if 'Shortened' in suggestion:
            for shortened in suggestion.get('Shortened', []):
                if shortened.get('type') == 'header':
                    markdown.append(f"## {shortened.get('text', '')}\n")

        # Add tags
        if 'Tags' in suggestion:
            markdown.append("## Tags")
            for tag in suggestion.get('Tags', []):
                markdown.append(f"- **{tag.get('type', '')}:** {tag.get('Value', '')} ({tag.get('Tooltip', '')})")

        # Add expanded content
        if 'Expanded' in suggestion:
            for expanded in suggestion.get('Expanded', []):
                if expanded.get('type') == 'text':
                    markdown.append(f"### {expanded.get('header', '')}\n")
                    markdown.append(expanded.get('text', ''))

        # Add insights
        if 'Insights' in suggestion:
            markdown.append("## Insights")
            for insight in suggestion.get('Insights', []):
                if 'data' in insight:
                    for data_point in insight.get('data', []):
                        if data_point.get('type') == 'Heatmap':
                            markdown.append(f"- **Heatmap (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                        elif data_point.get('type') == 'Session Recording':
                            markdown.append(f"- **Session Recording (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                        else:
                            markdown.append(f"- **{data_point.get('type')} (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                markdown.append(insight.get('text', ''))

        return "\n\n".join(markdown)
    except Exception as e:
        print(f"Error converting suggestion to markdown: {e}")
        return f"Error processing suggestion. Raw data:\n{json.dumps(suggestion, indent=4)}"

def get_prompt_from_dynamodb(ref: str) -> str:
    """Get prompt from DynamoDB PromptsTable by ref."""
    try:
        table = get_dynamodb_table('PromptsTable')
        response = table.get_item(Key={'ref': ref})
        return response['Item']['content']
    except Exception as e:
        print(f"Error getting prompt {ref} from DynamoDB: {e}")
        return ""


def get_github_files(token, repo="SitewizAI/sitewiz", target_path="backend/agents/data_analyst_group"):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    def get_contents(path=""):
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        if (response.status_code != 200):
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

def get_project_id(token: str, org_name: str = "SitewizAI", project_number: int = 21, project_name: str = "Evaluations") -> Optional[str]:
    """Get GitHub project ID using GraphQL API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v4+json"
    }
    
    query = """
    query($org: String!, $number: Int!) {
        organization(login: $org) {
            projectV2(number: $number) {
                id
                title
            }
        }
    }
    """
    
    variables = {
        "org": org_name,
        "number": project_number
    }
    
    try:
        response = requests.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": variables},
            headers=headers
        )
        response.raise_for_status()
        
        result = response.json()
        if 'errors' in result:
            print(f"GraphQL Error getting project ID: {result['errors']}")
            return None
        
        project_data = result.get('data', {}).get('organization', {}).get('projectV2', {})
        if project_data.get('title') == project_name:
            return project_data.get('id')
            
        print(f"Project with name '{project_name}' not found")
        return None
        
    except Exception as e:
        print(f"Error getting project ID: {str(e)}")
        return None

def get_github_project_issues(token: str, 
                            org_name: str = "SitewizAI", 
                            project_number: int = 21, 
                            project_name: str = "Evaluations") -> List[Dict[str, Any]]:
    """Get all issues from a specific GitHub project."""
    if not token:
        print("No GitHub token provided")
        return []

    # First get the project ID
    project_id = get_project_id(token, org_name, project_number, project_name)
    if not project_id:
        print("Could not get project ID")
        return []

    print(f"Found project ID: {project_id}")
        
    query = """
    query($project_id: ID!) {
        node(id: $project_id) {
            ... on ProjectV2 {
                title
                items(first: 100) {
                    nodes {
                        content {
                            ... on Issue {
                                number
                                title
                                body
                                createdAt
                                state
                                url
                                labels(first: 10) {
                                    nodes {
                                        name
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v4+json"
    }
    
    try:
        response = requests.post(
            'https://api.github.com/graphql',
            headers=headers,
            json={'query': query, 'variables': {'project_id': project_id}}
        )
        
        if response.status_code != 200:
            print(f"Error fetching project issues. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []
        
        data = response.json()
        
        # Debug response
        if 'errors' in data:
            print(f"GraphQL errors: {data['errors']}")
            return []
            
        if not data.get('data'):
            print(f"No data in response: {data}")
            return []
            
        if not data['data'].get('node'):
            print(f"No node in response data: {data['data']}")
            return []
            
        project = data['data']['node']
        if not project:
            print(f"Project not found with ID: {project_id}")
            return []
            
        items = project.get('items', {}).get('nodes', [])
        issues = []
        
        for item in items:
            if not item or not item.get('content'):
                continue
                
            content = item['content']
            if not isinstance(content, dict) or 'title' not in content:
                continue
                
            issue = {
                'number': content.get('number'),
                'title': content.get('title'),
                'body': content.get('body', ''),
                'createdAt': content.get('createdAt'),
                'state': content.get('state'),
                'url': content.get('url'),
                'labels': [
                    label['name'] 
                    for label in content.get('labels', {}).get('nodes', [])
                    if isinstance(label, dict) and 'name' in label
                ]
            }
            issues.append(issue)
            
        return issues
        
    except Exception as e:
        print(f"Error processing project issues: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

def get_context(
    stream_key: str, 
    current_eval_timestamp: Optional[float] = None,
    return_type: str = "string",
    include_github_issues: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Create context from evaluations, prompts, and files.
    
    Args:
        stream_key: The stream key to fetch evaluations for
        current_eval_timestamp: Optional timestamp for specific evaluation
        return_type: "string" or "dict" for return format
        include_github_issues: Whether to include recent GitHub issues
    
    Returns:
        Either a string with all context or a dictionary with separated sections
    """
    dynamodb = boto3.resource('dynamodb')
    
    # Get evaluations for the stream key
    evals_table = dynamodb.Table('EvaluationsTable')
    response = evals_table.query(
        KeyConditionExpression='streamKey = :sk',
        ExpressionAttributeValues={':sk': stream_key},
        ScanIndexForward=False,
        Limit=6
    )
    
    evaluations = sorted(response.get('Items', []), 
                        key=lambda x: float(x.get('timestamp', 0)), 
                        reverse=True)
    
    if not evaluations:
        raise ValueError(f"No evaluations found for stream key: {stream_key}")
    
    # If no timestamp provided, use most recent evaluation
    if current_eval_timestamp is None:
        current_eval = evaluations[0]
    else:
        current_eval = next(
            (e for e in evaluations if float(e['timestamp']) == current_eval_timestamp),
            evaluations[0]
        )
    
    # Get previous evaluations before current one
    prev_evals = [
        e for e in evaluations 
        if float(e['timestamp']) < float(current_eval['timestamp'])
    ][:5]
    
    # Get prompts
    prompts_table = dynamodb.Table('PromptsTable')
    prompts = prompts_table.scan().get('Items', [])
    
    # Get data for the stream key
    data = get_data(stream_key)
    
    # Get Python files
    github_token = os.getenv('GITHUB_TOKEN')
    file_contents = []
    python_files = []
    if github_token:
        python_files = get_github_files(github_token)
        file_contents = [
            {"file": file, "content": get_file_contents(file)}
            for file in python_files
        ]
    
    # Get GitHub issues if requested
    github_issues = []
    if include_github_issues and github_token:
        github_issues = get_github_project_issues(github_token)[:5]
    
    # Prepare context data
    context_data = {
        "current_eval": {
            "timestamp": datetime.fromtimestamp(float(current_eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            "type": current_eval.get('type', 'N/A'),
            "successes": current_eval.get('successes', 0),
            "attempts": current_eval.get('attempts', 0),
            "failure_reasons": current_eval.get('failure_reasons', []),
            "conversation": current_eval.get('conversation', ''),
            "raw": current_eval
        },
        "prev_evals": [{
            "timestamp": datetime.fromtimestamp(float(e['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            "type": e.get('type', 'N/A'),
            "successes": e.get('successes', 0),
            "attempts": e.get('attempts', 0),
            "failure_reasons": e.get('failure_reasons', []),
            "summary": e.get('summary', 'N/A'),
            "raw": e
        } for e in prev_evals],
        "prompts": prompts,
        "data": data,
        "files": file_contents
    }
    
    if include_github_issues:
        context_data["github_issues"] = github_issues
    
    if return_type == "dict":
        return context_data
        
    # Build context string
    context_str = f"""
Current Evaluation:
Timestamp: {context_data['current_eval']['timestamp']}
Type: {context_data['current_eval']['type']}
Successes: {context_data['current_eval']['successes']}
Attempts: {context_data['current_eval']['attempts']}
Failure Reasons: {context_data['current_eval']['failure_reasons']}
Conversation History:
{context_data['current_eval']['conversation']}

Previous Evaluations:
{' '.join(f'''
Evaluation from {e['timestamp']}:
- Type: {e['type']}
- Successes: {e['successes']}
- Attempts: {e['attempts']}
- Failure Reasons: {e['failure_reasons']}
- Summary: {e['summary']}
''' for e in context_data['prev_evals'])}

Current Prompts:
{' '.join(f'''
Prompt {p['ref']} (Version {p.get('version', 'N/A')}):
{p['content']}
''' for p in prompts)}

Current Data:
OKRs:
{' '.join(okr['markdown'] for okr in data.get('okrs', []))}

Insights:
{' '.join(insight['markdown'] for insight in data.get('insights', []))}

Suggestions:
{' '.join(suggestion['markdown'] for suggestion in data.get('suggestions', []))}

Python Files Content:
{' '.join(f'''
File {file['file']['path']}:
{file['content']}
''' for file in file_contents)}
"""

    if include_github_issues:
        context_str += f"""
Recent GitHub Issues:
{' '.join(f'''
#{issue['number']}: {issue['title']}
{issue['body'][:200]}...
''' for issue in github_issues)}
"""

    return context_str

# context = get_context("VPFnNcTxE78nD7fMcxfcmnKv2C5coD92vdcYBtdf", include_github_issues=True)
# print(context)

def get_most_recent_stream_key() -> Optional[str]:
    """
    Get the stream key with the most recent evaluation across all types using GSI.
    Uses 'TimestampIndex' GSI with partition key 'type' and sort key 'timestamp'.
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('EvaluationsTable')
        
        # Define the types you want to check
        types = ['okr', 'insights', 'suggestion', 'code']  # Replace with your actual types
        
        most_recent_item = None
        
        for type_val in types:
            response = table.query(
                IndexName='TimestampIndex',
                KeyConditionExpression='#type = :type_val',
                ExpressionAttributeNames={
                    '#type': 'type'
                },
                ExpressionAttributeValues={
                    ':type_val': type_val
                },
                ScanIndexForward=False,  # Get in descending order
                Limit=1
            )
            
            items = response.get('Items', [])
            if items:
                item = items[0]
                if most_recent_item is None or float(item['timestamp']) > float(most_recent_item['timestamp']):
                    most_recent_item = item
        
        if most_recent_item:
            return most_recent_item['streamKey']
        return None
        
    except Exception as e:
        print(f"Error getting most recent stream key: {e}")
        return None

# print(get_most_recent_stream_key())

def create_github_issue_with_project(
    token: str,
    title: str,
    body: str,
    org: str = "SitewizAI",
    repo: str = "sitewiz",
    project_name: str = "Evaluations",
    project_number: int = 21,
    labels: List[str] = ["fix-me"]
) -> Dict[str, Any]:
    """
    Create a GitHub issue, add it to a project, and apply labels.
    
    Args:
        token: GitHub token
        title: Issue title
        body: Issue body
        org: GitHub organization name
        repo: Repository name
        project_name: Project name
        project_number: Project number
        labels: List of labels to apply
    
    Returns:
        Dict containing issue details including project status
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v4+json"
    }

    # First create the issue using GraphQL
    create_issue_query = """
    mutation($repo_id: ID!, $title: String!, $body: String!, $labels: [String!]) {
        createIssue(input: {
            repositoryId: $repo_id,
            title: $title,
            body: $body,
            labelIds: $labels
        }) {
            issue {
                id
                number
                url
            }
        }
    }
    """

    # Get repository ID
    repo_query = """
    query($org: String!, $repo: String!) {
        repository(owner: $org, name: $repo) {
            id
        }
    }
    """

    try:
        # Get repository ID
        repo_response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={
                "query": repo_query,
                "variables": {
                    "org": org,
                    "repo": repo
                }
            }
        )
        repo_data = repo_response.json()
        if "errors" in repo_data:
            raise Exception(f"Error getting repo ID: {repo_data['errors']}")
        
        repo_id = repo_data["data"]["repository"]["id"]

        # Create the issue
        issue_response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={
                "query": create_issue_query,
                "variables": {
                    "repo_id": repo_id,
                    "title": title,
                    "body": body,
                    "labels": labels
                }
            }
        )
        issue_data = issue_response.json()
        if "errors" in issue_data:
            raise Exception(f"Error creating issue: {issue_data['errors']}")

        issue = issue_data["data"]["createIssue"]["issue"]
        
        # Get project ID
        project_id = get_project_id(token, org, project_number, project_name)
        if not project_id:
            raise Exception("Could not find project")

        # Add issue to project
        add_to_project_query = """
        mutation($project_id: ID!, $content_id: ID!) {
            addProjectV2ItemById(input: {
                projectId: $project_id,
                contentId: $content_id
            }) {
                item {
                    id
                }
            }
        }
        """

        project_response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={
                "query": add_to_project_query,
                "variables": {
                    "project_id": project_id,
                    "content_id": issue["id"]
                }
            }
        )
        project_data = project_response.json()
        if "errors" in project_data:
            print(f"Warning: Error adding to project: {project_data['errors']}")
            return {
                "success": True,
                "issue": issue,
                "project_added": False,
                "error": str(project_data['errors'])
            }

        return {
            "success": True,
            "issue": issue,
            "project_added": True
        }

    except Exception as e:
        print(f"Error creating issue: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    issue = create_github_issue_with_project(
        token=token,
        title="Test Issue",
        body="This is a test issue",
        labels=["bug"]
    )