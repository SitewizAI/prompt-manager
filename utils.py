import boto3
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import boto3
import json
import os
import litellm
from litellm import completion
from litellm.utils import trim_messages
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, List, Tuple, ClassVar
from dataclasses import dataclass, field
from collections import defaultdict
import requests
from botocore.exceptions import ClientError
import time
from functools import wraps
from decimal import Decimal
from boto3.dynamodb.conditions import Key, Attr
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

@dataclass
class ToolMessageTracker:
    """Track messages and responses from tool calls."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    _instance: ClassVar[Optional['ToolMessageTracker']] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.messages = []
        return cls._instance

    def add_message(self, tool_name: str, input_msg: str, response: str, error: Optional[str] = None):
        """Add a tool message with its response."""
        self.messages.append({
            "tool": tool_name,
            "input": input_msg,
            "response": response,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_context(self) -> str:
        """Get tool interaction context as string."""
        if not self.messages:
            return ""
        
        context = "\nTool Interaction History:\n"
        for msg in self.messages:
            context += f"\nTool: {msg['tool']}\n"
            context += f"Input: {msg['input']}\n"
            context += f"Response: {msg['response']}\n"
            if msg['error']:
                context += f"Error: {msg['error']}\n"
            context += f"Time: {msg['timestamp']}\n"
            context += "-" * 40 + "\n"
        return context

    def clear(self):
        """Clear message history."""
        self.messages = []

def log_debug(message: str):
    """Debug logging with tracking."""
    print(f"DEBUG: {message}")
    ToolMessageTracker().add_message(
        tool_name="debug",
        input_msg="",
        response=message
    )

def log_error(message: str, error: Exception = None):
    """Error logging with tracking."""
    error_msg = f"ERROR: {message}"
    if error:
        error_msg += f" - {str(error)}"
    print(error_msg)
    ToolMessageTracker().add_message(
        tool_name="error",
        input_msg=message,
        response="",
        error=str(error) if error else None
    )

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

# check if aws credentials are set
aws_region = os.getenv('AWS_REGION') or "us-east-1"
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')

model_fallback_list = ["video"]

# Function to get boto3 resource with credentials
def get_boto3_resource(service_name='dynamodb'):
    """Get DynamoDB table resource with debug logging."""
    log_debug(f"Creating boto3 resource for {service_name}")
    try:
        resource = boto3.resource(
            service_name,
            region_name=aws_region
        )
        log_debug(f"Successfully created {service_name} resource")
        return resource
    except Exception as e:
        log_error(f"Failed to create {service_name} resource", e)
        raise

# Function to get boto3 client with credentials
def get_boto3_client(service_name, region=None):
    return boto3.client(
        service_name,
        region_name=aws_region
    )

def get_api_key(secret_name):
    client = get_boto3_client('secretsmanager', region="us-east-1")
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


SYSTEM_PROMPT = """You are a helpful website optimization expert assistant assisting in creating an agentic workflow that automates digital experience optimization – from data analysis to insight/suggestion generation to code implementation. 
Your role is to analyze evaluations and provide recommendations to update the prompts and code files, thereby improving the quality and accuracy of outputs so that each evaluation is successful in a low number of turns. 
Use the provided context to generate specific, accurate, and traceable recommendations that update the code and prompt structure.

---------------------------------------------------------------------
Types of Suggestions to Provide:

1. Block-Level Prompt Optimization for Reasoning models (all agents use reasoning models)  
   - Techniques to Use:
     • Bootstrapped Demonstration Extraction: Analyze evaluation traces to identify 2–3 high-quality input/output demonstration examples and formatting that clarify task patterns.
     • Ensure your prompts are straightforward and easy to understand. Avoid ambiguity by specifying exactly what you need from the AI
     • Include specific details, constraints, and objectives to guide the model toward the desired output using domain specific knowledge of digital experience optimization and the agent role
     • Structure complex inputs with clear sections or headings
     • Specify end goal and desired output format explicitly
     
   - Prompt Formatting Requirements:
    • The variable substitions should use single brackets, {variable_name}, and the substitution variables must be the ones provided in the code in the .format() method
    • For python variables in prompts with python code, ensure that double brackets are used (eg {{ and }}) since we are using python multilined strings for the prompts, especially in example queries since the brackets must be escaped for the prompt to compile, unless we are making an allowed substitution specified in the code

   - Tool Usage Requirements:
    • When updating agent prompts, ONLY reference tools that are actually available to that agent in create_group_chat.py
    • Check which tools are provided to each agent type and ensure your prompt only mentions those specific tools
    • Never include instructions for using tools that aren't explicitly assigned to the agent in create_group_chat.py
    • If an agent needs access to data that requires a tool it doesn't have, suggest adding that tool to the agent in create_group_chat.py rather than mentioning unavailable tools in the prompt

   - Note that all agent instructions are independent
    • Instruction updates should only apply to the agent in question, don't put instructions for other agents in the system message for the agent
    
2. Evaluations Optimization (Improving Success Rate and Quality)
   - Techniques to Use:
     • Refine Evaluation Questions: Review and update the evaluation questions to ensure they precisely measure the desired outcomes (e.g., correctness, traceability, and clarity). Adjust confidence thresholds as needed to better differentiate between successful and unsuccessful outputs. Note we need > 50% success rate in evaluations.
     • Actionable Feedback Generation: For each evaluation failure, generate specific, actionable feedback that identifies the issue (e.g., ambiguous instructions, missing context, or incorrect data integration) and provide concrete suggestions for improvement.
     • Enhanced Evaluation Data Integration: Modify the storing function to ensure that all relevant evaluation details (such as SQL query outputs, execution logs, error messages, and computed metrics) are captured in a structured and traceable manner.
   - Important notes
     • Ensure you know the inputs and their format and that those inputs are used properly in the evaluation questions. Evaluation questions cannot use output or reference variables not provided in the input.
   - Output Requirements:
     • Present an updated list of evaluation questions with any new or adjusted confidence thresholds.
     • Describe specific modifications made to the storing function to improve data traceability and completeness, highlighting how these changes help in better evaluations.

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
Human Guidelines:

• Ensure the final output's data is fully traceable to the database and that the data used is directly reflected in the output.
• The final markdown output must be fully human-readable, contextually coherent, and useful to the business.
• Present smaller, verifiable results with nonzero outputs before constructing more complex queries. The higher the quality of the data, the more segmented and detailed the output should be.
• Avoid using dummy data; the provided data must be used to generate insights.
• Each new OKR, Insight, and Suggestion must offer a novel idea distinct from previous generations.
• Insights should detail problems or opportunities with a high severity/frequency/risk score and include a clear hypothesis for action.
• Insights must use calc statements in the data statement with references to variables and derivations so on the frontend we can see where every value in the data statement comes from.
• In the OKR and Insight, all the numbers must directly come from querying the data and cannot be hallucinated. Eg, do not estimate a [x]% increase, unless we know where the [x]% comes from. Otherwise do not include it.
• Suggestions must integrate all available data points, presenting a convincing, well-justified, and impactful story with high reach, impact, and confidence.
• Code generation should implement suggestions in a manner that meets the expectations of a conversion rate optimizer.

---------------------------------------------------------------------
Goals:

• We have the following goals ranked by priority (always start with the highest priority goal that is not yet achieved):
    1. Ensure there is no hallucinated outputs - do this through the evaluation questions
    2. Success Rate should be higher than 50% - do this primarily by making evaluation questions more permissive
    3. Output quality should be as high as possible
    4. The number of turns to get a successful output should be as low as possible
• Evaluation questions are prompts of the form [type]_questions
    - They must be minimal and permissive to increase success rate
    - They must be strict in ensuring there is no hallucination
        a. okr: all numbers come from queries
        b. insights: all numbers come from queries
        c. suggestions: suggestion comes from valid data points
        d. design: clearly verifies whether suggestion is implemented and if not, verifies locations to implement change
        e. code: verifies that the code actually changes the website
    - They must ensure a level of uniqueness of the output, that it has not been seen before
• Each task (okr, insights, suggestion, design, code) has 0 or 1 successes, and success rate is calculated as the number of successes / total number of tasks
    - Increase success rate by removing questions unlikely to succeed, reducing threshholds, and making questions more permissive. We must ensure a high success rate (> 50%)
    - Increase success rate by improving agent prompts / interactions to better specify what output format and tool usage is needed
• Here is how output quality is measured:
    - okr: (Metrics show change) * (Business relevance) * (Reach) * (Readability)
        a. Metrics show change (0 - 1): the OKR values show changes throughout the week, so we can impact it with our suggestions (1 is lots of change, 0 is no change)
        b. Business relevance (0 - 1): how relevant this is to the business
        c. Reach (# of users, no upper limit): how many users this OKR is relevant to
        d. Readability (0 - 1): how readable and intuitive this looks to the business owner
    - insights: (Severity) * (Frequency) * (Confidence) * (Readability)
        a. Severity (1 - 5): how severe the problem is or how big the opportunity is
        b. Frequency (# of occurrences, no upper limit): how often this problem occurs
        c. Confidence (0 - 1): how confident we are in this insight (evaluates confidence in queries and analysis)
        d. Readability (0 - 1): how readable and trustworthy this looks to the business owner (evaluates the storytelling of the insight)
    - suggestion: (Reach) * (Impact) * (Confidence) * (Business relevance) * (Readability)
        a. Reach (0 - 1): (# of users who will see the test) / (reach of OKR)
        b. Impact (0 - no upper limit): Estimated magnitude of impact per user as a percent increase / decrease in the metric for what we are targeting (eg 50 for 50% increase in conversion rate or 50 for 50% decrease in bounce rate)
        c. Confidence (0 - 1): how confident we are in this suggestion (evaluates data relevancy and quality)
        d. Business relevance (0 - 1): how relevant this is to the business (also evaluates if this is already implemented, if it is, this is a 0 - we get this from web agent in design workflow)
        e. Readability (0 - 1): how readable and trustworthy this looks to the business owner (evaluates the storytelling of the suggestion)
    - design: (Clarity):
        a. Clarity (0 - 1): how clear the design is to the business owner, shows all locations to implement and exactly what the change should look like
    - code: (Impact):
        a. Impact (0 - no upper limit): Estimated magnitude of impact per user as a percent increase / decrease in the metric for what we are targeting (we get this through predictive session recordings)
    * All # estimates are estimated by a daily average from the past week
• We aim to reduce the number of turns to get a successful output because the cost and time are proportional to the number of turns

---------------------------------------------------------------------
Helpful Tips:
• Tool Assignment in create_group_chat.py:
    - Different agents have access to different tools based on their role
    - Always refer to the create_group_chat.py file to see which specific tools are assigned to each agent type
    - Ensure your prompt updates only reference tools the agent can actually use
    - Common pattern: data_analyst has SQL query tools, insight_generator has reasoning tools, suggestion_generator has data access tools
    - If agent needs data from a specific tool, either suggest adding the tool to that agent OR implement a way for the agent to request that data from another agent that has access

• Optimizations should be aware of limitations of the data:
    - Using run_sitewiz_query, we can find time viewed on page / scroll depths -  to find elements viewed (from the funnels table), # of clicks, # of errors, # of hovers, and other similar metrics, but it is difficult to get metrics like conversation rate, ctr, etc. so they must be calculated from the available data / metrics.
    - We can segment on dimensions like browser, device, country, # of pages visited etc. but we cannot segment on metrics like conversion rate, revenue, etc. as they are not directly available in the data.
    - Revenue and e-commerce metrics might be inaccurate due to the way they are calculated, so it is better to focus on metrics like time on page, scroll depth, etc.
    - Session recordings / videos should be found from the get_similar_session_recordings which gets the videos and descriptions of similar session recordings (it precomputes summary and finds summaries similar to the query)
    - Heatmaps should be found from get_heatmap which returns an overlayed scroll+click+hover heatmap for a given page with top elements and attributes like location and color

• Tools must be called in the right order so the relevant data is available for the next tool to use.
    - eg, get_heatmap and get_similar_session_recordings tools should be called with outputs validated with retries before storing suggestions
    - Update agent prompts and interactions to ensure that the right tools are being used to get the right data to input into other tools
• You may revert prompts to previous versions if the current version is not performing well.

---------------------------------------------------------------------

By following these guidelines, you will produce a refined set of prompts and code changes to drive improved performance in digital experience optimization automation using vertical AI Agents.
"""

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@measure_time
def run_completion_with_fallback(
    messages=None, 
    prompt=None, 
    models=model_fallback_list, 
    response_format=None, 
    temperature=None, 
    num_tries=3,
    include_tool_messages: bool = True
) -> Optional[Union[str, Dict]]:
    """Run completion with fallback and tool message tracking."""
    initialize_vertex_ai()
    tracker = ToolMessageTracker()

    if messages is None:
        if prompt is None:
            raise ValueError("Either messages or prompt should be provided.")
        messages = [{"role": "user", "content": prompt}]

    # Add tool messages to context if requested
    if include_tool_messages and tracker.messages:
        tool_context = tracker.get_context()
        # Add tool context to the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] += "\n" + tool_context
                break

    for attempt in range(num_tries):
        for model in models:
            try:
                trimmed_messages = messages
                try:
                    trimmed_messages = trim_messages(messages, model)
                except Exception as e:
                    log_error(f"Error trimming messages", e)

                if response_format is None:
                    response = completion(
                        model="litellm_proxy/"+model, 
                        messages=trimmed_messages, 
                        temperature=temperature
                    )
                    return response.choices[0].message.content
                else:
                    response = completion(
                        model="litellm_proxy/"+model, 
                        messages=trimmed_messages,
                        response_format=response_format,
                        temperature=temperature
                    )
                    content = json.loads(response.choices[0].message.content)
                    if isinstance(response_format, BaseModel):
                        response_format.model_validate(content)
                    return content

            except Exception as e:
                error_msg = f"Failed to run completion with model {model}: {str(e)}"
                log_error(error_msg)
                # Add error to tracker
                tracker.add_message(
                    tool_name="completion",
                    input_msg=str(trimmed_messages),
                    response="",
                    error=error_msg
                )

    return None

def get_dynamodb_table(table_name: str):
    """Get DynamoDB table resource."""
    return get_boto3_resource('dynamodb').Table(table_name)

@measure_time
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
    """Get prompt with highest version from DynamoDB PromptsTable by ref."""
    try:
        table = get_dynamodb_table('PromptsTable')
        # Query the table for all versions of this ref
        response = table.query(
            KeyConditionExpression='#r = :ref',
            ExpressionAttributeNames={'#r': 'ref'},
            ExpressionAttributeValues={':ref': ref},
            ScanIndexForward=False,  # This will sort in descending order
            Limit=1  # We only need the most recent version
        )
        
        if not response['Items']:
            print(f"No prompt found for ref: {ref}")
            return ""
            
        return response['Items'][0]['content']
    except Exception as e:
        print(f"Error getting prompt {ref} from DynamoDB: {e}")
        return ""


@measure_time
def get_github_files(token, repo="SitewizAI/sitewiz", target_path="backend/agents/data_analyst_group"):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    def get_contents(path=""):
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        if (response.status_code != 200):
            print(response.json())
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

@measure_time
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
    """Get open issues from a specific GitHub project."""
    if not token:
        print("No GitHub token provided")
        return []

    # First get the project ID
    project_id = get_project_id(token, org_name, project_number, project_name)
    if not project_id:
        print("Could not get project ID")
        print("token: ", token)
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
        
        if (response.status_code != 200):
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
                
            # Only include OPEN issues
            if content.get('state') != 'OPEN':
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
        
        log_debug(f"Found {len(issues)} open issues")    
        return issues
        
    except Exception as e:
        print(f"Error processing project issues: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []


# Global cache for prompts
_prompt_cache: Dict[str, List[Dict[str, Any]]] = {}

@measure_time
def get_prompts(refs: Optional[List[str]] = None, max_versions: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get prompts from DynamoDB PromptsTable with version history.
    Returns the first version and the most recent (max_versions-1) versions.
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        
        if refs is None:
            # Scan for all unique refs using ExpressionAttributeNames
            response = table.scan(
                ProjectionExpression='#r',
                ExpressionAttributeNames={'#r': 'ref'}
            )
            refs = list(set(item['ref'] for item in response.get('Items', [])))
            
            # Handle pagination for the scan operation if necessary
            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ProjectionExpression='#r',
                    ExpressionAttributeNames={'#r': 'ref'},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                refs.extend(list(set(item['ref'] for item in response.get('Items', []))))
            
            # Remove duplicates
            refs = list(set(refs))
        
        prompts = {}
        for ref in refs:
            # Query for all versions of this ref
            response = table.query(
                KeyConditionExpression='#r = :ref',
                ExpressionAttributeNames={'#r': 'ref'},
                ExpressionAttributeValues={':ref': ref}
            )
            
            if not response['Items']:
                continue
            
            # Handle pagination if needed
            all_versions = response['Items']
            while 'LastEvaluatedKey' in response:
                response = table.query(
                    KeyConditionExpression='#r = :ref',
                    ExpressionAttributeNames={'#r': 'ref'},
                    ExpressionAttributeValues={':ref': ref},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                all_versions.extend(response['Items'])
                
            # Sort by version (convert to int for proper numerical sorting)
            all_versions.sort(key=lambda x: int(x.get('version', 0)))
            
            # Take the first version and most recent (max_versions - 1) versions
            if max_versions <= 1:
                selected_versions = [all_versions[-1]]  # Just the latest
            else:
                if len(all_versions) <= max_versions:
                    # If we have fewer versions than max_versions, take all of them
                    selected_versions = all_versions
                else:
                    # Take first version and (max_versions-1) most recent
                    selected_versions = [all_versions[0]] + all_versions[-(max_versions-1):]
            
            # Sort again to ensure latest versions are first
            selected_versions.sort(key=lambda x: int(x.get('version', 0)), reverse=True)
            
            prompts[ref] = selected_versions
            _prompt_cache[ref] = selected_versions
            
        log_debug(f"Retrieved prompt history for {len(prompts)} refs")
        return prompts
    except Exception as e:
        log_error(f"Error getting prompts: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}
    

def get_daily_metrics_from_table(eval_type: str, days: int = 30) -> Dict[str, Any]:
    """
    Fetch metrics directly from DateEvaluationsTable using the updated schema.
    This uses the primary key structure of type + timestamp.
    """
    try:
        if not eval_type:
            raise ValueError("eval_type must be specified")
            
        # Get daily metrics from DateEvaluationsTable
        dynamodb = get_boto3_resource('dynamodb')
        date_table = dynamodb.Table('DateEvaluationsTable')
        
        # Calculate timestamp range for the query - use start of day timestamps
        n_days_ago = datetime.now(timezone.utc) - timedelta(days=days)
        start_timestamp = int(n_days_ago.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        
        # Query metrics by type and timestamp range using the primary key (type + timestamp)
        response = date_table.query(
            KeyConditionExpression='#type = :type_val AND #ts >= :start_ts',
            ExpressionAttributeNames={
                '#type': 'type',
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':type_val': eval_type,
                ':start_ts': Decimal(str(start_timestamp))
            }
        )
        
        items = response.get('Items', [])
        print(f"Found {len(items)} metrics entries for {eval_type} in the last {days} days")
        
        if not items:
            return {
                'total_metrics': {
                    'total_evaluations': 0,
                    'total_successes': 0,
                    'total_attempts': 0,
                    'total_turns': 0,
                    'success_rate': 0.0
                },
                'daily_metrics': {}
            }
            
        # Process daily metrics
        daily_metrics = {}
        total_metrics = {
            'total_evaluations': 0,
            'total_successes': 0,
            'total_attempts': 0,
            'total_turns': 0
        }
        
        for item in items:
            # Extract date and data
            date = item['date']  # Using the date attribute directly
            data = item['data']
            
            # Convert Decimal values to float for calculations
            data_dict = {k: float(v) if isinstance(v, Decimal) else v for k, v in data.items()}
            
            # Populate metrics dictionary
            daily_metrics[date] = {
                'evaluations': data_dict.get('evaluations', 0),
                'successes': data_dict.get('successes', 0),
                'attempts': data_dict.get('attempts', 0), 
                'turns': data_dict.get('turns', 0),
                'quality_metric': data_dict.get('quality_metric', 0)
            }
            
            # Update total metrics
            total_metrics['total_evaluations'] += daily_metrics[date]['evaluations']
            total_metrics['total_successes'] += daily_metrics[date]['successes']
            total_metrics['total_attempts'] += daily_metrics[date]['attempts']
            total_metrics['total_turns'] += daily_metrics[date]['turns']
        
        # Calculate success rate for total metrics
        total_metrics['success_rate'] = (
            (total_metrics['total_successes'] / total_metrics['total_evaluations'] * 100)
            if total_metrics['total_evaluations'] > 0 else 0.0
        )
        
        # Calculate success rate for each day
        for date, metrics in daily_metrics.items():
            metrics['success_rate'] = (
                (metrics['successes'] / metrics['evaluations'] * 100)
                if metrics['evaluations'] > 0 else 0.0
            )
        
        return {
            'total_metrics': total_metrics,
            'daily_metrics': daily_metrics
        }
        
    except Exception as e:
        log_error(f"Error getting daily metrics: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'total_metrics': {},
            'daily_metrics': {}
        }

@measure_time
def get_context(
    stream_key: str, 
    current_eval_timestamp: Optional[float] = None,
    return_type: str = "string",
    include_github_issues: bool = False,
    include_code_files: bool = True,
    past_eval_count: int = 3  # New parameter to control how many past evaluations to include
) -> Union[str, Dict[str, Any]]:
    """Create context from evaluations, prompts, files, and daily metrics."""
    try:
        # Calculate timestamp for one week ago
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        one_week_ago_timestamp = one_week_ago.timestamp()
        
        # Prepare parallel DynamoDB queries
        queries = []
        
        # Add query for evaluations with filter for verified=True and within past week
        evaluations_table = get_dynamodb_table('EvaluationsTable')
        queries.append({
            'table': evaluations_table,
            'key': 'evaluations',
            'params': {
                'KeyConditionExpression': 'streamKey = :streamKey',
                'FilterExpression': '#verified = :verified AND #ts >= :week_ago',
                'ExpressionAttributeNames': {
                    '#verified': 'verified',
                    '#ts': 'timestamp'
                },
                'ExpressionAttributeValues': {
                    ':streamKey': stream_key,
                    ':verified': True,
                    ':week_ago': Decimal(str(one_week_ago_timestamp))
                },
                'ScanIndexForward': False,
                'Limit': past_eval_count + 1
            }
        })
        
        # Add queries for OKRs, insights, and suggestions with verified=True and within past week
        for table_info in [
            ('website-okrs', 'okrs'),
            ('website-insights', 'insights'),
            ('WebsiteReports', 'suggestions')
        ]:
            table = get_dynamodb_table(table_info[0])
            queries.append({
                'table': table,
                'key': table_info[1],
                'params': {
                    'KeyConditionExpression': 'streamKey = :sk AND #ts >= :week_ago',
                    'FilterExpression': '#verified = :verified',
                    'ExpressionAttributeNames': {
                        '#verified': 'verified',
                        '#ts': 'timestamp'
                    },
                    'ExpressionAttributeValues': {
                        ':sk': stream_key,
                        ':verified': True,
                        ':week_ago': Decimal(str(one_week_ago_timestamp))
                    }
                }
            })
        
        # Execute all queries in parallel
        results = parallel_dynamodb_query(queries)
        
        # Process results
        evaluations = results.get('evaluations', [])
        if not evaluations:
            raise ValueError(f"No evaluations found for stream key: {stream_key}")
        
        # Sort evaluations by timestamp
        evaluations.sort(key=lambda x: float(x.get('timestamp', 0)), reverse=True)
        
        # Get current evaluation
        if current_eval_timestamp:
            current_eval = next(
                (e for e in evaluations if float(e['timestamp']) == current_eval_timestamp),
                evaluations[0]
            )
        else:
            current_eval = evaluations[0]
            
        # Get previous evaluations (get more for prompt content extraction)
        prev_evals = [
            e for e in evaluations 
            if float(e['timestamp']) < float(current_eval['timestamp'])
        ][:past_eval_count]
        
        # Extract prompt refs from current and previous evaluations
        all_prompt_refs = []
        current_prompt_refs = current_eval.get('prompts', [])
        all_prompt_refs.extend([p.get('ref') for p in current_prompt_refs if isinstance(p, dict) and 'ref' in p])
        
        prev_prompts_by_eval = []
        for eval_idx, eval_item in enumerate(prev_evals):
            prompt_refs = []
            for p in eval_item.get('prompts', []):
                if isinstance(p, dict) and 'ref' in p:
                    prompt_refs.append(p)
                    all_prompt_refs.append(p.get('ref'))
            prev_prompts_by_eval.append({
                'eval_index': eval_idx,
                'timestamp': eval_item.get('timestamp'),
                'prompts': prompt_refs
            })

        
        # Get prompts with version history, including those from past evaluations
        prompts_dict = get_prompts()
        
        # Extract specific prompt contents for the versions used in evaluations
        current_prompt_contents = []
        for p in current_prompt_refs:
            if isinstance(p, dict) and 'ref' in p and 'content' in p:
                current_prompt_contents.append({
                    'ref': p['ref'],
                    'version': p.get('version', 'N/A'),
                    'content': p['content'],
                    'is_object': p.get('is_object', False)
                })
        
        past_prompt_contents = []
        for eval_prompts in prev_prompts_by_eval:
            eval_prompt_contents = []
            for p in eval_prompts.get('prompts', []):
                if isinstance(p, dict) and 'ref' in p and 'content' in p:
                    eval_prompt_contents.append({
                        'ref': p['ref'],
                        'version': p.get('version', 'N/A'),
                        'content': p['content'],
                        'is_object': p.get('is_object', False),
                        'eval_index': eval_prompts['eval_index'],
                        'timestamp': eval_prompts['timestamp']
                    })
            past_prompt_contents.append(eval_prompt_contents)
        
        # Get dates for daily metrics queries
        current_date = datetime.fromtimestamp(float(current_eval['timestamp']), tz=timezone.utc)
        dates_to_query = []
        
        # Get current date and 7 previous days for historical metrics
        for i in range(8):  # Current day + 7 previous days
            query_date = (current_date - timedelta(days=i)).strftime('%Y-%m-%d')
            dates_to_query.append(query_date)
        
        # Query for daily metrics and historical prompt versions
        daily_metrics = []
        historical_prompt_versions = []
        
        # Get evaluation type from current evaluation
        eval_type = current_eval.get('type')
        
        # Query DateEvaluationsTable for each date
        log_debug(f"Querying DateEvaluationsTable for {len(dates_to_query)} dates")
        date_metrics_table = get_dynamodb_table('DateEvaluationsTable')
        
        for date_str in dates_to_query:
            try:
                # Convert date to timestamp for the beginning of the day
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
                start_of_day_timestamp = int(date_obj.timestamp())
                
                # Query using the primary key structure (type + timestamp)
                response = date_metrics_table.get_item(
                    Key={
                        'type': eval_type,
                        'timestamp': Decimal(str(start_of_day_timestamp))
                    }
                )
                
                item = response.get('Item')
                if item:
                    log_debug(f"Found metrics for {eval_type} on {date_str}")
                    
                    # Add date information to the item for better context
                    item['query_date'] = date_str
                    daily_metrics.append(item)
                    
                    # Extract prompt versions if available
                    if 'promptVersions' in item and item['promptVersions']:
                        log_debug(f"Found {len(item['promptVersions'])} prompt versions for {date_str}")
                        
                        # Add date info to each prompt version
                        for prompt_version in item['promptVersions']:
                            prompt_version['date'] = date_str
                        
                        historical_prompt_versions.extend(item['promptVersions'])
            except Exception as e:
                log_error(f"Error getting daily metrics for {date_str}", e)
                continue
        
        # Log results
        log_debug(f"Retrieved {len(daily_metrics)} daily metrics entries")
        log_debug(f"Retrieved {len(historical_prompt_versions)} historical prompt versions")

        # Get data for the stream key
        data = get_data(stream_key)

        # Count items in each data category
        okr_count = len(data.get('okrs', []))
        insight_count = len(data.get('insights', []))
        suggestion_count = len(data.get('suggestions', []))
        code_count = len(data.get('code', []))
        
        github_token = os.getenv('GITHUB_TOKEN')

        # Get GitHub issues if requested
        github_issues = []
        if include_github_issues and github_token:
            github_issues = get_github_project_issues(github_token)
        issue_count = len(github_issues)
        print(f"# of GitHub issues: {issue_count}")
        
        # Get Python files only if requested
        file_contents = []
        if include_code_files and github_token:
            file_contents = asyncio.run(get_github_files_async(github_token))
            file_count = len(file_contents)
            print(f"# of GitHub files: {file_count}")
        else:
            file_count = 0
            file_contents = []
        
        # Generate data statistics
        data_stats = {
            "evaluations": {
                "current": 1,
                "previous": len(prev_evals)
            },
            "daily_metrics": len(daily_metrics),
            "historical_prompts": len(historical_prompt_versions),
            "okrs": okr_count,
            "insights": insight_count,
            "suggestions": suggestion_count,
            "code": code_count,
            "github_issues": issue_count,
            "code_files": file_count
        }
        
        # Prepare context data with enhanced prompt information
        context_data = {
            "current_eval": {
                "timestamp": datetime.fromtimestamp(float(current_eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
                "type": current_eval.get('type', 'N/A'),
                "successes": current_eval.get('successes', 0),
                "attempts": current_eval.get('attempts', 0),
                "failure_reasons": current_eval.get('failure_reasons', []),
                "conversation": current_eval.get('conversation', ''),
                "prompts_used": current_prompt_refs,
                "prompt_contents": current_prompt_contents,  # New field with full prompt contents
                "raw": current_eval
            },
            "prev_evals": [{
                "timestamp": datetime.fromtimestamp(float(e['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
                "type": e.get('type', 'N/A'),
                "successes": e.get('successes', 0),
                "attempts": e.get('attempts', 0),
                "failure_reasons": e.get('failure_reasons', []),
                "summary": e.get('summary', 'N/A'),
                "prompts_used": e.get('prompts', []),
                "prompt_contents": past_prompt_contents[idx] if idx < len(past_prompt_contents) else [],  # New field with full prompt contents
                "raw": e
            } for idx, e in enumerate(prev_evals)],
            "prompts": prompts_dict,
            "data": data,
            "files": file_contents,
            "daily_metrics": daily_metrics,
            "historical_prompt_versions": historical_prompt_versions,
            "data_stats": data_stats
        }
        
        if include_github_issues:
            context_data["github_issues"] = github_issues
        
        if return_type == "dict":
            return context_data
        
        # Create a data statistics section
        data_stats_str = f"""
Data Statistics:
- Evaluations: {data_stats['evaluations']['current']} current, {data_stats['evaluations']['previous']} previous
- Daily Metrics: {data_stats['daily_metrics']} entries
- Historical Prompts: {data_stats['historical_prompts']} versions
- OKRs: {data_stats['okrs']}
- Insights: {data_stats['insights']}
- Suggestions: {data_stats['suggestions']}
- Code: {data_stats['code']}
- GitHub Issues: {data_stats['github_issues']}
- Code Files: {data_stats['code_files']}
"""
            
        # Build enhanced context string with prompt contents from past evaluations
        context_str = f"""
{data_stats_str}

Daily Metrics (Past Week):
{' '.join(f'''
Date: {metrics['query_date']}
Metrics for Type {metrics['type']}:
- Evaluations: {metrics['data']['evaluations']}
- Successes: {metrics['data']['successes']}
- Success Rate: {(metrics['data']['successes'] / metrics['data']['evaluations'] * 100) if metrics['data']['evaluations'] > 0 else 0:.1f}%
- Quality Metric: {metrics['data']['quality_metric']}
- Turns: {metrics['data']['turns']}
- Attempts: {metrics['data']['attempts']}
''' for metrics in daily_metrics[:7])}  # Limit to most recent 7 days

Historical Prompt Versions:
{' '.join(f'''
Date: {pv['date']}
Prompt: {pv.get('ref', 'N/A')} (Version {pv.get('version', 'N/A')})
''' for pv in historical_prompt_versions[:10])}  # Limit to 10 most recent versions

Current Evaluation:
Timestamp: {context_data['current_eval']['timestamp']}
Type: {context_data['current_eval']['type']}
Successes: {context_data['current_eval']['successes']}
Attempts: {context_data['current_eval']['attempts']}
Failure Reasons: {context_data['current_eval']['failure_reasons']}
Conversation History:
{context_data['current_eval']['conversation']}

Current Evaluation Prompts Used:
{' '.join(f'''
Prompt {p.get('ref', 'N/A')} (Version {p.get('version', 'N/A')}):
Content:
{p.get('content', 'N/A')}
''' for p in current_prompt_contents)}

Past Evaluations Prompts Used:
{' '.join(f'''
Evaluation from {prev_evals[idx]['timestamp'] if idx < len(prev_evals) else 'N/A'}:
{' '.join(f'''
Prompt {p.get('ref', 'N/A')} (Version {p.get('version', 'N/A')}):
Content:
{p.get('content', 'N/A')}
''' for p in prompt_contents)}
''' for idx, prompt_contents in enumerate(past_prompt_contents))}

Previous Evaluations:
{' '.join(f'''
Evaluation from {e['timestamp']}:
- Type: {e['type']}
- Successes: {e['successes']}
- Attempts: {e['attempts']}
- Failure Reasons: {e['failure_reasons']}
- Summary: {e['summary']}
''' for e in context_data['prev_evals'])}

Current Data:
OKRs ({okr_count}):
{' '.join(okr['markdown'] for okr in data.get('okrs', []))}

Insights ({insight_count}):
{' '.join(insight['markdown'] for insight in data.get('insights', []))}

Suggestions ({suggestion_count}):
{' '.join(suggestion['markdown'] for suggestion in data.get('suggestions', []))}

Python Files Content ({file_count} files):
{' '.join(f'''
File {file['file']['path']}:
{file['content']}
''' for file in file_contents)}
"""

        if include_github_issues:
            context_str += f"""
Recent GitHub Issues ({issue_count}):
{' '.join(f'''
#{issue['number']}: {issue['title']}
{issue['body'][:200]}...
''' for issue in github_issues)}
"""

        return context_str
    except Exception as e:
        log_error(f"Error creating context for stream key {stream_key}", e)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}

def get_most_recent_stream_key(eval_type: Optional[str] = None) -> Tuple[Optional[str], Optional[float]]:
    """
    Get the stream key and timestamp with the most recent evaluation, optionally filtered by type.
    Uses 'type-timestamp-index' GSI with partition key 'type' and sort key 'timestamp'.
    
    Args:
        eval_type: Optional type to filter by (e.g. 'okr', 'insights', etc)
    
    Returns:
        Tuple of (stream_key, timestamp) or (None, None) if not found
    """
    try:
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('EvaluationsTable')
        
        if eval_type:
            # Query just for the specified type
            response = table.query(
                IndexName='type-timestamp-index',
                KeyConditionExpression='#type = :type_val',
                ExpressionAttributeNames={
                    '#type': 'type'
                },
                ExpressionAttributeValues={
                    ':type_val': eval_type
                },
                ScanIndexForward=False,  # Get in descending order
                Limit=1
            )
            
            items = response.get('Items', [])
            if items:
                item = items[0]
                return item['streamKey'], float(item['timestamp'])
            return None, None
        
        # If no type specified, check all types
        types = ['okr', 'insights', 'suggestion', 'code']
        most_recent_item = None
        
        for type_val in types:
            response = table.query(
                IndexName='type-timestamp-index',
                KeyConditionExpression='#type = :type_val',
                ExpressionAttributeNames={
                    '#type': 'type'
                },
                ExpressionAttributeValues={
                    ':type_val': type_val
                },
                ScanIndexForward=False,
                Limit=1
            )
            
            items = response.get('Items', [])
            if items:
                item = items[0]
                if most_recent_item is None or float(item['timestamp']) > float(most_recent_item['timestamp']):
                    most_recent_item = item
        
        if most_recent_item:
            return most_recent_item['streamKey'], float(most_recent_item['timestamp'])
        return None, None
        
    except Exception as e:
        log_error(f"Error getting most recent stream key", e)
        return None, None

def get_label_ids(token: str, org: str, repo: str, label_names: List[str]) -> List[str]:
    """Get GitHub label IDs from label names."""
    query = """
    query($org: String!, $repo: String!, $searchQuery: String!) {
        repository(owner: $org, name: $repo) {
            labels(first: 100, query: $searchQuery) {
                nodes {
                    id
                    name
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
        # Combine all label names into a single search query
        search_query = " ".join(label_names)
        
        response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={
                "query": query,
                "variables": {
                    "org": org,
                    "repo": repo,
                    "searchQuery": search_query
                }
            }
        )
        data = response.json()
        if "errors" in data:
            print(f"Error getting label IDs: {data['errors']}")
            return []
            
        labels = data.get("data", {}).get("repository", {}).get("labels", {}).get("nodes", [])
        # Only return IDs for exact name matches
        return [label["id"] for label in labels if label["name"] in label_names]
    except Exception as e:
        print(f"Error fetching label IDs: {str(e)}")
        return []

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
    """Create a GitHub issue, add it to a project, and apply labels."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v4+json"
    }

    # First get the label IDs
    label_ids = get_label_ids(token, org, repo, labels)
    if not label_ids:
        print("Warning: No valid label IDs found")
    
    # Get repository ID query
    repo_query = """
    query($org: String!, $repo: String!) {
        repository(owner: $org, name: $repo) {
            id
        }
    }
    """
    
    # Create issue mutation
    create_issue_query = """
    mutation($input: CreateIssueInput!) {
        createIssue(input: $input) {
            issue {
                id
                number
                url
            }
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
        issue_input = {
            "repositoryId": repo_id,
            "title": title,
            "body": body
        }
        if label_ids:
            issue_input["labelIds"] = label_ids

        issue_response = requests.post(
            "https://api.github.com/graphql",
            headers=headers,
            json={
                "query": create_issue_query,
                "variables": {
                    "input": issue_input
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
        mutation($input: AddProjectV2ItemByIdInput!) {
            addProjectV2ItemById(input: $input) {
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
                    "input": {
                        "projectId": project_id,
                        "contentId": issue["id"]
                    }
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
    

def get_test_parameters() -> List[str]:
    """Get list of all possible prompt format parameters."""
    return [
        "question",
        "business_context",
        "stream_key",
        "insight_example",
        "insight_notes",
        "insight_criteria",
        "okrs",
        "insights",
        "suggestions",
        "additional_instructions",
        "function_details",
        "functions_module",
        "name",
        "all_okr_prompts",
        "suggestion_example",
        "suggestion_notes",
        "suggestion_criteria",
        "questions",
        "okr_criteria",
        "okr_code_example",
        "okr_notes",
        "reach_example",
        "criteria",
        "code_example",
        "notes"
    ]

def validate_prompt_format(content: str) -> bool:
    """Validate that a prompt string can be formatted with test parameters."""
    try:
        # Create test parameters dictionary with empty strings
        test_params = {param: "" for param in get_test_parameters()}
        
        # Try formatting the content
        formatted = content.format(**test_params)
        log_debug("Prompt format validation successful")
        return True
        
    except KeyError as e:
        log_error(f"Invalid format key in prompt: {e}")
        return False
    except ValueError as e:
        log_error(f"Invalid format value in prompt: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error in prompt validation: {e}")
        return False
    

def update_prompt(ref: str, content: Union[str, Dict[str, Any]]) -> bool:    
    """Update or create a prompt in DynamoDB PromptsTable with versioning and validation."""
    try:
        table = get_dynamodb_table('PromptsTable')
        
        # Get latest version of the prompt
        response = table.query(
            KeyConditionExpression='#r = :ref',
            ExpressionAttributeNames={'#r': 'ref'},
            ExpressionAttributeValues={':ref': ref},
            ScanIndexForward=False,
            Limit=1
        )

        # Get current version and type information
        if not response.get('Items'):
            log_error(f"No prompt found for ref: {ref}")
            return False
            
        latest_prompt = response['Items'][0]
        current_version = int(latest_prompt.get('version', 0))
        is_object_original = latest_prompt.get('is_object', False)
        created_at = latest_prompt.get('createdAt', datetime.now().isoformat())
        
        log_debug(f"Updating prompt {ref} (current version: {current_version}, is_object: {is_object_original})")
        
        # Determine the content type of the new content
        is_object_new = isinstance(content, dict)
        
        # If content is a string, check if it's actually JSON
        if isinstance(content, str) and not is_object_new:
            try:
                # Try parsing as JSON - if successful, it should be treated as an object
                content_obj = json.loads(content)
                # Only change type if it's a valid object (dict)
                if isinstance(content_obj, dict):
                    is_object_new = True
                    content = content_obj
                    log_debug(f"String content parsed as JSON object for prompt {ref}")
            except json.JSONDecodeError:
                # Not valid JSON, keep as string
                is_object_new = False
                log_debug(f"Content for prompt {ref} is a regular string")
        
        # For backwards compatibility: if original is object but provided as string 
        # and the string is valid JSON, parse it
        if is_object_original and isinstance(content, str) and not is_object_new:
            try:
                content = json.loads(content)
                is_object_new = True
                log_debug(f"Parsed string content into JSON for object-type prompt {ref}")
            except json.JSONDecodeError:
                log_error(f"Content provided as string but prompt {ref} requires JSON object")
                return False
        
        # Make sure type is consistent
        if is_object_original != is_object_new:
            log_error(f"Content type mismatch for prompt {ref}: original={is_object_original}, new={is_object_new}")
            return False

        # Validate string content
        if isinstance(content, str):
            if not validate_prompt_format(content):
                log_error(f"Prompt validation failed for ref: {ref}")
                return False
        
        # Create new version
        new_version = current_version + 1
        
        # Prepare item for DynamoDB
        item = {
            'ref': ref,
            'content': json.dumps(content) if is_object_new else content,
            'version': new_version,
            'is_object': is_object_new,
            'updatedAt': datetime.now().isoformat(),
            'createdAt': created_at
        }
        
        log_debug(f"Creating new version {new_version} for prompt {ref}")
        
        # Store the content
        table.put_item(Item=item)
        
        log_debug(f"Successfully updated prompt {ref} to version {new_version}")
        return True
        
    except ClientError as e:
        log_error(f"DynamoDB error updating prompt {ref}", e)
        return False
    except Exception as e:
        log_error(f"Error updating prompt {ref}", e)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

@measure_time
def get_all_evaluations(limit_per_stream: int = 1000, eval_type: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all recent evaluations using type-timestamp-index.
    Uses the GSI to get evaluations efficiently.
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        
        # Calculate timestamp for filtering
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        one_week_ago_timestamp = Decimal(one_week_ago.timestamp())
        
        # Query params using type-timestamp-index
        query_params = {
            'IndexName': 'type-timestamp-index',
            'KeyConditionExpression': '#type = :type AND #ts >= :one_week_ago',
            'ExpressionAttributeNames': {
                '#type': 'type',
                '#ts': 'timestamp'
            },
            'ExpressionAttributeValues': {
                ':type': eval_type,
                ':one_week_ago': one_week_ago_timestamp
            },
            'ScanIndexForward': False,  # Get most recent first
            'Limit': limit_per_stream
        }
        
        # Single query to get all evaluations
        evaluations = []
        response = table.query(**query_params)
        evaluations.extend(response.get('Items', []))
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in response and len(evaluations) < limit_per_stream and len(evaluations) > 0:
            response = table.query(
                **query_params,
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            if len(response.get('Items', [])) == 0:
                break
            evaluations.extend(response.get('Items', []))
            if len(evaluations) >= limit_per_stream:
                evaluations = evaluations[:limit_per_stream]
                break
                
        log_debug(f"Retrieved {len(evaluations)} evaluations using type-timestamp-index")
        return evaluations
        
    except Exception as e:
        log_error("Error getting evaluations", e)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

def debug_dynamodb_table(table):
    """Debug helper to print table information."""
    try:
        log_debug(f"Table name: {table.name}")
        log_debug(f"Table ARN: {table.table_arn}")
        
        # Get table description
        description = table.meta.client.describe_table(TableName=table.name)['Table']
        
        # Log key schema
        log_debug("Key Schema:")
        for key in description.get('KeySchema', []):
            log_debug(f"- {key['AttributeName']}: {key['KeyType']}")
            
        # Log GSIs
        log_debug("Global Secondary Indexes:")
        for gsi in description.get('GlobalSecondaryIndexes', []):
            log_debug(f"- {gsi['IndexName']}:")
            for key in gsi['KeySchema']:
                log_debug(f"  - {key['AttributeName']}: {key['KeyType']}")
                
    except Exception as e:
        log_error(f"Error getting table info", e)

@measure_time
def get_stream_evaluations(stream_key: str, limit: int = 6, eval_type: str = None) -> List[Dict[str, Any]]:
    """
    Fetch recent evaluations for specific stream key, optionally filtered by type.
    Uses filter expression instead of index for type filtering.
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        
        # Debug table structure
        debug_dynamodb_table(table)
        
        # Validate inputs
        if not stream_key:
            raise ValueError("stream_key cannot be empty")
            
        # Base query parameters with proper key condition expression
        query_params = {
            'KeyConditionExpression': 'streamKey = :streamKey',
            'ExpressionAttributeValues': {
                ':streamKey': stream_key
            },
            'ScanIndexForward': False,  # Get most recent first
            'Limit': limit
        }

        log_debug(f"Initial query params: {json.dumps(query_params, indent=2)}")

        # Add type filter if specified
        if eval_type:
            query_params.update({
                'FilterExpression': '#type = :type_val',
                'ExpressionAttributeNames': {
                    '#type': 'type'
                },
            })
            query_params['ExpressionAttributeValues'][':type_val'] = eval_type
            log_debug(f"Updated query params with type filter: {json.dumps(query_params, indent=2)}")

        # First attempt with standard query
        try:
            log_debug("Attempting standard query...")
            response = table.query(**query_params)
            evaluations = response.get('Items', [])
            log_debug(f"Standard query returned {len(evaluations)} items")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                log_debug("Standard query failed, attempting alternative query approach...")
                # Try alternative query approach using scan with filter
                scan_params = {
                    'FilterExpression': 'streamKey = :streamKey',
                    'ExpressionAttributeValues': {
                        ':streamKey': stream_key
                    },
                    'Limit': limit
                }
                if eval_type:
                    scan_params['FilterExpression'] += ' AND #type = :type_val'
                    scan_params['ExpressionAttributeNames'] = {'#type': 'type'}
                    scan_params['ExpressionAttributeValues'][':type_val'] = eval_type
                
                log_debug(f"Fallback scan params: {json.dumps(scan_params, indent=2)}")
                response = table.scan(**scan_params)
                evaluations = response.get('Items', [])
                log_debug(f"Fallback scan returned {len(evaluations)} items")
            else:
                raise

        # Get more items if needed
        while len(evaluations) < limit and 'LastEvaluatedKey' in response and len(evaluations) > 0:
            log_debug(f"Fetching more items, current count: {len(evaluations)}")
            if 'LastEvaluatedKey' in query_params:
                query_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = table.query(**query_params)
            if len(response.get('Items', [])) == 0:
                break
            evaluations.extend(response.get('Items', []))

        log_debug(f"Total items found for stream key {stream_key}: {len(evaluations)}")
        
        # Sort by timestamp and limit results
        evaluations.sort(key=lambda x: float(x.get('timestamp', 0)), reverse=True)
        return evaluations[:limit]
        
    except Exception as e:
        log_error(f"Error getting evaluations for stream key {stream_key}", e)
        import traceback
        log_debug(f"Full error traceback: {traceback.format_exc()}")
        return []

@measure_time
def get_evaluation_metrics(days: int = 30, eval_type: str = None) -> Dict[str, Any]:
    """
    Get evaluation metrics for the last N days using the type-timestamp schema.
    Returns daily and total metrics with proper formatting for visualization.
    """
    try:
        if not eval_type:
            raise ValueError("eval_type must be specified")
            
        # Get daily metrics from DateEvaluationsTable
        dynamodb = get_boto3_resource('dynamodb')
        date_table = dynamodb.Table('DateEvaluationsTable')
        
        # Calculate timestamp range for the query - use start of day timestamps
        n_days_ago = datetime.now(timezone.utc) - timedelta(days=days)
        start_timestamp = int(n_days_ago.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        
        # Query metrics by type and timestamp range
        response = date_table.query(
            KeyConditionExpression='#type = :type_val AND #ts >= :start_ts',
            ExpressionAttributeNames={
                '#type': 'type',
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':type_val': eval_type,
                ':start_ts': Decimal(str(start_timestamp))
            }
        )
        
        # Process daily metrics
        daily_metrics = {}
        total_metrics = {
            'total_evaluations': 0,
            'total_successes': 0,
            'total_attempts': 0,
            'total_turns': 0
        }
        
        # Initialize dates for complete date range
        current_date = n_days_ago.date()
        end_date_obj = datetime.now(timezone.utc).date()
        while current_date <= end_date_obj:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_metrics[date_str] = {
                'evaluations': 0,
                'successes': 0,
                'attempts': 0,
                'turns': 0,
                'quality_metric': 0
            }
            current_date += timedelta(days=1)
        
        # Process metrics from query response
        for item in response.get('Items', []):
            date = item['date']  # Now using the date attribute
            data = item['data']
            
            # Populate metrics dictionary
            metrics = daily_metrics.get(date, {
                'evaluations': 0,
                'successes': 0,
                'attempts': 0, 
                'turns': 0,
                'quality_metric': 0
            })
            
            metrics['evaluations'] = data.get('evaluations', 0)
            metrics['successes'] = data.get('successes', 0)
            metrics['attempts'] = data.get('attempts', 0)
            metrics['turns'] = data.get('turns', 0)
            metrics['quality_metric'] = data.get('quality_metric', 0)
            daily_metrics[date] = metrics
            
            # Update total metrics
            total_metrics['total_evaluations'] += metrics['evaluations']
            total_metrics['total_successes'] += metrics['successes']
            total_metrics['total_attempts'] += metrics['attempts']
            total_metrics['total_turns'] += metrics['turns']
        
        # Calculate success rate for total metrics
        total_metrics['success_rate'] = (
            (total_metrics['total_successes'] / total_metrics['total_evaluations'] * 100)
            if total_metrics['total_evaluations'] > 0 else 0.0
        )
        
        # Calculate success rate and add to daily metrics
        for date, metrics in daily_metrics.items():
            metrics['success_rate'] = (
                (metrics['successes'] / metrics['evaluations'] * 100)
                if metrics['evaluations'] > 0 else 0.0
            )
        
        return {
            'total_metrics': total_metrics,
            'daily_metrics': daily_metrics
        }
    except Exception as e:
        log_error(f"Error getting evaluation metrics: {str(e)}")
        return {
            'total_metrics': {},
            'daily_metrics': {}
        }

@measure_time
def get_recent_evaluations(eval_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get most recent evaluations using type-timestamp-index more efficiently.
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        log_debug(f"Fetching {limit} recent evaluations for type: {eval_type}")
        
        # Single query to get complete data using GSI
        query_params = {
            'IndexName': 'type-timestamp-index',
            'KeyConditionExpression': '#type = :type',
            'ExpressionAttributeNames': {
                '#type': 'type',
                '#fr': 'failure_reasons',
                '#q': 'question',
                '#ts': 'timestamp'  # Add timestamp to expression attribute names
            },
            'ExpressionAttributeValues': {
                ':type': eval_type
            },
            'ProjectionExpression': 'streamKey, #type, successes, attempts, num_turns, #ts, prompts, #fr, #q, summary',
            'ScanIndexForward': False,  # Most recent first
            'Limit': limit
        }
        
        response = table.query(**query_params)
        evaluations = response.get('Items', [])
        log_debug(f"Retrieved {len(evaluations)} evaluations")
        
        if evaluations:
            log_debug(f"Got Evaluation")
            
        return evaluations
        
    except Exception as e:
        log_error(f"Error getting recent evaluations for {eval_type}", e)
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}")
        return []

async def get_github_files_async(token, repo="SitewizAI/sitewiz", target_path="backend/agents/data_analyst_group"):
    """Async version of get_github_files using aiohttp."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    async def get_contents_async(session, path=""):
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                log_error(f"Error accessing {path}: {response.status}")
                return []
            return await response.json()

    async def get_file_content_async(session, file_info):
        async with session.get(file_info["download_url"]) as response:
            if response.status == 200:
                return {"file": file_info, "content": await response.text()}
            log_error(f"Error downloading {file_info['path']}")
            return {"file": file_info, "content": ""}

    async def process_contents_async(session, path=""):
        contents = await get_contents_async(session, path)
        if not isinstance(contents, list):
            contents = [contents]

        python_files = []
        tasks = []

        for item in contents:
            full_path = os.path.join(path, item["name"])
            if item["type"] == "file" and item["name"].endswith(".py"):
                python_files.append({
                    "path": full_path,
                    "download_url": item["download_url"]
                })
            elif item["type"] == "dir":
                tasks.append(process_contents_async(session, item["path"]))

        if tasks:
            results = await asyncio.gather(*tasks)
            for result in results:
                python_files.extend(result)

        return python_files

    async with aiohttp.ClientSession() as session:
        # Get all Python files first
        python_files = await process_contents_async(session, target_path)
        
        # Then fetch all file contents in parallel
        tasks = [get_file_content_async(session, file) for file in python_files]
        return await asyncio.gather(*tasks)

@measure_time
def get_prompt_versions_by_date(date: str, eval_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get stored prompt versions for a specific date using the type-timestamp schema.
    
    Args:
        date: The date string in format 'YYYY-MM-DD'
        eval_type: Optional evaluation type to filter by (defaults to first available type)
    
    Returns:
        List of prompt versions for that date
    """
    try:
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('DateEvaluationsTable')
        
        # Convert date to timestamp for the beginning of the day
        date_obj = datetime.strptime(date, '%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_day_timestamp = int(date_obj.timestamp())
        
        if eval_type:
            # Query directly using the primary key (type + timestamp)
            response = table.get_item(
                Key={
                    'type': eval_type,
                    'timestamp': Decimal(str(start_of_day_timestamp))
                }
            )
            
            item = response.get('Item')
            if item and 'promptVersions' in item:
                return item['promptVersions']
            
            log_debug(f"No prompt versions found for type {eval_type} on {date}")
            return []
        else:
            # If no type is specified, query for all eval types on that date
            types = ['okr', 'insights', 'suggestion', 'code', 'design']
            
            for type_val in types:
                response = table.get_item(
                    Key={
                        'type': type_val,
                        'timestamp': Decimal(str(start_of_day_timestamp))
                    }
                )
                
                item = response.get('Item')
                if item and 'promptVersions' in item:
                    return item['promptVersions']
            
            log_debug(f"No prompt versions found for any type on {date}")
            return []
    
    except Exception as e:
        log_error(f"Error getting prompt versions for date {date}", e)
        return []

# Add parallel_dynamodb_query function
@measure_time
def parallel_dynamodb_query(queries: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Execute multiple DynamoDB queries in parallel.
    
    Args:
        queries: List of dictionaries with keys 'table', 'key', and 'params'
            - table: DynamoDB table resource
            - key: A key to identify the result in the output dictionary
            - params: Parameters to pass to the query method
    
    Returns:
        Dictionary with query keys mapped to results
    """
    try:
        log_debug(f"Executing {len(queries)} parallel DynamoDB queries")
        results = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=min(10, len(queries))) as executor:
            # Create a future for each query
            future_to_key = {}
            
            for query in queries:
                table = query['table']
                key = query['key']
                params = query['params']
                
                # Submit query to executor
                future = executor.submit(
                    lambda t, p: t.query(**p).get('Items', []),
                    table,
                    params
                )
                future_to_key[future] = key
            
            # Process results as they complete
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    items = future.result()
                    results[key] = items
                    log_debug(f"Query for key {key} returned {len(items)} items")
                except Exception as e:
                    log_error(f"Query for key {key} failed", e)
                    results[key] = []
        
        return results
    except Exception as e:
        log_error(f"Error in parallel DynamoDB query", e)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}

