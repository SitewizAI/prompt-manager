"""Utilities for running AI completions."""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
import tiktoken
import litellm
from litellm import completion
from litellm.utils import trim_messages
from pydantic import BaseModel
import boto3


from .logging_utils import log_debug, log_error, ToolMessageTracker, measure_time

# System prompt and instructions
PROMPT_INSTRUCTIONS = """
1. Block-Level Prompt Optimization for Reasoning models (all agents use reasoning models)  
   - Techniques to Use:
     • Bootstrapped Demonstration Extraction: Analyze evaluation traces to identify 2–3 high-quality input/output demonstration examples and formatting that clarify task patterns.
     • Ensure your prompts are straightforward and easy to understand. Avoid ambiguity by specifying exactly what you need from the AI
     • Include specific details, constraints, and objectives to guide the model toward the desired output using domain specific knowledge of digital experience optimization and the agent role
     • Structure complex inputs with clear sections or headings
     • Specify end goal and desired output format explicitly
     • You must ensure agents don't hallucinate outputs by providing clear and detailed prompts
     
   - Prompt Formatting Requirements:
    • The variable substitions should use single brackets, {variable_name}, and the substitution variables must be the ones provided in the code as a second parameter to get_prompt_from_dynamodb
    • Please analyze the code to find the variables being substituted. When the completion is run, the variables will be replaced with the actual values
    • All the substitution variables provided in `get_prompt_from_dynamodb` for the prompt must be used in the prompt
    • For python variables in prompts with python code, ensure that double brackets are used (eg {{ and }}) since we are using python multilined strings for the prompts, especially in example queries since the brackets must be escaped for the prompt to compile, unless we are making an allowed substitution specified in the code

   - Tool Usage Requirements:
    • When updating agent prompts, ONLY reference tools that are actually available to that agent in create_group_chat.py
    • Check which tools are provided to each agent type and ensure your prompt only mentions those specific tools
    • Can update tool prompts with examples so agents better understand how to use them
    • You must ensure that tools are executed with parameters required by the tool function. For code execution, you must ensure that code provided to the code executor is in python blocks, eg ```python ... ```
    • Never include instructions for using tools that aren't explicitly assigned to the agent in create_group_chat.py
    • If an agent needs access to data that requires a tool it doesn't have, suggest adding that tool to the agent in create_group_chat.py rather than mentioning unavailable tools in the prompt

   - Note that all agent instructions are independent
    • IMPORTANT: Instruction updates should only apply to the agent in question, don't put instructions for other agents in the system message for the agent
    • IMPORTANT: Tool calling and python code execution (with database querying) is core to the workflow since final output stored should be based on environment feedback. That means prompts should ensure the right information is fetched from the environment before proceeding to store the output.
    • IMPORTANT: Only the python analyst can do code execution and query the database for data, so it should be core to the workflow
    • IMPORTANT: Using the agents provided, the tools available, and task, each agent should be very clear on what the optimal workflow is to complete the task including the ordering of the agents and information they need from the environment and to provide to the next agent.

    
2. Evaluations Optimization (Improving Success Rate and Quality)
   - Techniques to Use:
     • Refine Evaluation Questions: Review and update the evaluation questions to ensure they precisely measure the desired outcomes (e.g., correctness, traceability, and clarity). Adjust confidence thresholds as needed to better differentiate between successful and unsuccessful outputs. Note we need > 50% success rate in evaluations.
     • Actionable Feedback Generation: For each evaluation failure, generate specific, actionable feedback that identifies the issue (e.g., ambiguous instructions, missing context, or incorrect data integration) and provide concrete suggestions for improvement.
     • Enhanced Evaluation Data Integration: Modify the storing function to ensure that all relevant evaluation details (such as SQL query outputs, execution logs, error messages, and computed metrics) are captured in a structured and traceable manner.
   - Important notes
     • Ensure you know the inputs and their format and that those inputs are used properly in the evaluation questions. Evaluation questions cannot use output or reference variables not provided in the input.
   - Output Requirements:
     • Present an updated list of evaluation questions with any new or adjusted confidence thresholds.
     • Describe specific modifications made to the storing function to improve data traceability and completeness, highlighting how these changes help in better evaluations."""

SYSTEM_PROMPT = """You are a helpful website optimization expert assistant assisting in creating an agentic workflow that automates digital experience optimization – from data analysis to insight/suggestion generation to code implementation. 
Your role is to analyze evaluations and provide recommendations to update the prompts and code files, thereby improving the quality and accuracy of outputs so that each evaluation is successful in a low number of turns. 
Use the provided context to generate specific, accurate, and traceable recommendations that update the code and prompt structure.

---------------------------------------------------------------------
Types of Suggestions to Provide:

{PROMPT_INSRUCTIONS}

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

By following these guidelines, you will produce a refined set of prompts and code changes to drive improved performance in digital experience optimization automation using vertical AI Agents.
""".format(PROMPT_INSRUCTIONS=PROMPT_INSTRUCTIONS)

# Fallback model list
model_fallback_list = ["reasoning", "long"]

aws_region = os.getenv('AWS_REGION') or "us-east-1"

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

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        log_error(f"Error counting tokens: {str(e)}")
        # Fallback token counting: estimate 4 chars per token
        return len(text) // 4
