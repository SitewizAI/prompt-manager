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
from utils.completion_examples_utils import REACH_CODE_EXAMPLE, CODE_EXAMPLE, INSIGHT_EXAMPLE, INSIGHT_STORE_EXAMPLE, SUGGESTION_EXAMPLE
from utils.prompt_utils import AGENT_TOOLS, AGENT_GROUPS, AGENT_GROUPS_TEXT
from .logging_utils import log_debug, log_error, ToolMessageTracker, measure_time

# Examples
AGENT_SYSTEM_PROMPT_TEMPLATE = """This is the system prompt provided to the agent. Guidelines for the prompt are as follows:
- There should be no blocks in the prompt (eg ```text ...```, ```python ...```, etc) except if it is a python analyst providing python code blocks to execute. IMPORTANT: Python analyst should know that python code blocks cannot include ``` anwhere inside the code block as that would mess up the code execution.

Prompt Template:
---------------------------------------------------------------------
**Role & Objective**  
You are an expert [ROLE, e.g., Data Analyst/Support Agent/Research Assistant] tasked with [PRIMARY GOAL].  
[Optional Secondary Objective, e.g., "Ensure responses align with {brand voice/policy}"]

**Context**  
[Optional: RELEVANT BACKGROUND INFO OR DOMAIN KNOWLEDGE - can include context: {business_context} here if necessary]  
[Optional: "Access to these tools: {tool_name}: {tool_description}"]

**Output Format**  
[Choose ONE:]  
- **Structured**: Respond in [JSON/XML/YAML] with [required fields]  
- **Natural Language**: Use [bullet points/paragraphs] with [tone/style guidance]  
- **Hybrid**: Combine structured data and explanations using [markdown formatting]

**Reasoning Guidelines**  
[Optional for Reasoning Models (agents with 'main' don't use reasoning models)]  
1. [Internal Process, e.g., "Compare against {dataset} before answering"]  
2. [Error Checking, e.g., "Validate calculations against {standard}"]  
3. [Decision Criteria, e.g., "Prioritize solutions meeting {constraint}"]

**Warnings**  
[Optional: "Avoid assumptions about {topic}. Verify via {tool/source} if uncertain"]

**Examples**  
[Optional Few-Shot:]  
Input: "[Sample Query]"  
Output: "[Modeled Response]"
---------------------------------------------------------------------  
"""

AGENT_DESCRIPTION_TEMPLATE = """This is the description that will be used by the group chat to choose the right agent and order of agents for the task. The description should be brief and include the agent's role, responsibilities, and the tools available to the agent."""

TOOL_DESCRIPTION_TEMPLATE = """This is the description used by the agent to determine which tools are available to them and how to use them. The description should include the tool's purpose, inputs, outputs, and any other relevant information."""

TASK_CONTEXT_TEMPLATE = """This context requires previous outputs and is used to provide context for the agent group to better complete the task"""

TASK_QUESTION_TEMPLATE = """This is the question that the agent group must answer to complete the task. The question should be clear and concise, and should guide the agents towards the desired output."""

AGENT_GROUP_INSTRUCTIONS_TEMPLATE = """These are the instructions provided to the agent group to guide them in completing the subtask. The instructions should be clear and concise, and should provide all the information the agents need to complete the task successfully."""

EVALUATION_QUESTIONS_TEMPLATE = """These are the evaluation questions that will be used to evaluate the agent group's output to ensure there is no hallucation in the output. They should be clear, concise, and minimal

"""

PROMPT_INSTRUCTIONS = """
1. Block-Level Prompt Optimization
   - Techniques to Use:
     • Bootstrapped Demonstration Extraction: Analyze evaluation traces to identify 2–3 high-quality input/output demonstration examples and formatting that clarify task patterns.
     • Ensure your prompts are straightforward and easy to understand. Avoid ambiguity by specifying exactly what you need from the AI. Use clear formatting and structure to guide the AI toward the desired output.
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
    • You must ensure that tools are executed with parameters required by the tool function. For code execution, you must ensure that code provided to the code executor is in python blocks, eg ```python ... ``` for all code block examples
    • Never include instructions for using tools that aren't explicitly assigned to the agent in create_group_chat.py
    • If an agent needs access to data that requires a tool it doesn't have, suggest adding that tool to the agent in create_group_chat.py rather than mentioning unavailable tools in the prompt

   - Note that all agent instructions are independent
    • IMPORTANT: Instruction updates should only apply to the agent in question, don't put instructions for other agents in the system message for the agent
    • IMPORTANT: Tool calling and python code execution (with database querying) is core to the workflow since final output stored should be based on environment feedback. That means prompts should ensure the right information is fetched from the environment before proceeding to store the output.
    • IMPORTANT: Only the python analyst can do code execution and query the database for data, so it should be core to the workflow. For the code to run, it must output python code blocks in the form ```python ... ```, make sure agent instructions reflect this. Make sure agents don't output any other code blocks which confuses the code executor (eg text / json / sql / etc code blocks - agent prompts should not have text or python code blocks unless it is sample python code for python analyst)
     - eg all other agent instructions / system messages should be treated as system message and should not contain any ```text ...```, ```python ```, or other blocks since they confuse the code executor and the agent output
    
    • IMPORTANT: Using the agents provided, the tools available, and task, each agent should be very clear on what the optimal workflow is to complete the task including the ordering of the agents and information they need from the environment and to provide to the next agent.
    • IMPORTANT: You must ensure agent and tool prompts are updated so that agents are calling tools with the required parameters, eg:
        - store_okr requires full python function code for reach_code and code. It should not have a goal, it should simply store the OKR with reach we are tracking.
        - store_insight requires full python code for each derivation and the data statement should use calc expressions correctly
        - store_suggestion requires insights from heatmaps / session recordings / insights
        etc
        Also make sure the agents are correctly incentivized, so the store function is attempted at least once and more if it fails.
    • IMPORTANT: Ensure the modularity of the prompt, it should be a viable prompt for any of the groups it is a part of
    
2. Evaluations Optimization (Improving Success Rate and Quality)
   - Techniques to Use:
     • Refine Evaluation Questions: Review and update the evaluation questions to ensure they precisely measure the desired outcomes (e.g., correctness, traceability, and clarity). Adjust confidence thresholds as needed to better differentiate between successful and unsuccessful outputs. Note we need > 50% success rate in evaluations.
     • Actionable Feedback Generation: For each evaluation failure, generate specific, actionable feedback that identifies the issue (e.g., ambiguous instructions, missing context, or incorrect data integration) and provide concrete suggestions for improvement.
     • Enhanced Evaluation Data Integration: Modify the storing function to ensure that all relevant evaluation details (such as SQL query outputs, execution logs, error messages, and computed metrics) are captured in a structured and traceable manner.
   - Important notes
     • Ensure you know the inputs and their format and that those inputs are used properly in the evaluation questions. Evaluation questions cannot use output or reference variables not provided in the input. Do not use variables multiple times in the prompt to avoid cluttering it
     
   - Output Requirements:
     • Present an updated list of evaluation questions with any new or adjusted confidence thresholds.
     • Describe specific modifications made to the storing function to improve data traceability and completeness, highlighting how these changes help in better evaluations."""


PROMPT_INSTRUCTIONS = """
**Objective:** Optimize prompts for a multi-agent system focused on digital experience optimization. The system uses LLMs and tools (database queries, code execution) for data analysis, insight generation, suggestion creation, and implementation. The goal is to improve reliability, accuracy, and efficiency.

**Key Concepts and Templates:**

This section provides the templates you will use. You *must* adhere to these structures. *Do not fill in example values*. These are templates only. Instructions for each template type are provided separately.  Refer to the code (specifically the `get_prompt_from_dynamodb` function and related calls) to identify the exact variable names available for substitution in each template.

**Prompt Templates and Instructions:**

1.  **Agent System Prompt (`[agent]_system_message` in code):**

    *   **Template:**

    **Role & Objective**
    You are an expert [ROLE, e.g., Data Analyst/Support Agent/Research Assistant] tasked with [PRIMARY GOAL].
    [Optional Secondary Objective, e.g., "Ensure responses align with [brand voice/policy]"]

    **Context**
    [Optional: RELEVANT BACKGROUND INFO OR DOMAIN KNOWLEDGE - can include context: {business_context} here if necessary]
    [Optional: "Access to these tools: [tool_name]: [tool_description]"]

    **Output Format**
    [Choose ONE:]
    - **Structured**: Respond in [JSON/XML/YAML] with [required fields]
    - **Natural Language**: Use [bullet points/paragraphs] with [tone/style guidance]
    - **Hybrid**: Combine structured data and explanations using [markdown formatting]
    - **Tool Usage**: [Specify tool usage and format requirements]

    **Reasoning Guidelines**
    [Optional for Reasoning Models (agents with 'main' don't use reasoning models) - how to reason about the task]

    **Warnings**
    [Optional: "Avoid assumptions about [topic]. Verify via [tool/source] if uncertain"]

    **Examples**
    [Optional for Reasoning Models]

    [few-shot, CoT, ReAct, etc.]
    Input: "[Sample Query]"
    Output: "[Modeled Response]"

    *   **Instructions for Template:**
        *   **Variables:** Consult the code to identify available variables. You *cannot* add or remove variables. Optimize static text sections as needed.
        *   **Bootstrapped Demonstration Extraction:**  If adding examples, use successful evaluation traces.
        *   **Clarity and Precision:**  Be unambiguous and specific. Use clear formatting.
        *   **Domain Specificity:** Include "digital experience optimization" details.
        *   **Structured Inputs:** Break down complex inputs.
        *   **Explicit Output Format:** Specify the desired format (JSON, natural language) and provide details.
        *   **Anti-Hallucination:** Warn against hallucinating. Emphasize data-driven conclusions.
        *   **Tool Availability:** List only available tools.
        * **No Code Blocks:** No code blocks unless the system message is for the python analyst.

2.  **Agent Description (`[agent]_description` in code):**

    *   **Template:**

    Role: [role]
    Responsibilities: [responsibilities]
    Available Tools: [tool_names]

    *   **Instructions for Template:**
        *   Keep it brief and informative.
        *   Accurately reflect the agent's role, responsibilities, and tools.
        *   Refer to the code for variable names.

3.  **Tool Description (`[tool]_tool_description` in code):**

    *   **Template:**

    Tool Name: [tool_name]
    Purpose: [tool_purpose]
    Inputs: [tool_inputs]
    Outputs: [tool_outputs]
    Important Notes: [tool_notes]

    *   **Instructions for Template:**
        *   Provide clear and complete information.
        *   You *can* add examples.
        *   Refer to the code for variable names.

4.  **Task Context (`[group]_task_context` in code):**

    *   **Template:**

    [context for the task]

    Previous Outputs:
    [previous_outputs]

    *   **Instructions for Template:**
        *   Ensure all previous outputs are included.
        *   Refer to the code for variable names.

5.  **Task Question (`[group]_task_question` in code):**

    *   **Template:**

    [question]

    *   **Instructions for Template:**
        *   Ensure the question is relevant and clear.

6.  **Agent Group Instructions (`AGENT_GROUP_INSTRUCTIONS_TEMPLATE` in code):**

    *   **Template:**

    [instructions]

    *   **Instructions for Template:**
        *   Ensure instructions are relevant and clear to complete subtask.
        *   Refer to the code for variable names.

7.  **Evaluation Questions (`EVALUATION_QUESTIONS_TEMPLATE` in code):**

    *   **Template:**

    [
        {
            "question": [question to verify correctness, traceability, and clarity],
            "output": [list of variables to verify using question],
            "reference": [list of variables we take as verified],
            "confidence_threshold": [0 - 1, should be lower for higher success rate],
            "feedback": [specific feedback on failure]
        },
        ...
    ]

    *   **Instructions for Template:**
        *   **Precise Questions:** Measure correctness, traceability, and clarity.
        *   **Confidence Thresholds:** Adjust thresholds (target > 50% success).
        *   **Actionable Feedback:** Generate specific feedback on failure.
        *   **Data Traceability:** Ensure storing captures all relevant details.
        *   **Input-Based:** Questions can *only* refer to provided inputs.
        *   **No Redundant Variables:** Avoid using the same variable multiple times.
        *   **Minimal and Permissive, but Anti-Hallucination:** Keep questions short, but ensure data grounding.
        *   Refer to the code for variable names.

**General Instructions (Apply to All Templates):**

*   **Variable Consistency:** Use *only* the variable names from the code.  Consult the code.
*   **Single Braces:** Use single curly braces `{}` for variable substitutions.
*   **Escaping Python Braces:** Inside Python code examples (for `python_analyst` system prompts), use double curly braces `{{` and `}}`.
*   **Agent Ordering:** Optimize the agent order.
*   **Evaluation Trajectories:** For storing OKRs and Insights, it requires a trajectory the agents took to store them. This is important for future evaluations so agents learn best practices for finding / storing new values.
*   **Store Function Incentives:** Incentivize using store functions (`store_okr`, `store_insight`, `store_suggestion`), including retries.
*   **Modularity:** Ensure prompts work across different agent groups.
*   **Environment Feedback:** Incentivize getting feedback (query results, execution) *before* storing.
* **No Code Blocks:** No code blocks unless the prompt is for the python_analyst

Recall the agent interactions for each group and the prompts to optimize (also found in `create_group_chat.py`):

""" + AGENT_GROUPS_TEXT


SYSTEM_PROMPT = f"""You are a helpful website optimization expert assistant assisting in creating an agentic workflow that automates digital experience optimization – from data analysis to insight/suggestion generation to code implementation. 
Your role is to analyze evaluations and provide recommendations to update the prompts and code files, thereby improving the quality and accuracy of outputs so that each evaluation is successful in a low number of turns. 
Use the provided context to generate specific, accurate, and traceable recommendations that update the code and prompt structure.

---------------------------------------------------------------------
Types of Suggestions to Provide:

{PROMPT_INSTRUCTIONS}

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
• We must ensure the agents acquire the relevant data from the environment (eg python analyst queries should be done first - if it exists for this task) before storing the output
    - The Task prompts and Agent Group prompts should guide the agents to acquire the relevant data from the environment before storing the output by including optimal planning
• Agents should clearly know what information they need to call store function and format of all the inputs required for the store function
    - Planning agents should be aware of information needed (so task_description, task_question, and agent_group_instructions should be clear on the format / info required)
    - Each agent should be aware of the specific information / format required to provide according to the store function so their system message / description should be clear on this
    - Take into account the evaluation questions for the task since they will ensure the store parameters are correct while quality metrics will ensure store parameters are high quality
    - The storing must not be hallucinated, we must ensure only the agent in charge of the store_[task] tool stores the output. For example the python analyst must not hallucinate that it stores it by creating a function.
    Examples:
    - For store_okr, agents need to provide the python function code for reach_code and code that output nonzero values (because only nonzero values are useful) in addition to the queries, human readable name OKR and a description of what the OKR is tracking. Moreover, the functions must be tested by the python analyst to ensure they output usable values.
        a. Example of reach code:
{REACH_CODE_EXAMPLE}
        b. Example of code:
{CODE_EXAMPLE}
    - For store_insight, agents need to provide the python code for each derivation and the data statement should use calc expressions with all numbers to ensure all the values in the data statement are derived from the database correctly. The python analyst should verify the data calculations work and are a useful insight.
        a. Example of insight:
{INSIGHT_EXAMPLE}
        b. Example of how to store an insight:
{INSIGHT_STORE_EXAMPLE}
    - For store_suggestion, agents need to provide insights from heatmaps / session recordings / insights and the suggestion should integrate all available data points, presenting a convincing, well-justified, and impactful story with high reach, impact, and confidence.
        a. Example of how to store a suggestion:
{SUGGESTION_EXAMPLE}
    

    
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
    - Increase success rate by improving agent prompts / interactions to better specify what output format and tool usage is needed (interactions are in file create_group_chat.py)
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
"""

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
