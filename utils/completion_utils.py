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
    [Context of its role in the workflow and where it fits in the task]

    **Available Tools** (optional if tools are available. If tools are available, agents must use them, they cannot output a response without using the tools):
    [Tool Name 1]: [Tool Purpose 1]
    [Tool Name 2]: [Tool Purpose 2]
    ...

    **Output Format**
    [Choose ONE:]
    - **Structured** (only use structured if no tools available, otherwise use tool usage): Respond in [JSON/XML/YAML] with [required fields]
    - **Natural Language**: Use [bullet points/paragraphs] with [tone/style guidance]
    - **Tool Usage**: [Specify format of input (using code file) to tool and tool to use]

    **Reasoning Guidelines**
    [Optional for Reasoning Models (agents with 'main' don't use reasoning models) - how to reason about the task]

    **Rules**
    [Create a list of rules for each agent to ensure they do their task properly. If an agent caused a failure in evaluation, a rule should be made so it doesn't happen again.]
    [Ensure the list of rules ensures each agent has a response according to their responsibilies and never wait for another agent / output an empty response]

    **Examples**
    [Ensure there are examples and demonstrations so the agent understands the format and requirements of output for task success]
    [IMPORTANT: Examples must follow tool and function signatures in the code]

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
        *   **Self-Contained:** We must ensure agents don't fall into useless / harmful loops. This can happen if they ask for information that cannot be provided by agents in the chat or ask help from a team outside - eg a data team (nothing outside the chat is available and there is no human interaction). Agents should be instructed so this doesn't happen.
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
        *   **Confidence Thresholds:** Adjust thresholds so success is greater than 50%, eg reduce them if success is low.
        *   **Only Necessary Questions:** Remove unnecessary questions. The questions should just verify the key parts of the store data are not hallucinated.
        *   **Actionable Feedback:** Generate specific feedback on failure.
        *   **Data Traceability:** Ensure storing captures all relevant details.
        *   **Input-Based:** Questions can *only* refer to provided inputs.
        *   **No Redundant Variables:** Avoid using the same variable multiple times.
        *   **Minimal and Permissive, but Anti-Hallucination:** Keep questions short, but ensure data grounding.
        *   Refer to the code for variable names.

**General Instructions (Apply to All Templates):**

*   **Variable Consistency:** Use *only* the variable names from the code.  Consult the code.
*   **Single Braces:** Use single curly braces `{}` for variable substitutions. There should be at most 1 instance of each variable. Remember that all text of the form {variable} will be replaced by the system, so do not unnecessarily repeat variables.
*   **Escaping Braces:** Inside Python code examples (for `python_analyst` system prompts) and other prompts where we want to represent variables without actually doing substitutions, use double curly braces `{{` and `}}`.
*   **Agent Ordering:** Optimize the agent order.
*   **Evaluation Trajectories:** For storing OKRs and Insights, it requires a trajectory the agents took to store them. This is important for future evaluations so agents learn best practices for finding / storing new values.
*   **Store Function Incentives:** Incentivize using store functions (`store_okr`, `store_insight`, `store_suggestion`), including retries.
*   **Modularity:** Ensure prompts work across different agent groups.
*   **Environment Feedback:** Incentivize getting feedback (query results, execution) *before* storing.
*   **Standalone Prompts:** Do not wrap outputs / system messages in blocks (eg do not prepend or append ```) or add additional text. These prompts are used as-is in the system. Also do not use code blocks (eg ```text ```, ```json ```, ```python ```, etc) unless it is for the python_analyst as python code block examples
* **No Code Blocks:** No code blocks unless the prompt is for the python_analyst

Recall the agent interactions for each group and the prompts to optimize (also found in `create_group_chat.py`):

**Prompt Specific Instructions:**

*   **python_analyst, okr_python_analyst:**

All python analysts should be asked to define this at the start of every code block:

```python
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from functions import run_sitewiz_query 
from typing import TypedDict, List, Tuple

# Get yesterday's date as end_date
end_date = (datetime.datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
# Get date 6 days before end_date as start_date
start_date = (datetime.datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

stream_key = "{stream_key}" # This must be defined correctly using the variable substitution

# comments to help the python analyst output the correct data
# define and run query with run_sitewiz_query
# eg
# query = f\"\"\"
# SELECT * FROM table
# WHERE date BETWEEN '{{start_date}}' AND '{{end_date}}'
# AND stream_key = '{{stream_key}}' # We use double curly braces because when compiled the system will turn it into single curly braces and they will be replaced properly in the query
# LIMIT 10
# \"\"\"
# data = run_sitewiz_query(query)


# print results
# print(data)
```

- Ensure the python analyst hardcodes stream key is hard coded in the query or as a variable
- The only role of the python analyst is to create and run queries using the given data, stream key, and timestamps to answer a data question to the best of their ability. They should not be asked to store data or hallucinate data. They should only be asked to output the results of the query.
- It should print as much information as possible (but limit 10 to not overflow) to find where there is data available.
- Include the variable {function_details} in the prompt so the python analyst knows the schema of the database
- There is no need for .replace since the fstring will replace the variables correctly
- Intermediate results must be printed so we can see what works and what doesn't

*   **okr_questions**:
- We want verify the python code is not hallucinated and the data comes from the database
- We want the OKR to be relatively unique
- Simplify questions to ensure success rate > 50%

*   **insight_questions**:
- We want verify the python code is not hallucinated and the insight data output comes from the database
- We want the insight to be relatively unique
- Simplify questions to ensure success rate > 50%

**Ideal Flow per task:**

1.  **OKR task:**
    -   Insights Behavioral Analyst: Finds directions for finding OKR since it has access to most visited urls and website heatmaps
    -   Python Analyst: Creates python blocks to query the data. Once reach code and okr code is found, send to OKR store group to create / store the OKR
    -   OKR store agent: Creates / formats an OKR, then stores it using the store_okr tool, it must format the input correctly.

2.  **Insight task:**
    -   Insights Behavioral Analyst: Finds directions for finding Insight since it has access to most visited urls and website heatmaps
    -   Python Analyst: Creates python blocks to query the data. Once insight code is found, send to Insight Analyst group to create / store the Insight
    -   Insight Analyst group / Insights Analyst: Creates / formats an Insight and stores it, it must format the input correctly.

3.  **Suggestion task:**
    -   Behavioral Analyst: Finds heatmaps and session recordings to find user behavior hypotheses based on insight
    -   UX Researcher: Finds UX research to back up hypotheses and find how to implement it
    -   Suggestions Analyst: Creates / formats a suggestion, then stores it using the store_suggestion tool, it must format the input correctly.

4.  **Design task:**
    -   Web Agent: Finds locations to implement the suggestion and find if it is already implemented. Once it finds evidence for either way, sends to the design group
    -   Design Agent: Creates a design for the suggestion and stores it using store_design, it must format the input correctly.

5.  **Code task:**
    -   Website Developer: Implements the design and sends to the Website Save group
    -   Website Get Save Agent: Stores the code using store_code, it must format the input correctly.

Notes for ideal flow:
-   The flow specified shows the ideal direction though it will likely have retries and back/forth between agents. 
-   The flow should be optimized to reduce the number of turns to get a successful output.
-   Each agent should execute the tools they have available and should not have an empty response. They should not wait for other agents responses before executing their tools. Rules should be minimal to ensure they do their job correctly.

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
    2. Success Rate should be higher than 50% - do this by ensuring agents output useful responses and making evaluation questions more permissive
    3. Output quality should be as high as possible
    4. The number of turns to get a successful output should be as low as possible
• We must ensure the agents acquire the relevant data from the environment (eg python analyst queries should be done first - if it exists for this task) before storing the output
    - The Task prompts and Agent Group prompts should guide the agents to acquire the relevant data from the environment before storing the output by including optimal planning
    - Optimal planning ensures that the agents that fetch the necessary data from environment go first with a plan (eg python analyst, behavioral analyst, etc.)
    - If store_[task] tool fails, feedback should show what information is needed for a successful storage
    - Primarily update the task and group prompts to accomplish this in addition to the agent prompts
• Agents should clearly know what information they need to call store function and format of all the inputs required for the store function
    - Planning agents should be aware of information needed (so task_description, task_question, and agent_group_instructions should be clear on the format / info required)
    - Each agent should be aware of the specific information / format required to provide according to the store function so their system message / description should be clear on this
    - Take into account the evaluation questions for the task since they will ensure the store parameters are correct while quality metrics will ensure store parameters are high quality
    - The storing must not be hallucinated, we must ensure only the agent in charge of the store_[task] tool stores the output. For example the python analyst must not hallucinate that it stores it by creating a function.
    Examples:
    - For store_okr, parameters include the python function code for reach_code and code that output nonzero values (because only nonzero values are useful) in addition to the queries, human readable name OKR and a description of what the OKR is tracking.
        a. Example of reach code:
{REACH_CODE_EXAMPLE}
        b. Example of code:
{CODE_EXAMPLE}
    - For store_insight, parameters include the python code for each derivation and the data statement should use calc expressions with all numbers to ensure all the values in the data statement are derived from the database correctly. 
        a. Example of insight:
{INSIGHT_EXAMPLE}
        b. Example of how to store an insight:
{INSIGHT_STORE_EXAMPLE}
    - For store_suggestion, parameters include insights from heatmaps / session recordings / insights and the suggestion should integrate all available data points, presenting a convincing, well-justified, and impactful story with high reach, impact, and confidence.
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
