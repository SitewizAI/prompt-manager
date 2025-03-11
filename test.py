from utils.validation_utils import get_document_structure, find_prompt_usage_in_code
from utils.prompt_utils import validate_prompt_parameters, get_prompt_expected_parameters, AGENT_TOOLS, AGENT_GROUPS, AGENT_GROUPS_TEXT
from utils.completion_examples_utils import INSIGHT_STORE_EXAMPLE
from utils.get_weekly_stored import get_weekly_storage_stats, format_storage_stats_table
# print(get_document_structure("to_be_implemented_questions"))
# print(validate_prompt_parameters("insights_task_context", """Here are the other OKRs stored, choose one as a base for the insight (use its okr_name when calling store_insight):
# {{okrs}}
# Here are the other insights stored, DO NOT REPEAT THEM. WE WANT UNIQUE INSIGHTS
# {{insights}}
# Only choose 1 OKR to use as a base. Prioritize choosing an OKR where the insight count is low and has a greater impact on the business.

# Make sure at least [x]%% sessions are being tracked on the site to consider the insight.


# Example 1: Test for total sessions on the form. If > 100, focus on form.
# Example 2: Then if %% mobile visits is > 70, focus on mobile.
# Example 3: Choose OKR Enhance_User_Engagement_and_Conversion_Efficiency because there are 0 insights.
# Example 4: Choose OKR Click_Interaction_Analysis because it can directly impact user behavior on the website.
# Example 5: Choose OKR Enhance User Engagement and Streamline Navigation if there are navigational elements on a site that can be easily fixed with design updates.


# IMPORTANT: The task is not complete unless the Insight is explicitly stored by the insights_analyst agent. Be careful of circular reasoning. The data statement must be proven using data and not derived from previous assumptions. The severity should be as high as possible given the data available. Always use the most recent OKR.

# The data must be high quality in order to find an insight that can be stored. If the data available does not have a reasonable quality in terms of total sessions on the site, lack of heatmap data, or the values are all 0s or close to 0, please ask the research analyst to find another OKR.
# """))

#print(get_prompt_expected_parameters('suggestions_user_proxy_system_message'))
# stats = get_weekly_storage_stats()
# print(format_storage_stats_table(stats))

print(validate_prompt_parameters("insights_analyst_group_instructions", '''```python
"""
These are instructions provided to the agent group to guide them in completing the subtask of storing an insight, ensuring the insight is data-driven and well-structured.
"""

TASK: """Given the available context, your task is to collaboratively utilize the available tools and agent expertise to prepare and store a high-quality, data-backed insight. Follow this multi-step process:

1. **Data Acquisition and Validation:**
    - The `python_analyst` and `python_analyst_interpreter` must first use the functions to query the database for data and create the calculations that will be used as variables for the output.
    - The data values from the python analyst should all be nonzero. If any value is zero, do not store, and try again
    - The insights behavioral agent should only be used after the python_analyst has the relevant data to use.

2. **Insight Formulation:**
    - The `insights_analyst` and `insights_analyst_code` must be able to structure the data as an insight.
    - Use the `insights_behavioral_analyst` to gather additional contextual information using the `get_heatmap`, `get_element`, and `get_top_pages` tools if necessary to enhance the insight, but only after the data analysis is done by the python agents.

3. **Insight Storage:**
   - The insight must only be stored by the `insights_user_proxy` agent and it should call `store_insight`
   - The `store_insight` function requires:
        - `okr_name`: The name of the relevant OKR.
        - `insight_data`: A dictionary with keys for `data_statement`, `problem_statement`, `business_objective`, `hypothesis`, `frequency`, `severity`, `severity_reasoning`, `confidence`, `confidence_reasoning`, and `derivation`.
            - **data_statement**: This must include a clear statement of the insight, incorporating calculated values from the data. Use `{{calc(...)}}` expressions for all numerically derived values.
            - **derivation**: A list of dictionaries, each detailing the calculation of a variable used in the `data_statement`. Each dictionary must have keys for `variable_name`, `value` (the code, must return a single value), and `description`.  The Python code must be a complete, self-contained, executable block. The functions are available to call.
            - **Example Derivation**:
                ```python
                [
                    {
                        "variable_name": "click_sessions",
                        "value": """import pandas as pd
    from functions import run_sitewiz_query
    query = "SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = 'your_stream_key' AND f.base_url LIKE '%your_funnel_url%' AND s.device_form = 2 AND f.timestamp BETWEEN 1704448353214 AND 1704448653214"
    result = run_sitewiz_query(query, 'click_sessions')
    print(result)""",
                        "description": "# sessions where user clicks the CTA"
                    },
                    ...
                ]

                ```
    - **trajectory** string description for the steps taken to get the insight. This must be one sentence and must be human readable, describing the thought process of the agent.

4. **Ensure the following:**
    - All numerical values in the `data_statement` are derived from the `derivation` code, enclosed in `{{calc(...)}}`. This ensures traceability back to the raw data.
    - The insight is novel and distinct from previously stored insights.
    - The insight is directly linked to a verified OKR and addresses a significant business problem or opportunity.
    - The insight and all derivations are validated by the `python_analyst` for correctness before storing.

5. **Conversation Control:**
    - Maintain a structured conversation flow. The `insights_user_proxy` should not attempt to store the insight until directed by `insights_analyst`.
    - The final `store_insight` call *must* come from the `insights_user_proxy`.
    - The trajectory value for `store_insight` must be one complete human readable sentence.

**IMPORTANT**:
- Insights must be fully traceable to database values. All variables used in the data_statement must have a corresponding entry in the derivation array and must be wrapped in a `{{calc()}}` block.
- Do not hallucinate any data that should come from the database.
- Prioritize insights that are significant (high severity/frequency/risk) and actionable, supported by robust data.
- Ensure that the python code provided in `derivation` is self-contained, uses the allowed imports, and follows the required structure to return a single value.
- The `insights_user_proxy` *must* call the `store_insight` function with the correctly formatted parameters. Do *not* create a function definition.

EXAMPLE:
Here is an example of how to format the final store insight call:

```tool_code
store_insight(
    okr_name="Homepage Engagement", 
    insight_data={
        "data_statement": "Users who click on the {items_xpath} on the homepage are {calc(({click_thank_you_sessions}/{click_sessions} - ({thank_you_sessions}-{click_thank_you_sessions})/({home_sessions}-{click_sessions})) / (({thank_you_sessions}-{click_thank_you_sessions})/({home_sessions}-{click_sessions})) * 100)}% more likely to visit the Thank You page than users who don't. However, the element is only engaged by {calc({scroll_items_xpath_sessions} / {home_sessions} * 100)}% of users.",
        
        "problem_statement": "Low engagement with the CTA hinders conversion. The issue is evident in 40% of sessions.",
        
        "business_objective": "Increase purchase conversions by improving the visibility and engagement of key CTAs on the homepage.",
        
        "hypothesis": "By repositioning and emphasizing the See Items button, we expect to increase its engagement and drive higher conversions.",
        
        "frequency": 40, 
        "severity": 5,
        "severity_reasoning": "The CTA is a primary conversion driver, so low engagement has a severe impact.",
        "confidence": 0.9,
        "confidence_reasoning": "Data is based on a large sample of clickstream and session recording analytics.",
        
        "derivation": [
            {
                "variable_name": "click_thank_you_sessions",
                "value": "import pandas as pd
from functions import run_sitewiz_query
query = "SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND f.base_url = '...' AND s.device_form = 2 AND f.timestamp BETWEEN 17... AND 17..."
result = run_sitewiz_query(query, 'click_thank_you_sessions')
print(result)",
                "description": "# sessions where user clicks the CTA and visits the Thank You page (using funnels)"
            },
            {
                "variable_name": "click_sessions",
                "value": "import pandas as pd
from functions import run_sitewiz_query
query = "SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND s.device_form = 2 AND f.timestamp BETWEEN 17... AND 17..."
result = run_sitewiz_query(query, 'click_sessions')
print(result)",
                "description": "# sessions with any funnel event (clicks) for the CTA"
            },
            {
                "variable_name": "thank_you_sessions",
                "value": "import pandas as pd
from functions import run_sitewiz_query
query = "SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND f.base_url = '...'"
result = run_sitewiz_query(query, 'thank_you_sessions')
print(result)",
                "description": "# sessions where user visits the Thank You page (via funnels)"
            },
            {
                "variable_name": "home_sessions",
                "value": "import pandas as pd
from functions import run_sitewiz_query
query = "SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND f.base_url = '...' AND f.timestamp BETWEEN 17... AND 173..."
result = run_sitewiz_query(query, 'home_sessions')
print(result)",
                "description": "# sessions for the homepage (using funnels)"
            },
            {
                "variable_name": "scroll_items_xpath_sessions",
                "value": "import pandas as pd
from functions import run_sitewiz_query
query = "SELECT COUNT(*) FROM heatmaps h JOIN funnels f ON h.session_id = f.session_id WHERE h.stream_key = '...' AND f.base_url = '...' AND h.timestamp BETWEEN 17... AND 17... AND h.type = 2 AND h.xpath IS NOT NULL AND h.xpath <> ''"
result = run_sitewiz_query(query, 'scroll_items_xpath_sessions')
print(result)",
                "description": "# sessions where user scrolls past the button (filtered via funnels)"
            }
        ],
        
        "variables": [
            {
                "variable_name": "items_xpath",
                "readable": "See Items Button",
                "tooltip": "Technical identifier: xpath='//*[@id="items"/...]'"
            }
        ]
    },
    "trajectory": "Heatmap of homepage -> Query of clicks on Items button -> Query on page visits after clicking Items button -> Query on page visits after scrolling past Items button -> Insight creation"
)
```
"""
```
'''))