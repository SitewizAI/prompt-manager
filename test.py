from utils.validation_utils import get_document_structure, find_prompt_usage_in_code
from utils.prompt_utils import validate_prompt_parameters, get_prompt_expected_parameters, get_top_prompt_content, AGENT_TOOLS, AGENT_GROUPS, AGENT_GROUPS_TEXT, update_prompt
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

#print(get_prompt_expected_parameters('okr_store_agent_system_message'))
# print(AGENT_GROUPS_TEXT)

# print(validate_prompt_parameters("okr_python_analyst_description", """{
#     "role": "Python Analyst",
#     "responsibilities": [
#         "Write and execute Python code to retrieve and analyze data using the `run_sitewiz_query` function. Only this agent can use `run_sitewiz_query`.",
#         "Validate data existence by querying for non-zero session, funnel event, and heatmap event counts.",
#         "Explore data to identify potential OKR metrics that show variability and business alignment.",
#         "Provide the Python code for `calculate_metrics` and `calculate_reach` functions."
#     ],
#     "available_tools": [
#         "run_sitewiz_query"
#     ]
# }"""))

print(get_top_prompt_content("insights_task_question", eval_type="insights"))

# stats = get_weekly_storage_stats()
# print(format_storage_stats_table(stats))
