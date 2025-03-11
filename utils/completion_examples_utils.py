REACH_CODE_EXAMPLE = '''
# you must use these exact imports in your code
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from functions import run_sitewiz_query 
from typing import TypedDict, List, Tuple

class ReachOutput(TypedDict):
    Description: str
    start_date: str
    end_date: str
    values: List[Tuple[str, float]]

stream_key = '{stream_key}'  # THIS MUST BE DEFINED AND USED IN THE QUERIES

# Get yesterday's date as end_date
end_date = (datetime.datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
# Get date 6 days before end_date as start_date
start_date = (datetime.datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

def calculate_reach(start_date: str, end_date: str) -> ReachOutput:   # do not change this function signature or ReachOutput
    # Calculate total sessions per day using the materialized date column in session_recordings.
    sql = f"""
    SELECT 
        sr.date AS date,
        COUNT(DISTINCT s.session_id) AS total_sessions
    FROM sessions s
    JOIN session_recordings sr ON s.session_id = sr.session_id
    WHERE s.stream_key = '{{stream_key}}'
      AND sr.date BETWEEN '{{start_date}}' AND '{{end_date}}'
    GROUP BY sr.date
    ORDER BY sr.date;
    """
    results = run_sitewiz_query(sql)
    
    # Convert query results to a dictionary for lookup by date
    reach_dict = {{ row[0]: row[1] for row in results }}
    
    # Build a list of dates between start_date and end_date (inclusive)
    date_range = pd.date_range(start=start_date, end=end_date)
    values = []
    for dt in date_range:
        date_str = dt.strftime("%Y-%m-%d")
        total_sessions = reach_dict.get(date_str, 0)
        values.append((date_str, total_sessions))
    
    return {{
        "Description": "Daily total sessions grouped by date from sessions joined with session_recordings.",
        "start_date": start_date,
        "end_date": end_date,
        "values": values
    }}

output = calculate_reach(start_date, end_date)
print("Calculate Reach Output:")
print(output)
'''

CODE_EXAMPLE = '''
# you must use these exact imports in your code, you cannot add, remove, or change any imports
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from functions import run_sitewiz_query
from typing import TypedDict, List, Tuple

class MetricOutput(TypedDict):
    Metric: str
    Description: str
    start_date: str
    end_date: str
    values: List[Tuple[str, float]]

stream_key = '{stream_key}'  # THIS MUST BE DEFINED AND USED IN THE QUERIES

# Get yesterday's date as end_date
end_date = (datetime.datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
# Get date 6 days before end_date as start_date
start_date = (datetime.datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

def calculate_metrics(start_date: str, end_date: str) -> MetricOutput:  # do not change this function signature or MetricOutput
    # Calculate daily signup conversion rate as (signup visits / total sessions)
    # For signup visits, join funnels with session_recordings to use the materialized date column.
    sql_signup = f"""
    SELECT 
        sr.date AS date,
        COUNT(DISTINCT f.session_id) AS signup_visits
    FROM funnels f
    JOIN session_recordings sr ON f.session_id = sr.session_id
    WHERE f.base_url = '.../signup'
      AND sr.stream_key = '{{stream_key}}'
      AND sr.date BETWEEN '{{start_date}}' AND '{{end_date}}'
    GROUP BY sr.date
    ORDER BY sr.date;
    """
    results_signup = run_sitewiz_query(sql_signup)

    # Total sessions are obtained from sessions joined with session_recordings,
    # using the materialized date column (sr.date) as the main timestamp reference.
    sql_total = f"""
    SELECT 
        sr.date AS date,
        COUNT(DISTINCT s.session_id) AS total_sessions
    FROM sessions s
    JOIN session_recordings sr ON s.session_id = sr.session_id
    WHERE sr.stream_key = '{{stream_key}}'
      AND sr.date BETWEEN '{{start_date}}' AND '{{end_date}}'
    GROUP BY sr.date
    ORDER BY sr.date;
    """
    results_total = run_sitewiz_query(sql_total)

    # Convert query results to dictionaries for lookup by date
    signup_dict = {{row[0]: row[1] for row in results_signup}}
    total_dict = {{row[0]: row[1] for row in results_total}}

    # Build a list of dates between start_date and end_date (inclusive)
    date_range = pd.date_range(start=start_date, end=end_date)
    values = []
    for dt in date_range:
        date_str = dt.strftime("%Y-%m-%d")
        signup_count = signup_dict.get(date_str, 0)
        total_count = total_dict.get(date_str, 0)
        conversion_rate = signup_count / total_count if total_count > 0 else 0.0
        values.append((date_str, conversion_rate))

    return {{
        "Metric": "signup_conversion",
        "Description": "Daily signup conversion rate calculated as signup visits from funnels (exact match on '.../signup') over total sessions from sessions, grouped by the materialized date.",
        "start_date": start_date,
        "end_date": end_date,
        "values": values
    }}

# print results for testing
print(calculate_metrics(start_date, end_date))
'''

INSIGHT_EXAMPLE = """Example of a unique insight:

The following is an example of a well structured, high quality insight that meets the following requirements:
1. The python code calculation of the current value of the OKR the insight is about
2. The python code calculation of the target value of the OKR the insight is about (or a benchmark, but this is less preferable). The goal must be a number based off another segment or benchmark and there must be a reason why this goal makes sense.
3. The python code calculation of the reach of the insight (eg fraction of audience it affects)
4. Python code calculations for all values in the insight

#### Insight n: Low Engagement with Key CTAs on Authority Links Page for Mobile Users

- Data Statement: Mobile users show only a 0.5% click-through rate (CTR) on the "Get Started" CTA on the Authority Links page, while desktop users achieve a 2.5% CTR. This is a critical performance gap because mobile accounts for 40% of our traffic. # though must use calc expressions with variables derived from data instead
- Problem Statement: The low mobile CTR suggests the primary CTA is not effectively engaging mobile users. We believe this is because the button is below the fold or is not thumb friendly.
- Hypothesis: By making the "Get Started" button more visible above the fold and thumb friendly, we can significantly improve mobile CTR, boosting conversions and revenue.
- Business Objective: Enhance click-through rates (CTR)
- Prioritization:
    - Reach: High (40% of traffic)
    - Frequency: Daily (ongoing issue)
    - Severity: 9 (Directly impacts revenue; significantly underperforming)
    - Severity reasoning: Captures a substantial portion of traffic and affects the core KPI of CTR.
    - Confidence: 0.9
    - Confidence reasoning: Inaccuracies in the database are assumed to be random.

- Derivation:
```python
import pandas as pd
from functions import run_sitewiz_query

# Define the stream key
stream_key = "your_stream_key"  # Replace with the actual stream key
# Define the time range
end_time = int(datetime.now().timestamp() * 1000)
start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

# Function to calculate the click-through rate for a given device type
def calculate_ctr(device_form: int, url: str) -> float:
    query = f'''
        SELECT 
            COUNT(DISTINCT h.session_id) AS click_sessions,
            COUNT(DISTINCT s.session_id) AS total_sessions
        FROM heatmaps h
        JOIN sessions s ON h.session_id = s.session_id
        WHERE h.stream_key = '{stream_key}'
          AND h.url = '{url}'
          AND h.type = 1
          AND s.device_form = {device_form}
          AND h.timestamp >= {start_time}
          AND h.timestamp <= {end_time}
    '''
    results = run_sitewiz_query(query, f"Query for device form {device_form}")
    if results and results[0][1] > 0:
        click_sessions = results[0][0]
        total_sessions = results[0][1]
        return click_sessions / total_sessions
    else:
        return 0  # Return 0 if there are no sessions to avoid division by zero

# Calculate CTR for desktop and mobile
desktop_ctr = calculate_ctr(0, "https://loganix.com/authority-links/")
mobile_ctr = calculate_ctr(2, "https://loganix.com/authority-links/")

# Print Results for storage by the insight code analyst
print("Desktop CTR:", desktop_ctr)
print("Mobile CTR:", mobile_ctr)

# Reach calculation code (assuming number of sessions is tracked)
def calculate_reach(device_form: int) -> float:
    query = f'''
        SELECT 
            COUNT(DISTINCT s.session_id) AS total_sessions
        FROM sessions s
        JOIN session_recordings sr ON s.session_id = sr.session_id
        WHERE s.stream_key = '{stream_key}'
          AND s.device_form = {device_form}
          AND sr.date BETWEEN '{start_date}' AND '{end_time}'
    '''
    results = run_sitewiz_query(query, f"Query to get reach for device form {device_form}")
    if results and results[0][0] is not None:
        reach = results[0][0]
        print(f"Reach for device {{device_form}}: {{reach}}")
        return reach
    else:
        reach = 0
        print(f"No reach found for {device_form}, check query")
        return reach

desktop_reach = calculate_reach(0)
mobile_reach = calculate_reach(2)
# Calculate Reach
mobile_fraction = mobile_reach / (desktop_reach + mobile_reach)
print("Mobile Fraction:", mobile_fraction)"""


"""def store_insight(
        insight_data: Annotated[Insight, "Insight data to be stored; must be connected to a verified OKR"],
        okr_name: Annotated[str, "The name of the OKR this insight is meant to improve"],
        trajectory: Annotated[str, "A short description of the trajectory the agents took to get the insight"]
    ) -> Annotated[tuple[str, bool], "Result message and success boolean"]:"""

INSIGHT_STORE_EXAMPLE = f"""store_insight(
    okr_name="Homepage Engagement", 
    insight_data={{
        "data_statement": "Users who click on the {{items_xpath}} on the homepage are {{calc(({{click_thank_you_sessions}}/{{click_sessions}} - ({{thank_you_sessions}}-{{click_thank_you_sessions}})/({{home_sessions}}-{{click_sessions}})) / (({{thank_you_sessions}}-{{click_thank_you_sessions}})/({{home_sessions}}-{{click_sessions}})) * 100)}}% more likely to visit the Thank You page than users who don't. However, the element is only engaged by {{calc({{scroll_items_xpath_sessions}} / {{home_sessions}} * 100)}}% of users.",
        
        "problem_statement": "Low engagement with the CTA hinders conversion. The issue is evident in 40% of sessions.",
        
        "business_objective": "Increase purchase conversions by improving the visibility and engagement of key CTAs on the homepage.",
        
        "hypothesis": "By repositioning and emphasizing the See Items button, we expect to increase its engagement and drive higher conversions.",
        
        "frequency": 40, 
        "severity": 5,
        "severity_reasoning": "The CTA is a primary conversion driver, so low engagement has a severe impact.",
        "confidence": 0.9,
        "confidence_reasoning": "Data is based on a large sample of clickstream and session recording analytics.",
        
        "derivation": [
            {{
                "variable_name": "click_thank_you_sessions",
                "value": 49,
                "derivation": "import pandas as pd\nfrom functions import run_sitewiz_query\nquery = \"SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND f.base_url = '...' AND s.device_form = 2 AND f.timestamp BETWEEN 17... AND 17...\"\nresult = run_sitewiz_query(query, 'click_thank_you_sessions')\nprint(result)",
                "description": "# sessions where user clicks the CTA and visits the Thank You page (using funnels)"
            }},
            {{
                "variable_name": "click_sessions",
                "value": 320,
                "derivation": "import pandas as pd\nfrom functions import run_sitewiz_query\nquery = \"SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND s.device_form = 2 AND f.timestamp BETWEEN 17... AND 17...\"\nresult = run_sitewiz_query(query, 'click_sessions')\nprint(result)",
                "description": "# sessions with any funnel event (clicks) for the CTA"
            }},
            {{
                "variable_name": "thank_you_sessions",
                "value": 54,
                "derivation": "import pandas as pd\nfrom functions import run_sitewiz_query\nquery = \"SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND f.base_url = '...'\"\nresult = run_sitewiz_query(query, 'thank_you_sessions')\nprint(result)",
                "description": "# sessions where user visits the Thank You page (via funnels)"
            }},
            {{
                "variable_name": "home_sessions",
                "value": 523,
                "derivation": "import pandas as pd\nfrom functions import run_sitewiz_query\nquery = \"SELECT COUNT(*) FROM sessions s JOIN funnels f ON s.session_id = f.session_id WHERE s.stream_key = '...' AND f.base_url = '...' AND f.timestamp BETWEEN 17... AND 173...\"\nresult = run_sitewiz_query(query, 'home_sessions')\nprint(result)",
                "description": "# sessions for the homepage (using funnels)"
            }},
            {{
                "variable_name": "scroll_items_xpath_sessions",
                "value": 82,
                "derivation": "import pandas as pd\nfrom functions import run_sitewiz_query\nquery = \"SELECT COUNT(*) FROM heatmaps h JOIN funnels f ON h.session_id = f.session_id WHERE h.stream_key = '...' AND f.base_url = '...' AND h.timestamp BETWEEN 17... AND 17... AND h.type = 2 AND h.xpath IS NOT NULL AND h.xpath <> ''\"\nresult = run_sitewiz_query(query, 'scroll_items_xpath_sessions')\nprint(result)",
                "description": "# sessions where user scrolls past the button (filtered via funnels)"
            }}
        ],
        
        "variables": [
            {{
                "variable_name": "items_xpath",
                "readable": "See Items Button",
                "tooltip": "Technical identifier: xpath='//*[@id=\"items\"/...]'"
            }}
        ]
    }},
    "trajectory": "Heatmap of homepage -> Query of clicks on Items button -> Query on page visits after clicking Items button -> Query on page visits after scrolling past Items button -> Insight creation"
)
"""

SUGGESTION_EXAMPLE = """store_suggestion(suggestion={"Shortened": [
        {
          "type": "header",
          "text": [This should be the full 1 sentence action statement to maintain convincibility. Do not shorten this despite the key name.]
        }
      ],
      "Expanded": [
        {
          "type": "text",
          "header": "Expected Outcomes",
          "text": "A/B tests from [sources, you must name the sources] have driven an **x%%** increase in blah. We can achieve **y%** ..." // Any text keys should be in markdown format (in objects with type text). Make sure to use correct markdown to make the text more readable. Use benchmarks from similar experiments to support the expected outcome.
        },
        {
          "type": "text",
          "header": "Visual Details",
          "text": "Blah should be positioned blah with blah attributes." // use markdown format to make more readable
        },
        {
          "type": "text",
          "header": "Sitewiz's ICE Prioritization",
          "text": "- **Impact**: blah – High/Medium/Low (Reason) 
- **Confidence**: blah – High/Medium/Low (Reason) 
- **Ease**: blah – High/Medium/Low (Reason) 
- **Time to Implement**: blah days"
        },
        {
          "type": "text",
          "header": "Alternatives Considered",
          "text": "blah blah blah."  // use markdown format to make more readable
        }
      ],
      "Detailed": [],
      "Insights": [ // this should be found from the data insight
          {
            "text": "Key insight from the data statement(s) that supports the recommendation. - this must be readable and encapsulate the why this works",
            "data": [
              {
                "type": "Analytics",
                "name": 1-2 word describing analytics,
                "key": "[exact timestamp of the insight]",
                "explanation": "[how data statement connects to the main insight text]" // this should be human readable
              },
            ]
          }
          {
            "text": "Heatmap and Session Recordings suggest this gap is happening due to ... - should encapsulate the why this works",
            "data": [
              {
                "type": "Heatmap",
                "name": 1-2 word describing url / type of heatmap,
                "key": "clickmaps/xxxx/xxxx_heatmap.png", # this must be the heatmap id from the behavioral analyst, we need th efull heatmap id ending in .png
                "explanation": "bla blah why heatmap supports insight" // remember not to specify any specific keys, any text, explanation, or header key should be human readable
              },
              {
                "type": "Heatmap",
                "name": 1-2 word describing url / type of heatmap,
                "key": "clickmaps/xxxx/xxxx_heatmap.png", # this must be the heatmap id from the behavioral analyst, we need th efull heatmap id ending in .png
                "explanation": "bla blah why heatmap supports insight" // remember not to specify any specific keys, any text, explanation, or header key should be human readable
              },
              {
                "type": "Session Recording",
                "name": 1-2 word describing user behavior,
                "key": "xxxxx-xxxx-xxxx-xxxx-xxxxxxxx", # this  should be the exact session id from the behavioral analyst
                      "explanation": "bla blah why session recording supports insight"
              },
            ]
          }

      ],
      "Tags": [
        {
          "type": "Page",
          "Icon": "Page", # this must be "Page"
          "Value": "Homepage",
          "Tooltip": "[url of page]"

        },
        {
          "type": "Metric",
          "Icon": "CVR", # this must be an abbrevie
          "Value": "x%", // must be a single number, not a range, if it was a range, use the average
          "Tooltip": "Conversion Rate"
        }
      ],
      InsightConnectionTimestamp: "17xxxxxxx...", // this should be the timestamp int (casted to a string) of the insight stored used to connect the insight to the suggestion
    })"""