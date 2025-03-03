import os
import json
import time
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from utils.logging_utils import log_debug, log_error, measure_time


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
    
# Add code file cache
_code_file_cache = {}

@measure_time
async def fetch_and_cache_code_files(token=None, repo="SitewizAI/sitewiz", refresh=False):
    """
    Fetch and cache code files from GitHub repository.
    
    Args:
        token: GitHub API token (if None, uses environment variable)
        repo: Repository name in format "owner/repo"
        refresh: Whether to refresh the cache
        
    Returns:
        Dictionary of file paths to file contents
    """
    global _code_file_cache
    
    # Return from cache if available and not refreshing
    if _code_file_cache and not refresh:
        log_debug(f"Using cached code files ({len(_code_file_cache)} files)")
        return _code_file_cache
        
    # Use token from environment if not provided
    if not token:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            log_error("No GitHub token available")
            return {}
    
    log_debug(f"Fetching code files from {repo}")
    
    # Fetch files from multiple directories in parallel
    target_paths = [
        "backend/agents",
        "backend/lib",
        "backend/models",
        "backend/tools"
    ]
    
    all_files = {}
    
    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for path in target_paths:
                tasks.append(get_github_files_async(token, repo=repo, target_path=path))
                
            results = await asyncio.gather(*tasks)
            
            # Combine all results
            for path_results in results:
                for file_info in path_results:
                    all_files[file_info['file']['path']] = file_info['content']
                    
        # Update cache
        _code_file_cache = all_files
        log_debug(f"Cached {len(_code_file_cache)} code files")
        return all_files
    except Exception as e:
        log_error(f"Error fetching code files", e)
        return {}
