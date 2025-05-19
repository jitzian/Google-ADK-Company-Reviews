import requests
from typing import List, Optional
from .IRepositories import IRepositories
from ...constants.global_constants import github_api_url
from ..model.GitHubRepo import GitHubRepo

class RepositoriesImpl(IRepositories):
    def __init__(self, repositories):
        self.repositories = repositories

    def get_repositories(self, user: str) -> Optional[List[GitHubRepo]]:
        """
        Fetches repositories for a given GitHub user from the GitHub API.
        Returns a list of GitHubRepo objects or None if there's an error.
        """
        try:
            # Construct the API URL
            url = f"{github_api_url}/{user}/repos"
            
            # Make the GET request
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the JSON response and convert to GitHubRepo objects
            repositories = [GitHubRepo.from_dict(repo_data) for repo_data in response.json()]
            return repositories
        except requests.exceptions.RequestException as e:
            # Handle any request errors
            print(f"Error fetching repositories for user {user}: {e}")
            return None