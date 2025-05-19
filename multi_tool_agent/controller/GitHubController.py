from typing import List, Optional
from ..domain.repository.IRepositories import IRepositories
from ..domain.model.GitHubRepo import GitHubRepo

class GitHubController:
    def __init__(self, repository: IRepositories):
        self._repository = repository

    def get_user_repositories(self, username: str) -> Optional[List[GitHubRepo]]:
        """
        Fetches and returns a list of repositories for the given GitHub username.
        
        Args:
            username (str): The GitHub username to fetch repositories for
            
        Returns:
            Optional[List[GitHubRepo]]: A list of GitHubRepo objects if successful, None otherwise
        """
        return self._repository.get_repositories(username)
    
    def get_repository_languages(self, repositories: List[GitHubRepo]) -> dict:
        """
        Analyzes the languages used across all repositories.
        
        Args:
            repositories (List[GitHubRepo]): List of repositories to analyze
            
        Returns:
            dict: A dictionary with language counts
        """
        language_counts = {}
        for repo in repositories:
            if repo.language:
                language_counts[repo.language] = language_counts.get(repo.language, 0) + 1
        return language_counts
