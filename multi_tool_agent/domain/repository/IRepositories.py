from abc import ABC, abstractmethod
from typing import List, Optional
from ..model.GitHubRepo import GitHubRepo

class IRepositories(ABC):
    @abstractmethod
    def get_repositories(self, user: str) -> Optional[List[GitHubRepo]]:
        """
        Fetches repositories for a given GitHub user.
        
        Args:
            user (str): GitHub username
            
        Returns:
            Optional[List[GitHubRepo]]: List of repositories if successful, None otherwise
        """
        pass