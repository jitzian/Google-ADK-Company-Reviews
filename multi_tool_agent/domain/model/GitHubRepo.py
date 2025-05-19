from dataclasses import dataclass
from typing import Optional

@dataclass
class GitHubRepo:
    id: int
    name: str
    full_name: str
    private: bool
    html_url: str
    description: Optional[str] = None
    fork: bool = False
    url: str = ""
    created_at: str = ""
    updated_at: str = ""
    pushed_at: str = ""
    language: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GitHubRepo':
        return cls(
            id=data.get('id'),
            name=data.get('name'),
            full_name=data.get('full_name'),
            private=data.get('private', False),
            html_url=data.get('html_url'),
            description=data.get('description'),
            fork=data.get('fork', False),
            url=data.get('url', ''),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            pushed_at=data.get('pushed_at', ''),
            language=data.get('language')
        )
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'full_name': self.full_name,
            'private': self.private,
            'html_url': self.html_url,
            'description': self.description,
            'fork': self.fork,
            'url': self.url,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'pushed_at': self.pushed_at,
            'language': self.language
        }
