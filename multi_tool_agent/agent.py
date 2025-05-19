from datetime import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from typing import List, Dict
from collections import defaultdict
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import TextGenerationModel
from .controller.AppReviewController import AppReviewController
from .controller.HistoricalReviewController import HistoricalReviewController
from .constants import global_constants
from .domain.model.GitHubRepo import GitHubRepo
from .agent_descriptions.agent_repositories_description import github_repositories_description
from .agent_descriptions.agent_repositories_description import family_members_description

names = ["anne", "phil","morgan","matthew","john", "mathilda"]

def check_name(name: str) -> dict:

    """
    Checks if a name is available.

    Args:
        name (str): The name to check.

    Returns:
        dict: status and result or error msg.
    """


    if name.lower() in names:
        return {
            "status": "success", 
            "report": f"{name} is available."
        }
    else:
        return {
            "status": "error", 
            "error_message": f"{name} is not available."
        }

def check_gender(name: str) -> dict:
    """
    Checks the gender of a name.

    Args:
        name (str): The name to check.

    Returns:
        dict: status and result or error msg.
    """

    if name.lower() == "anne" or name.lower() == "mathilda"or name.lower() == "morgan":
        return {
            "status": "success", 
            "report": f"{name} is female."
        }
    elif name.lower() == "phil" or  name.lower() == "matthew" or name.lower() == "john":
        return {
            "status": "success", 
            "report": f"{name} is male."
        }
    else:
        return {
            "status": "error", 
            "error_message": f"{name} is not available."
        }

def get_repositories(user: str) -> dict:
    repository = RepositoriesImpl([])  # Initialize with empty list
    controller = GitHubController(repository)
    result_repos = []
    
    # Get repositories for the user
    repos = controller.get_user_repositories(user)
    if repos:
        result_repos.extend(repos)

    language_type = analyze_repositories(result_repos)

    return {
        "status": "success",
        "report": language_type
    }
        
def analyze_repositories(result_repos: List[GitHubRepo]) -> dict:
    language_type = {
        "python": 0,
        "kotlin": 0,
        "java": 0,
    }
    if result_repos:  # Changed from 'is not empty' to proper Python syntax
        for repo in result_repos:
            if repo.language and repo.language.lower() in language_type:  # Added .lower() for case insensitive comparison
                language_type[repo.language.lower()] += 1
    return language_type

family_members_agent = Agent(
    name = "family_members_agent",
    model = "gemini-2.0-flash-exp",
    description= family_members_description,
    instruction = (
        "Check if a name is part of the family members list."
    ),
)


# root_agent = Agent(
repositories_agent = Agent(
    name="repositories_agent",
    model="gemini-2.0-flash-exp",
    description = github_repositories_description,
    instruction = (
        "Search in the repositories the type of language used in them"
    ),
    tools=[get_repositories],
    sub_agents=[family_members_agent],
)

def analyze_app_reviews(limit: int = 200) -> dict:
    """
    Analyzes Duke Energy app reviews for sentiment.

    Args:
        limit: Maximum number of reviews to analyze

    Returns:
        dict: Analysis results including sentiment scores and latest reviews
    """
    controller = AppReviewController()
    reviews = controller.get_app_reviews(limit)
    
    if not reviews:
        return {
            "status": "error",
            "error_message": "No reviews found"
        }
    
    # Calculate average sentiment
    total_polarity = 0
    total_subjectivity = 0
    total_stars = 0
    
    for review in reviews:
        total_polarity += review['sentiment']['polarity']
        total_subjectivity += review['sentiment']['subjectivity']
        total_stars += review['stars']
    
    num_reviews = len(reviews)
    
    # Format latest reviews for display
    latest_reviews = [{
        'date': review['date_formatted'],
        'stars': '⭐' * int(review['stars']),  # Visual representation of stars
        'text': review['review_text'],
        'sentiment_score': round(review['sentiment']['polarity'], 2),
        'sentiment_label': 'Positive' if review['sentiment']['polarity'] > 0 else 'Negative' if review['sentiment']['polarity'] < 0 else 'Neutral'
    } for review in reviews[:10]]  # Show latest 10 reviews
    
    return {
        "status": "success",
        "report": {
            "summary": {
                "average_sentiment": round(total_polarity / num_reviews, 2),
                "average_subjectivity": round(total_subjectivity / num_reviews, 2),
                "average_stars": round(total_stars / num_reviews, 1),
                "total_reviews_analyzed": num_reviews
            },
            "latest_reviews": latest_reviews
        }
    }

# Create app review agent
app_review_agent = Agent(
    name="app_review_agent",
    model="gemini-2.0-flash-exp",
    description="Analyzes customer sentiment from Duke Energy app reviews on Google Play Store",
    instruction="Analyze app reviews to understand customer sentiment",
    tools=[analyze_app_reviews],
)


def analyze_historical_app_reviews(years: int = 5) -> dict:
    """
    Analyzes Duke Energy app reviews over multiple years to identify problem trends.

    Args:
        years: Number of years of historical data to analyze (default: 5)

    Returns:
        dict: Analysis results including yearly trends, problem categories, and sentiment analysis
    """
    controller = HistoricalReviewController()
    analysis = controller.get_historical_reviews(years)
    
    if not analysis:
        return {
            "status": "error",
            "error_message": "No historical reviews found"
        }
    
    # Process the analysis data
    total_reviews = sum(data['total_reviews'] for data in analysis.values())
    overall_categories = defaultdict(int)
    yearly_trends = {}
    
    for year, data in analysis.items():
        # Calculate percentages for this year
        sentiment_percentages = {}
        for sentiment, count in data['sentiment'].items():
            sentiment_percentages[sentiment] = round((count / data['total_reviews']) * 100, 1)
        
        category_percentages = {}
        for category, count in data['categories'].items():
            category_percentages[category] = round((count / data['total_reviews']) * 100, 1)
            overall_categories[category] += count
        
        yearly_trends[year] = {
            "total_reviews": data['total_reviews'],
            "average_rating": round(data['average_rating'], 2),
            "sentiment_distribution": sentiment_percentages,
            "problem_categories": category_percentages
        }
    
    # Calculate overall category distribution
    overall_category_distribution = {}
    for category, count in overall_categories.items():
        overall_category_distribution[category] = round((count / total_reviews) * 100, 1)
    
    return {
        "status": "success",
        "report": {
            "summary": {
                "total_years_analyzed": len(analysis),
                "total_reviews_analyzed": total_reviews,
                "overall_problem_distribution": overall_category_distribution
            },
            "yearly_trends": yearly_trends
        }
    }

historical_review_agent = Agent(
    name="historical_review_agent",
    model="gemini-2.0-flash-exp",
    description="Analyzes historical Duke Energy app reviews to identify problem trends, categorize issues, and track sentiment changes over time. Provides yearly segmentation of customer feedback and problem categories.",
    instruction="Analyze app reviews over multiple years to identify trends in customer-reported problems, track sentiment changes, and provide actionable insights.",
    tools=[analyze_historical_app_reviews],
)

root_agent = app_review_agent