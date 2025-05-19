from typing import List, Dict
import google_play_scraper
from textblob import TextBlob
from datetime import datetime, timedelta
from collections import defaultdict
import re

class HistoricalReviewController:
    def __init__(self):
        self.app_id = "com.dukeenergy.customerapp.release"
        self.problem_categories = {
            'login': r'login|sign.?in|password|authentication|account access',
            'billing': r'bill|payment|charge|balance|invoice',
            'performance': r'slow|crash|freeze|bug|error|loading',
            'usability': r'difficult|confusing|hard to|unclear|user.?friendly',
            'features': r'missing|need|should have|feature|functionality',
            'updates': r'update|version|upgrade|latest',
            'technical': r'connection|server|network|offline|error|fail',
            'customer_service': r'support|service|help|contact|representative'
        }

    def get_historical_reviews(self, years: int = 5) -> Dict:
        """
        Fetches and analyzes reviews from the past specified years
        
        Args:
            years: Number of years to analyze (default: 5)
            
        Returns:
            Dictionary containing analysis by year and category
        """
        # Calculate the timestamp for X years ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Fetch a large number of reviews to ensure we get enough historical data
        result, _ = google_play_scraper.reviews(
            self.app_id,
            lang='en',
            country='us',
            count=10000,  # Maximum number of reviews
            sort=google_play_scraper.Sort.NEWEST
        )
        
        # Initialize data structures for analysis
        yearly_analysis = defaultdict(lambda: {
            'total_reviews': 0,
            'average_rating': 0.0,
            'categories': defaultdict(int),
            'sentiment': {'positive': 0, 'neutral': 0, 'negative': 0}
        })
        
        total_reviews = 0
        ratings_sum = 0
        
        for review in result:
            review_date = review['at'] if isinstance(review['at'], datetime) else datetime.fromtimestamp(review['at'])
            
            # Skip reviews outside our time range
            if review_date < start_date:
                continue
                
            year = review_date.year
            yearly_data = yearly_analysis[year]
            
            # Update basic metrics
            yearly_data['total_reviews'] += 1
            ratings_sum += review['score']
            total_reviews += 1
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(review['content'])
            if sentiment['polarity'] > 0.1:
                yearly_data['sentiment']['positive'] += 1
            elif sentiment['polarity'] < -0.1:
                yearly_data['sentiment']['negative'] += 1
            else:
                yearly_data['sentiment']['neutral'] += 1
                
            # Categorize problems
            categories_found = self._categorize_problems(review['content'])
            for category in categories_found:
                yearly_data['categories'][category] += 1
        
        # Calculate averages and format results
        analysis_result = {}
        for year, data in yearly_analysis.items():
            if data['total_reviews'] > 0:
                analysis_result[year] = {
                    'total_reviews': data['total_reviews'],
                    'average_rating': ratings_sum / data['total_reviews'],
                    'categories': dict(data['categories']),
                    'sentiment': data['sentiment']
                }
        
        return analysis_result

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyzes the sentiment of given text using TextBlob
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing polarity and subjectivity scores
        """
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def _categorize_problems(self, text: str) -> List[str]:
        """
        Categorizes the problems mentioned in the review text
        
        Args:
            text: The review text to analyze
            
        Returns:
            List of identified problem categories
        """
        text = text.lower()
        categories = []
        
        for category, pattern in self.problem_categories.items():
            if re.search(pattern, text):
                categories.append(category)
                
        return categories
