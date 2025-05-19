from typing import List, Dict
import google_play_scraper
from textblob import TextBlob
from datetime import datetime

class AppReviewController:
    def __init__(self):
        self.app_id = "com.dukeenergy.customerapp.release"
        
    def get_app_reviews(self, limit: int = 200) -> List[Dict]:
        """
        Fetches reviews from the Duke Energy app on Google Play Store
        
        Args:
            limit: Maximum number of reviews to fetch
            
        Returns:
            List of reviews with sentiment analysis, sorted by date
        """
        reviews = []
        result, _ = google_play_scraper.reviews(
            self.app_id,
            lang='en',
            country='us',
            count=limit,
            sort=google_play_scraper.Sort.NEWEST  # Ensure we get newest reviews first
        )
        
        for review in result:
            sentiment = self._analyze_sentiment(review['content'])
            # Convert timestamp to integer if it's a datetime object
            timestamp = int(review['at'].timestamp()) if isinstance(review['at'], datetime) else review['at']
            reviews.append({
                'review_text': review['content'],
                'stars': review['score'],  # More clearly named as stars
                'sentiment': sentiment,
                'date': timestamp,
                'date_formatted': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            })
            
        # Sort by date (newest first) - although the API should already return sorted
        reviews.sort(key=lambda x: x['date'], reverse=True)
        return reviews
    
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
