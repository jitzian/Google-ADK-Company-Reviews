from vertexai.language_models import TextGenerationModel
from controller.HistoricalReviewController import HistoricalReviewController
from agent_descriptions.app_review_analysis_description import app_review_analysis_description

class AppReviewAnalysisAgent:
    def __init__(self):
        self.review_controller = HistoricalReviewController()
        self.model = TextGenerationModel.from_pretrained("text-bison")
        self.description = app_review_analysis_description

    def analyze_historical_reviews(self, years: int = 5) -> str:
        """
        Analyzes app reviews for the specified number of years and generates insights
        
        Args:
            years: Number of years to analyze (default: 5)
            
        Returns:
            String containing analysis and insights
        """
        # Get the historical review data
        analysis_data = self.review_controller.get_historical_reviews(years)
        
        # Prepare the prompt for the LLM
        prompt = self._create_analysis_prompt(analysis_data)
        
        # Generate insights using the LLM
        response = self.model.predict(prompt, temperature=0.2)
        
        return response.text

    def _create_analysis_prompt(self, analysis_data: dict) -> str:
        """
        Creates a prompt for the LLM to analyze the review data
        
        Args:
            analysis_data: Dictionary containing the review analysis data
            
        Returns:
            String containing the formatted prompt
        """
        prompt = "Based on the following app review data, provide a concise analysis of:\n"
        prompt += "1. Key trends in customer issues over time\n"
        prompt += "2. Most common problem categories and their evolution\n"
        prompt += "3. Changes in sentiment over the years\n"
        prompt += "4. Specific recommendations for improvement\n\n"
        prompt += "Review Data:\n"
        
        for year, data in sorted(analysis_data.items()):
            prompt += f"\nYear {year}:\n"
            prompt += f"- Total Reviews: {data['total_reviews']}\n"
            prompt += f"- Average Rating: {data['average_rating']:.2f}\n"
            prompt += "- Problem Categories:\n"
            for category, count in data['categories'].items():
                percentage = (count / data['total_reviews']) * 100
                prompt += f"  * {category}: {count} ({percentage:.1f}%)\n"
            prompt += "- Sentiment Distribution:\n"
            for sentiment, count in data['sentiment'].items():
                percentage = (count / data['total_reviews']) * 100
                prompt += f"  * {sentiment}: {count} ({percentage:.1f}%)\n"
        
        prompt += "\nProvide your analysis in a clear, structured format."
        return prompt
