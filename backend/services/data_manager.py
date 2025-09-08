"""
Data Manager Service
Centralized data management and service initialization
"""

import pandas as pd
from datetime import datetime
from backend.services.ml_predictor import MLPredictor
from backend.services.search_engine import SearchEngine

# Global service instances
_ml_predictor = None
_search_engine = None
_dataframe = None
_new_reviews = []  # Store new reviews in memory

def initialize_services():
    """Initialize ML models and search engine with enhanced error handling"""
    global _ml_predictor, _search_engine, _dataframe
    
    try:
        # Initialize ML predictor
        _ml_predictor = MLPredictor()
        _ml_predictor.load_models_and_data()
        _dataframe = _ml_predictor.df
        
        # Initialize search engine
        _search_engine = SearchEngine(_dataframe)
        
        print("Services initialized successfully!")
        print(f"Dataset loaded with {len(_dataframe)} records")
        
        return True
        
    except Exception as e:
        print(f"Error initializing services: {e}")
        # Create fallback search engine
        _search_engine = SearchEngine(None)
        return False

def get_ml_predictor():
    """Get the ML predictor instance"""
    return _ml_predictor

def get_search_engine():
    """Get the search engine instance"""
    return _search_engine

def get_dataframe():
    """Get the main dataframe"""
    return _dataframe

def add_new_review(clothing_id, title, review_text, rating, recommended, age=None):
    """Add a new review to the in-memory storage"""
    global _new_reviews
    
    new_review = {
        'Clothing ID': clothing_id,
        'Title': title,
        'Review Text': review_text,
        'Rating': rating,
        'Recommended IND': recommended,
        'Age': age if age else 30,  # Default age
        'Date Added': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Is New': True  # Flag to identify new reviews
    }
    
    _new_reviews.append(new_review)
    print(f"Added new review for product {clothing_id}: {title}")
    return new_review

def get_product_reviews(clothing_id):
    """Get all reviews for a product including new ones"""
    global _dataframe, _new_reviews
    
    # Get original reviews from dataset
    original_reviews = []
    if _dataframe is not None and not _dataframe.empty:
        product_reviews = _dataframe[_dataframe['Clothing ID'] == clothing_id]
        original_reviews = product_reviews[
            ['Title', 'Review Text', 'Rating', 'Recommended IND', 'Age']
        ].to_dict('records')
        
        # Add Is New flag to original reviews
        for review in original_reviews:
            review['Is New'] = False
            review['Date Added'] = None
    
    # Get new reviews for this product
    new_product_reviews = [
        review for review in _new_reviews 
        if review['Clothing ID'] == clothing_id
    ]
    
    # Combine original and new reviews
    all_reviews = original_reviews + new_product_reviews
    
    return all_reviews

def get_review_statistics(clothing_id):
    """Get review statistics including new reviews"""
    all_reviews = get_product_reviews(clothing_id)
    
    if not all_reviews:
        return {'avg_rating': 0, 'review_count': 0}
    
    ratings = [review['Rating'] for review in all_reviews]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    review_count = len(all_reviews)
    
    return {
        'avg_rating': round(avg_rating, 1),
        'review_count': review_count
    }
