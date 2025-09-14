"""
Data Manager Service
Centralized data management and service initialization
"""

import pandas as pd
import os
from datetime import datetime
from backend.services.ml_predictor import MLPredictor
from backend.services.search_engine import SearchEngine

# Global service instances
_ml_predictor = None
_search_engine = None
_dataframe = None
_csv_file_path = "assignment3_II.csv"  # Path to the CSV file

def initialize_services():
    """Initialize ML models and search engine with enhanced error handling"""
    global _ml_predictor, _search_engine, _dataframe, _csv_file_path
    
    try:
        # Set the correct path to the CSV file
        _csv_file_path = os.path.join(os.getcwd(), "assignment3_II.csv")
        
        # Initialize ML predictor
        _ml_predictor = MLPredictor()
        _ml_predictor.load_models_and_data()
        _dataframe = _ml_predictor.df
        
        # Initialize search engine
        _search_engine = SearchEngine(_dataframe)
        
        print("Services initialized successfully!")
        if _dataframe is None:
            print("[ERROR] _dataframe is None before len() check!")
        print(f"Dataset loaded with {len(_dataframe) if _dataframe is not None else 'None'} records")
        print(f"CSV file path: {_csv_file_path}")
        
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
    """Add a new review directly to the CSV file and reload data"""
    global _dataframe, _ml_predictor, _search_engine
    
    try:
        # Create new review record
        new_review = {
            'Clothing ID': clothing_id,
            'Age': age if age else 30,  # Default age
            'Title': title,
            'Review Text': review_text,
            'Rating': rating,
            'Recommended IND': recommended,
            'Positive Feedback Count': 0,  # Default value
            'Division Name': '',  
            'Department Name': '',  
            'Class Name': '',  
            'Clothes Title': '',  
            'Clothes Description': ''  
        }
        
        # Get product details from existing data to fill missing fields
        if _dataframe is not None and not _dataframe.empty:
            existing_product = _dataframe[_dataframe['Clothing ID'] == clothing_id].iloc[0]
            if not existing_product.empty:
                new_review['Division Name'] = existing_product.get('Division Name', '')
                new_review['Department Name'] = existing_product.get('Department Name', '')
                new_review['Class Name'] = existing_product.get('Class Name', '')
                new_review['Clothes Title'] = existing_product.get('Clothes Title', '')
                new_review['Clothes Description'] = existing_product.get('Clothes Description', '')
        
        # Convert to DataFrame row
        new_review_df = pd.DataFrame([new_review])
        
        # Append to CSV file
        if os.path.exists(_csv_file_path):
            # Append to existing file
            new_review_df.to_csv(_csv_file_path, mode='a', header=False, index=False)
            print(f" Review appended to {_csv_file_path}")
        else:
            # Create new file with headers
            new_review_df.to_csv(_csv_file_path, mode='w', header=True, index=False)
            print(f" New CSV file created: {_csv_file_path}")
        
        # Reload the dataframe to include the new review
        _dataframe = pd.read_csv(_csv_file_path)
        
        # Reinitialize ML predictor and search engine with updated data
        if _ml_predictor:
            _ml_predictor.df = _dataframe
        if _search_engine:
            _search_engine.df = _dataframe
        
        print(f" Added new review for product {clothing_id}: '{title}' (Rating: {rating})")
        print(f" Dataset now has {len(_dataframe)} total records")
        
        return new_review
        
    except Exception as e:
        print(f" Error adding review to CSV: {e}")
        return None

def get_product_reviews(clothing_id):
    """Get all reviews for a product from the CSV data"""
    global _dataframe
    
    # Get all reviews from the updated dataset
    all_reviews = []
    if _dataframe is not None and not _dataframe.empty:
        product_reviews = _dataframe[_dataframe['Clothing ID'] == clothing_id]
        all_reviews = product_reviews[
            ['Title', 'Review Text', 'Rating', 'Recommended IND', 'Age']
        ].to_dict('records')
        
        # All reviews are considered "original" now since they're in the CSV
        for review in all_reviews:
            review['Is New'] = False
            review['Date Added'] = None
    
    return all_reviews

def get_review_statistics(clothing_id):
    """Get review statistics including new reviews"""
    all_reviews = get_product_reviews(clothing_id)
    
    if not all_reviews:
        return {'avg_rating': 0, 'review_count': 0}
    
    ratings = [review['Rating'] for review in all_reviews]
    if ratings is None:
        print("[ERROR] ratings is None before len() check!")
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    if all_reviews is None:
        print("[ERROR] all_reviews is None before len() check!")
    review_count = len(all_reviews) if all_reviews is not None else 0
    
    return {
        'avg_rating': round(avg_rating, 1),
        'review_count': review_count
    }

def enrich_products_with_review_counts(products):
    """Add review count to product data from the updated CSV"""
    if not products:
        return products
    
    enriched_products = []
    for product in products:
        # Create a copy of the product dict
        enriched_product = product.copy() if hasattr(product, 'copy') else dict(product)
        
        # Get review statistics for this product from the updated CSV data
        clothing_id = enriched_product.get('Clothing ID')
        if clothing_id:
            stats = get_review_statistics(clothing_id)
            enriched_product['Review Count'] = stats['review_count']
        else:
            enriched_product['Review Count'] = 0
            
        enriched_products.append(enriched_product)
    
    return enriched_products

def reload_data():
    """Reload data from CSV file - useful after adding new reviews"""
    global _dataframe, _ml_predictor, _search_engine
    
    try:
        if os.path.exists(_csv_file_path):
            _dataframe = pd.read_csv(_csv_file_path)
            
            # Update ML predictor and search engine with new data
            if _ml_predictor:
                _ml_predictor.df = _dataframe
            if _search_engine:
                _search_engine.df = _dataframe
                
            print(f" Data reloaded from {_csv_file_path}: {len(_dataframe)} records")
            return True
        else:
            print(f" CSV file not found: {_csv_file_path}")
            return False
            
    except Exception as e:
        print(f" Error reloading data: {e}")
        return False
