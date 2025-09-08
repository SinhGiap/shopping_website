"""
Data Manager Service
Centralized data management and service initialization
"""

from backend.services.ml_predictor import MLPredictor
from backend.services.search_engine import SearchEngine

# Global service instances
_ml_predictor = None
_search_engine = None
_dataframe = None

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
