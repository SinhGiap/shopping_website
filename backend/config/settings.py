"""
Application Configuration Settings
Exact configuration matching Milestone I implementation
"""

import os

class Config:
    """Base configuration exactly matching Milestone I"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-shopping-website-milestone1'
    DEBUG = True
    THREADED = True
    HOST = '127.0.0.1'
    PORT = 5002
    
    # File paths for the dataset (Milestone I approach)
    DATASET_PATHS = [
        "assignment3_II.csv",
        os.path.join(os.getcwd(), "assignment3_II.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assignment3_II.csv"),
        r"c:\Users\Admin\Desktop\advprogramming\shopping_website\assignment3_II.csv"
    ]
    
    # ML model parameters exactly as in Milestone I
    MAX_FEATURES_BOW = 10000  # Match Milestone I
    MAX_FEATURES_TITLE = 5000  # Match Milestone I  
    MIN_DF = 2  # Match Milestone I
    MAX_DF = 0.95  # Match Milestone I default
    
    # Model parameters exactly as in Milestone I
    MAX_ITER = 2000  # Match Milestone I
    SOLVER = "liblinear"  # Match Milestone I
    RANDOM_STATE = 42  # Match Milestone I
    CLASS_WEIGHT = "balanced"  # Match Milestone I
    
    # Search configuration matching Milestone I
    SEARCH_RESULTS_LIMIT = 50
    FUZZY_THRESHOLD = 80  # Percentage threshold for fuzzy matching
    FEATURED_ITEMS_LIMIT = 6  # For home page
    SEARCH_PER_PAGE = 12  # For pagination
    MIN_SEARCH_SCORE = 5  # Minimum search score threshold
    
    # Text processing settings matching Milestone I
    MIN_REVIEW_LENGTH = 3
    MAX_REVIEW_LENGTH = 5000
    
    # NLTK data path configuration
    NLTK_DATA_PATH = os.path.expanduser('~/nltk_data')
    
    # Spell checking configuration (Milestone I)
    SPELL_CHECK_ENABLED = False  # Disabled for performance
    
    # Collocation settings (Milestone I)
    COLLOCATION_THRESHOLD = 5.0
    MIN_COLLOCATION_FREQ = 2

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or Config.SECRET_KEY

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
