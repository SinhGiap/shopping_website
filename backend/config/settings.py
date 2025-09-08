"""
Application Configuration Settings
"""

import os

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'enhanced_milestone2_secret_key_for_shopping_website'
    
    # Dataset settings
    DATASET_FILENAME = 'assignment3_II.csv'
    DATASET_PATHS = [
        DATASET_FILENAME,
        f"../{DATASET_FILENAME}",
        f"../../{DATASET_FILENAME}"
    ]
    
    # Stopwords file paths
    STOPWORDS_PATHS = [
        "stopwords_en.txt",
        "../task1/stopwords_en.txt",
        "../../task1/stopwords_en.txt"
    ]
    
    # ML Model settings
    MAX_FEATURES_BOW = 5000
    MAX_FEATURES_TITLE = 2000
    MIN_DF = 1
    MAX_DF = 0.95
    
    # Search settings
    SEARCH_PER_PAGE = 12
    FEATURED_ITEMS_LIMIT = 6
    SEARCH_LIMIT = 1000
    MIN_SEARCH_SCORE = 5
    
    # API settings
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5002

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
