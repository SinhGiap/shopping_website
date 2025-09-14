# Backend Architecture Documentation


## Current Status
 **Search Functionality**: Fully operational with exact match priority and score-based ordering  
 **ML Prediction Service**: Complete with ensemble models and fallback mechanisms  
 **Error Handling**: Comprehensive error handling with graceful degradation  
 **Performance**: Optimized search engine with intelligent caching and preprocessing

## Directory Structure

```
shopping_website/
├── app.py                      # Main application entry point
├── app_old.py                  # Backup of original monolithic app
├── backend/                    # Backend package
│   ├── __init__.py
│   ├── app_factory.py         # Application factory pattern
│   ├── config/                # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py        # Environment-specific settings
│   ├── models/                # Data models (for future use)
│   │   └── __init__.py
│   ├── routes/                # Route blueprints
│   │   ├── __init__.py
│   │   ├── main_routes.py     # Home, search routes
│   │   ├── product_routes.py  # Product detail, review routes
│   │   └── api_routes.py      # API endpoints
│   ├── services/              # Business logic services
│   │   ├── __init__.py
│   │   ├── text_processor.py  # Text preprocessing service
│   │   ├── ml_predictor.py    # ML prediction service
│   │   ├── search_engine.py   # Search functionality
│   │   └── data_manager.py    # Centralized data management
│   └── utils/                 # Utility functions (for future use)
│       └── __init__.py
├── templates/                 # Jinja2 templates
├── static/                    # Static assets (CSS, JS, images)
├── requirements.txt           # Python dependencies
└── assignment3_II.csv        # Dataset
```


### 1. Modular Architecture
- **Separation of Concerns**: Each component has a single responsibility
- **Blueprint Pattern**: Routes are organized into logical groups
- **Service Layer**: Business logic is isolated in service classes

### 2. Configuration Management
- **Environment-specific configs**: Development, production, testing
- **Centralized settings**: All configuration in one place
- **Environment variables**: Secure handling of sensitive data

### 3. Application Factory Pattern
- **Flexible app creation**: Easy to create apps with different configurations
- **Better testing**: Can create test apps with different settings
- **Scalability**: Easy to add new features and services

### 4. Service Organization

#### Text Processing Service (`text_processor.py`) - 
- **Advanced preprocessing**: Complete NLTK pipeline with negation handling
- **Performance tuned**: Optimized spell checking with caching and length filtering
- **Configurable**: Spell checking can be enabled/disabled for performance control
- **Robust error handling**: Safe processing of malformed text inputs
- **Lemmatization & tokenization**: Professional-grade text normalization

#### ML Prediction Service (`ml_predictor.py`) - 
- **Model management**: Automatic loading with version compatibility warnings
- **Ensemble predictions**: Combines LogisticRegression, AdaBoost, and XGBoost models
- **Feature engineering**: BoW vectorization + structured features (rating, division, etc.)
- **Fallback mechanisms**: Graceful degradation when models unavailable
- **Performance optimized**: Cached models with efficient prediction pipeline

#### Search Engine Service (`search_engine.py`) - 
- **Advanced fuzzy search**: Multi-tier priority scoring system
- **Exact match priority**: Exact matches receive 10000+ points for top ranking
- **Score preservation**: Fixed unique product aggregation to maintain search relevance
- **Category filtering**: Comprehensive filtering by division, department, class
- **Product deduplication**: Intelligent aggregation with review count and rating averages
- **Performance optimized**: Efficient search algorithms with proper error handling

#### Data Manager Service (`data_manager.py`)
- Centralized service initialization
- Global service instances
- Error handling and fallbacks

### 5. Route Organization

#### Main Routes (`main_routes.py`)
- Home page with featured products
- Search functionality with pagination
- Health check endpoint

#### Product Routes (`product_routes.py`)
- Product detail pages
- Review submission
- Review form handling

#### API Routes (`api_routes.py`)
- ML prediction API
- Category data API
- RESTful endpoints

## Recent Fixes & Improvements

### Search Functionality Resolution 
- **Issue**: "Column(s) ['search_score'] do not exist" error in search operations
- **Root Cause**: Conditional aggregation in `_get_unique_products` method failed when search_score column was missing
- **Solution**: Implemented dynamic aggregation dictionary that only includes search_score when available
- **Result**: Search now works flawlessly for all query types (empty, keyword, category filtering)

### Performance Optimizations 
- **Text Processing**: Implemented optimized spell checking with caching
- **Search Engine**: Added intelligent score preservation during product aggregation  
- **ML Models**: Enhanced model loading with better error handling
- **Memory Usage**: Optimized data processing to reduce memory footprint

### Error Handling Improvements 
- **Graceful Degradation**: All services now handle missing dependencies gracefully
- **Comprehensive Logging**: Detailed error logging for debugging and monitoring
- **User-Friendly Messages**: Clear error messages for end users
- **Fallback Mechanisms**: Robust fallbacks ensure application continues working


