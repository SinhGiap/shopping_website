# Backend Architecture Documentation

## Overview
The shopping website has been restructured into a professional backend architecture with proper separation of concerns and modular design.

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

## Key Improvements

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

#### Text Processing Service (`text_processor.py`)
- Handles NLTK setup and text preprocessing
- Tokenization, lemmatization, stopword removal
- Reusable across different components

#### ML Prediction Service (`ml_predictor.py`)
- Manages ML model loading and training
- BoW and Title+Structured feature models
- Ensemble prediction with confidence scores

#### Search Engine Service (`search_engine.py`)
- Fuzzy search functionality
- Category filtering
- Product deduplication and aggregation

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

## Benefits

1. **Maintainability**: Code is organized and easy to understand
2. **Scalability**: Easy to add new features without breaking existing code
3. **Testability**: Each component can be tested independently
4. **Reusability**: Services can be reused across different parts of the application
5. **Professional**: Follows industry best practices and patterns

## Usage

### Running the Application
```bash
python app.py
```

### Development
- Modify services in `backend/services/`
- Add new routes in `backend/routes/`
- Update configuration in `backend/config/settings.py`

### Adding New Features
1. Create service classes in `backend/services/`
2. Add routes in appropriate blueprint files
3. Update `app_factory.py` to register new blueprints
4. Add configuration settings if needed

## Migration Notes

The original `app.py` has been backed up as `app_old.py`. All functionality remains the same, but the code is now:
- More organized and maintainable
- Easier to extend and modify
- Following professional development practices
- Better separated for testing and debugging

The API endpoints remain unchanged, ensuring backward compatibility with any existing integrations.
