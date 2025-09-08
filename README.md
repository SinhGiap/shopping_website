# Enhanced Shopping Website - Milestone II

A comprehensive Flask-based online shopping website with machine learning-powered review prediction capabilities.

## 🌟 Features

### Core Functionality
- **Smart Product Search**: Advanced fuzzy search with intelligent matching
- **ML-Powered Predictions**: AI-driven review recommendation predictions
- **Responsive Design**: Mobile-first Bootstrap 5 responsive interface
- **Product Catalog**: Browse products by categories, departments, and classes
- **Review System**: Write and read customer reviews with ML predictions

### Machine Learning Integration
- **Bag-of-Words Model**: Text analysis using Milestone I preprocessing pipeline
- **Ensemble Predictions**: Multiple model combination for better accuracy
- **Real-time Predictions**: Instant ML predictions for new reviews
- **Confidence Scoring**: Prediction confidence levels

### Technical Features
- **Enhanced Error Handling**: Robust fallback mechanisms
- **Smart Fallbacks**: Graceful degradation when ML models unavailable
- **Modern UI/UX**: Professional design with smooth animations
- **RESTful API**: JSON endpoints for ML predictions
- **Pagination**: Efficient handling of large product catalogs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (recommend 3.9)
- pip package manager

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd shopping_website
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure dataset is available**
   - Place `assignment3_II.csv` in the project root or parent directory
   - The app will automatically search multiple locations

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the website**
   - Open your browser to: http://127.0.0.1:5002
   - The enhanced shopping website will be available

## 📊 Dataset Requirements

The application expects the enhanced dataset (`assignment3_II.csv`) with the following columns:
- `Clothing ID`: Unique product identifier
- `Age`: Customer age
- `Title`: Review title
- `Review Text`: Customer review content
- `Rating`: Product rating (1-5)
- `Recommended IND`: Recommendation indicator (0/1)
- `Positive Feedback Count`: Helpful votes
- `Division Name`: Product division
- `Department Name`: Product department
- `Class Name`: Product class
- `Clothes Title`: Product name
- `Clothes Description`: Product description

## 🏗️ Architecture

### Backend Structure
```
shopping_website/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── home.html        # Homepage
│   ├── search.html      # Search results
│   ├── product.html     # Product details
│   └── add_review.html  # Review form
└── static/              # Static assets
    └── style.css        # Enhanced CSS styles
```

### Key Components

#### TextProcessor Class
- NLTK-based text preprocessing
- Tokenization, lemmatization, stopword removal
- Fallback mechanisms for missing dependencies

#### MLPredictor Class
- Multiple ML model management
- Ensemble prediction system
- Enhanced error handling and fallbacks

#### SearchEngine Class
- Fuzzy search with intelligent matching
- Category filtering
- Performance optimization

## 🤖 Machine Learning Features

### Model Integration
- **Bag-of-Words Model**: Primary text classification model
- **Title + Structured Features**: Combined text and metadata model
- **Ensemble Prediction**: Weighted combination of multiple models

### Prediction Pipeline
1. Text preprocessing using Milestone I pipeline
2. Feature extraction (BoW, TF-IDF, structured features)
3. Model prediction with confidence scoring
4. Ensemble result calculation
5. JSON response with detailed metrics

### Fallback Strategy
- Smart fallbacks when ML models fail to load
- Rating-based simple predictions
- Graceful error handling
- User-friendly error messages

## 🎨 UI/UX Features

### Modern Design
- Bootstrap 5.1.3 framework
- Custom CSS with gradients and animations
- Responsive mobile-first design
- Professional color scheme

### Interactive Elements
- Real-time form validation
- Dynamic star ratings
- Live character counters
- Smooth page transitions
- Hover effects and animations

### User Experience
- Intuitive navigation
- Clear visual hierarchy
- Loading states and feedback
- Error handling with helpful messages
- Accessible design patterns

## 🔧 Configuration

### Environment Variables
No environment variables required for basic operation.

### Customization Options
- Modify `ensemble_weights` in MLPredictor for different model combinations
- Adjust search parameters in SearchEngine class
- Customize UI colors in `static/style.css`
- Configure Flask settings in `app.py`

## 📱 API Endpoints

### Core Routes
- `GET /`: Homepage with featured products
- `GET /search`: Product search with filters
- `GET /product/<id>`: Product detail page
- `GET /add_review/<id>`: Review form
- `POST /submit_review`: Submit new review

### API Endpoints
- `POST /predict_recommendation`: ML prediction API
- `GET /api/categories`: Category data API
- `GET /health`: Health check endpoint

### API Usage Example
```javascript
fetch('/predict_recommendation', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        title: "Great product",
        text: "I love this item, fits perfectly",
        rating: 5,
        clothing_id: 1077
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🔍 Search Features

### Fuzzy Search
- Intelligent text matching
- Typo tolerance
- Partial word matching
- Fashion-specific term variations

### Category Filtering
- Division-based filtering
- Department categorization
- Class-specific searches
- Combined filter support

### Search Performance
- Optimized query processing
- Pagination for large results
- Caching for common searches
- Real-time result updates

## 📈 Performance Optimizations

### Loading Performance
- Lazy loading of ML models
- Efficient dataset processing
- Optimized template rendering
- Compressed static assets

### Runtime Performance
- Cached search results
- Efficient pagination
- Minimal database queries
- Smart fallback mechanisms

## 🛠️ Development

### Code Structure
- Modular class-based architecture
- Separation of concerns
- Comprehensive error handling
- Extensive inline documentation

### Testing
- Built-in health check endpoint
- Error logging and monitoring
- Graceful failure handling
- Development debug mode

### Deployment
- Production-ready Flask configuration
- Gunicorn WSGI server support
- Static file serving
- Environment-specific settings

## 🎯 Future Enhancements

### Planned Features
- User authentication system
- Shopping cart functionality
- Order management
- Advanced recommendation system
- Real-time chat support

### Technical Improvements
- Database integration (PostgreSQL/MongoDB)
- Redis caching layer
- API rate limiting
- Advanced analytics
- A/B testing framework

## 📚 Dependencies

### Core Dependencies
- **Flask 2.3.3**: Web framework
- **pandas 2.0.3**: Data manipulation
- **scikit-learn 1.3.0**: Machine learning
- **nltk 3.8.1**: Natural language processing
- **numpy 1.24.3**: Numerical computing

### UI Dependencies
- **Bootstrap 5.1.3**: CSS framework (CDN)
- **Bootstrap Icons**: Icon library (CDN)
- **jQuery 3.6.0**: JavaScript library (CDN)

## 🐛 Troubleshooting

### Common Issues

1. **Dataset not found**
   - Ensure `assignment3_II.csv` is in project directory or parent directory
   - Check file permissions and path

2. **ML models not loading**
   - Install all required dependencies: `pip install -r requirements.txt`
   - Check NLTK data downloads
   - Verify dataset format

3. **Search not working**
   - Ensure dataset is loaded properly
   - Check for data format issues
   - Verify text preprocessing pipeline

### Debug Mode
Run with debug enabled for detailed error messages:
```bash
python app.py
```

## 📄 License

This project is developed for educational purposes as part of Milestone II coursework.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in console
3. Verify all dependencies are installed
4. Check dataset format and location

---

**Enhanced Shopping Website** - Bringing AI-powered insights to online shopping! 🛍️✨
