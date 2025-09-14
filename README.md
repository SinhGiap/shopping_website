# Enhanced Shopping Website 

### DEMO VIDEO ONEDRIVE LINK
 https://rmiteduau.sharepoint.com/:v:/s/ItNguyenSinhGiap/ETLNPrCLFWxKu2Rvv-fWO5cBgXkWV7DaJ2v_lJmEfShQWQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=WdenY1

Core Dependencies
- **Flask 3.1.2**: Modern web framework with latest security updates
- **pandas 2.3.1**: Advanced data manipulation and analysis
- **scikit-learn 1.7.1**: Machine learning library with latest algorithms
- **nltk 3.9.1**: Natural language processing toolkit
- **xgboost 3.0.5**: Gradient boosting framework for ML ensemble

### Frontend Stack
- **Bootstrap 5.1.3**: Responsive CSS framework with modern components
- **Bootstrap Icons**: Comprehensive icon library for consistent UI
- **Vanilla JavaScript ES6+**: Modern JavaScript without external dependencies
- **CSS Custom Properties**: Dynamic theming and responsive design

### Development Tools
- **requirements_freeze.txt**: Complete environment snapshot with 55+ packages
- **Debug Mode**: Enhanced error reporting and development features
- **Health Monitoring**: Built-in system health and performance monitoring

##  Quick Start Guide# 


### Prerequisites
- **Python 3.8+** (Python 3.9+ recommended for optimal performance)
- **pip package manager** (latest version recommended)
- **4GB RAM minimum** (8GB recommended for ML model operations)

### Installation Steps

1. **Navigate to project directory**
   ```bash
   cd shopping_website
   ```

2. **Install dependencies** (Choose one option):
   ```bash
   pip install -r requirements_freeze.txt
   ```

3. **Verify dataset availability**
   - Ensure `assignment3_II.csv` is present in project root
   - Application auto-detects dataset location with fallback mechanisms

4. **Launch the application**
   ```bash
   python app.py
   ```

5. **Access the enhanced platform**
   - **Main Application**: http://127.0.0.1:5002
   - **Admin Dashboard**: http://127.0.0.1:5002/admin  
   - **Health Check**: http://127.0.0.1:5002/health
   - **API Documentation**: Available through /api/ endpoints

##  Development & Customization

##  Dataset Requirements

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


### Architecture Documentation
- **[Backend Architecture](BACKEND_ARCHITECTURE.md)**: Complete backend system documentation
- **[Frontend Architecture](FRONTEND_ARCHITECTURE.md)**: Frontend design patterns and components



#### Advanced API Features
```javascript
// Enhanced ML Prediction API
fetch('/predict_recommendation', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        title: "Great product quality",
        text: "I love this item, fits perfectly and great value",
        rating: 5,
        clothing_id: 1077,
        division: "General",
        department: "Tops", 
        class_name: "Blouses"
    })
})
.then(response => response.json())
.then(data => {
    // Returns ensemble prediction with confidence scores
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence);
    console.log('Model Details:', data.model_predictions);
});
```

### Architecture Customization
- **Service Configuration**: Modify services in `backend/services/` for custom business logic
- **Route Management**: Add new blueprints in `backend/routes/` for additional functionality  
- **ML Model Tuning**: Adjust `ensemble_weights` and model parameters in `ml_predictor.py`
- **Search Optimization**: Configure search scoring and filters in `search_engine.py`
- **UI Theming**: Customize design system in `static/style.css` with CSS variables

### Development Workflow
```bash
# Enable debug mode for development
export FLASK_DEBUG=1  # Linux/Mac
set FLASK_DEBUG=1     # Windows

# Run with auto-reload
python app.py

# Monitor system health
curl http://127.0.0.1:5002/health

# Test ML predictions
curl -X POST http://127.0.0.1:5002/predict_recommendation \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","text":"Great product","rating":5}'
```

### Adding New Features
1. **Create service classes** in `backend/services/` for business logic
2. **Add route handlers** in appropriate blueprint files (`backend/routes/`)  
3. **Register blueprints** in `backend/app_factory.py`
4. **Update templates** in `templates/` for UI components
5. **Add configuration** in `backend/config/settings.py` if needed

##  Monitoring & Analytics

### System Health Monitoring
- **Real-time Status**: `/health` endpoint provides system status JSON
- **ML Model Status**: Monitor model loading and prediction performance
- **Search Performance**: Track search response times and accuracy
- **Error Tracking**: Comprehensive error logging with stack traces

### Administrative Features
- **Dashboard Analytics**: Comprehensive overview at `/admin`
- **Category Statistics**: Real-time product distribution analysis  
- **Review Analytics**: User engagement and recommendation patterns
- **Performance Metrics**: Response times and system resource usage


### Contribution 

- **Nguyen Phuoc Minh Phuc**: 25%
- **Nguyen Sinh Giap**: 25%
- **Dong Duc Binh**: 25%
- **Nguyen Vu Linh**: 25%

