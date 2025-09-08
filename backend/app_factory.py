"""
Application Factory
Creates and configures the Flask application
"""

import os
from flask import Flask

from backend.config.settings import config
from backend.services.data_manager import initialize_services

def create_app(config_name=None):
    """Application factory pattern"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize services
    with app.app_context():
        initialize_services()
    
    # Register blueprints
    from backend.routes.main_routes import main_bp
    from backend.routes.product_routes import product_bp
    from backend.routes.api_routes import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(product_bp)
    app.register_blueprint(api_bp)
    
    # Legacy routes for backward compatibility
    register_legacy_routes(app)
    
    return app

def register_legacy_routes(app):
    """Register legacy routes for backward compatibility"""
    from flask import request, jsonify
    from backend.services.data_manager import get_ml_predictor, get_dataframe
    
    @app.route('/predict_recommendation', methods=['POST'])
    def predict_recommendation_legacy():
        """Legacy prediction endpoint"""
        try:
            ml_predictor = get_ml_predictor()
            df = get_dataframe()
            
            # Handle both JSON and form data
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form.to_dict()
            
            review_title = str(data.get('title', '')).strip()
            review_text = str(data.get('text', '')).strip()
            rating = int(data.get('rating', 5))
            
            # Get product info for structured features
            clothing_id = data.get('clothing_id')
            if clothing_id and df is not None and not df.empty:
                try:
                    product_rows = df[df['Clothing ID'] == int(clothing_id)]
                    if not product_rows.empty:
                        product = product_rows.iloc[0]
                        division = str(product.get('Division Name', 'General'))
                        department = str(product.get('Department Name', 'Tops'))
                        class_name = str(product.get('Class Name', 'Blouses'))
                    else:
                        division = str(data.get('division', 'General'))
                        department = str(data.get('department', 'Tops'))
                        class_name = str(data.get('class_name', 'Blouses'))
                except:
                    division = str(data.get('division', 'General'))
                    department = str(data.get('department', 'Tops'))
                    class_name = str(data.get('class_name', 'Blouses'))
            else:
                division = str(data.get('division', 'General'))
                department = str(data.get('department', 'Tops'))
                class_name = str(data.get('class_name', 'Blouses'))
            
            # Get ML prediction
            if ml_predictor:
                prediction_result = ml_predictor.predict_recommendation(
                    review_title, review_text, rating, division, department, class_name
                )
            else:
                prediction_result = {
                    'prediction': 1 if rating >= 4 else 0,
                    'confidence': 0.5,
                    'status': 'fallback'
                }
            
            return jsonify(prediction_result)
            
        except Exception as e:
            print(f"Error in prediction endpoint: {e}")
            return jsonify({
                'prediction': 1 if int(request.form.get('rating', 5)) >= 4 else 0, 
                'confidence': 0.5, 
                'error': str(e),
                'status': 'error'
            })
