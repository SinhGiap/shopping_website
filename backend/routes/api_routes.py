"""
API Routes Blueprint
Handles API endpoints for predictions and data
"""

from flask import Blueprint, request, jsonify

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/predict_recommendation', methods=['POST'])
def predict_recommendation():
    """Enhanced API endpoint for ML prediction"""
    try:
        from backend.services.data_manager import get_ml_predictor, get_dataframe
        
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

@api_bp.route('/categories')
def get_categories():
    """Enhanced API endpoint to get category data"""
    try:
        from backend.services.data_manager import get_dataframe
        
        df = get_dataframe()
        
        if df is not None and not df.empty:
            categories = {
                'divisions': sorted(df['Division Name'].dropna().unique().tolist()),
                'departments': sorted(df['Department Name'].dropna().unique().tolist()),
                'classes': sorted(df['Class Name'].dropna().unique().tolist())
            }
        else:
            categories = {
                'divisions': ['General', 'Petite'],
                'departments': ['Tops', 'Bottoms', 'Dresses'],
                'classes': ['Shirts', 'Pants', 'Dresses']
            }
        
        return jsonify(categories)
    except Exception as e:
        print(f"Error getting categories: {e}")
        return jsonify({
            'divisions': ['General'],
            'departments': ['Tops'],
            'classes': ['Shirts'],
            'error': str(e)
        })

# Legacy route for backward compatibility
@api_bp.route('/predict_recommendation', methods=['POST'])
def predict_recommendation_legacy():
    """Legacy endpoint - redirects to main prediction endpoint"""
    return predict_recommendation()
