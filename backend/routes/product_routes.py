"""
Product Routes Blueprint
Handles product-related pages and functionality
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify

product_bp = Blueprint('product', __name__)

@product_bp.route('/product/<int:clothing_id>')
def product_detail(clothing_id):
    """Enhanced product detail page with new review support"""
    try:
        from backend.services.data_manager import get_dataframe, get_product_reviews, get_review_statistics
        
        df = get_dataframe()
        
        if df is None or df.empty:
            flash('Product database not available', 'error')
            return redirect(url_for('main.home'))
        
        # Get product details
        product_rows = df[df['Clothing ID'] == clothing_id]
        
        if product_rows.empty:
            flash('Product not found', 'error')
            return redirect(url_for('main.home'))
        
        product = product_rows.iloc[0]
        
        # Get all reviews for this product (original + new)
        reviews = get_product_reviews(clothing_id)
        
        # Calculate statistics including new reviews
        stats = get_review_statistics(clothing_id)
        
        return render_template('product.html',
                             product=product.to_dict(),
                             reviews=reviews,
                             avg_rating=stats['avg_rating'],
                             review_count=stats['review_count'])
        
    except Exception as e:
        print(f"Error in product detail route: {e}")
        flash('Error loading product details', 'error')
        return redirect(url_for('main.home'))

@product_bp.route('/add_review/<int:clothing_id>')
def add_review_form(clothing_id):
    """Enhanced form to add new review"""
    try:
        from backend.services.data_manager import get_dataframe
        
        df = get_dataframe()
        
        if df is None or df.empty:
            flash('Product database not available', 'error')
            return redirect(url_for('main.home'))
        
        product_rows = df[df['Clothing ID'] == clothing_id]
        if product_rows.empty:
            flash('Product not found', 'error')
            return redirect(url_for('main.home'))
        
        product = product_rows.iloc[0]
        return render_template('add_review.html', product=product.to_dict())
        
    except Exception as e:
        print(f"Error in add review form: {e}")
        flash('Error loading review form', 'error')
        return redirect(url_for('main.home'))

@product_bp.route('/submit_review', methods=['POST'])
def submit_review():
    """Enhanced review submission"""
    try:
        clothing_id = int(request.form['clothing_id'])
        title = str(request.form['title']).strip()
        review_text = str(request.form['review_text']).strip()
        rating = int(request.form['rating'])
        recommended = int(request.form.get('recommended', 1))
        
        # Validate input
        if not title or not review_text:
            flash('Please provide both title and review text', 'error')
            return redirect(url_for('product.add_review_form', clothing_id=clothing_id))
        
        if rating < 1 or rating > 5:
            flash('Rating must be between 1 and 5', 'error')
            return redirect(url_for('product.add_review_form', clothing_id=clothing_id))
        
        # Save the new review
        from backend.services.data_manager import add_new_review, get_dataframe, get_ml_predictor
        
        # Add the review to our storage
        new_review = add_new_review(
            clothing_id=clothing_id,
            title=title,
            review_text=review_text,
            rating=rating,
            recommended=recommended
        )
        
        # Get ML prediction for the review
        try:
            df = get_dataframe()
            ml_predictor = get_ml_predictor()
            
            if df is not None and not df.empty and ml_predictor:
                product_rows = df[df['Clothing ID'] == clothing_id]
                if not product_rows.empty:
                    product = product_rows.iloc[0]
                    prediction = ml_predictor.predict_recommendation(
                        title, review_text, rating,
                        product.get('Division Name', 'General'),
                        product.get('Department Name', 'Tops'),
                        product.get('Class Name', 'Blouses')
                    )
                    
                    pred_text = "recommend" if prediction['prediction'] == 1 else "not recommend"
                    confidence = prediction.get('confidence', 0.5)
                    flash(f'Review submitted successfully! Our AI predicts you would {pred_text} this product (confidence: {confidence:.1%})', 'success')
                else:
                    flash('Review submitted successfully!', 'success')
            else:
                flash('Review submitted successfully!', 'success')
        except Exception as pred_error:
            print(f"Error in ML prediction: {pred_error}")
            flash('Review submitted successfully!', 'success')
        
        return redirect(url_for('product.product_detail', clothing_id=clothing_id))
        
    except Exception as e:
        print(f"Error submitting review: {e}")
        flash('Error submitting review. Please try again.', 'error')
        return redirect(url_for('main.home'))

@product_bp.route('/api/reviews/<int:clothing_id>')
def get_product_reviews_api(clothing_id):
    """API endpoint to get product reviews with pagination"""
    try:
        from backend.services.data_manager import get_product_reviews
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        filter_type = request.args.get('filter', 'all')  # all, recommended, new
        sort_by = request.args.get('sort', 'newest')  # newest, rating, helpful
        
        # Get all reviews for this product
        all_reviews = get_product_reviews(clothing_id)
        
        if not all_reviews:
            return jsonify({
                'reviews': [],
                'total_count': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0,
                'has_more': False
            })
        
        # Apply filtering
        filtered_reviews = all_reviews
        if filter_type == 'recommended':
            filtered_reviews = [r for r in all_reviews if r.get('Recommended IND') == 1]
        elif filter_type == 'new':
            filtered_reviews = [r for r in all_reviews if r.get('Is New')]
        
        # Apply sorting
        if sort_by == 'rating':
            filtered_reviews = sorted(filtered_reviews, key=lambda x: x.get('Rating', 0), reverse=True)
        elif sort_by == 'helpful':
            # For now, sort by rating as we don't have helpful votes stored
            filtered_reviews = sorted(filtered_reviews, key=lambda x: x.get('Rating', 0), reverse=True)
        else:  # newest
            # If we have date added, use it; otherwise keep original order
            if any(r.get('Date Added') for r in filtered_reviews):
                filtered_reviews = sorted(filtered_reviews, key=lambda x: x.get('Date Added', ''), reverse=True)
        
        # Apply pagination
        total_count = len(filtered_reviews)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_reviews = filtered_reviews[start_idx:end_idx]
        
        total_pages = (total_count + per_page - 1) // per_page
        has_more = page < total_pages
        
        # Format reviews for JSON response
        formatted_reviews = []
        for review in paginated_reviews:
            formatted_reviews.append({
                'title': review.get('Title', 'Review'),
                'rating': review.get('Rating', 0),
                'review_text': review.get('Review Text', ''),
                'recommended': review.get('Recommended IND', 0),
                'date_added': review.get('Date Added', ''),
                'age': review.get('Age', ''),
                'is_new': review.get('Is New', False)
            })
        
        return jsonify({
            'reviews': formatted_reviews,
            'total_count': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'has_more': has_more
        })
        
    except Exception as e:
        print(f"Error in get reviews API: {e}")
        return jsonify({'error': 'Failed to load reviews'}), 500
