"""
Main Routes Blueprint
Handles the main web pages (home, search, etc.)
"""

from flask import Blueprint, render_template, request
from backend.config.settings import Config

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    """Enhanced home page with better error handling"""
    try:
        from backend.services.data_manager import get_search_engine, get_dataframe
        
        search_engine = get_search_engine()
        df = get_dataframe()
        
        if search_engine and not search_engine.df.empty:
            # Get featured items
            featured_items = search_engine.get_featured_items(Config.FEATURED_ITEMS_LIMIT)
            
            # Get categories for navigation
            divisions = sorted(df['Division Name'].dropna().unique())
            departments = sorted(df['Department Name'].dropna().unique())
            classes = sorted(df['Class Name'].dropna().unique())
        else:
            # Fallback for empty dataset
            featured_items = []
            divisions = ['General', 'Petite']
            departments = ['Tops', 'Bottoms', 'Dresses']
            classes = ['Shirts', 'Pants', 'Dresses']
        
        return render_template('home.html', 
                             featured_items=featured_items,
                             divisions=divisions,
                             departments=departments,
                             classes=classes)
    except Exception as e:
        print(f"Error in home route: {e}")
        return render_template('home.html', 
                             featured_items=[], 
                             divisions=['General'], 
                             departments=['Tops'], 
                             classes=['Shirts'])

@main_bp.route('/search')
def search():
    """Enhanced search page with better pagination and error handling"""
    query = request.args.get('q', '').strip()
    division = request.args.get('division', '').strip()
    department = request.args.get('department', '').strip()
    class_name = request.args.get('class', '').strip()
    page = int(request.args.get('page', 1))
    per_page = Config.SEARCH_PER_PAGE
    
    try:
        from backend.services.data_manager import get_search_engine
        
        search_engine = get_search_engine()
        
        if not search_engine or search_engine.df.empty:
            return render_template('search.html', 
                                 results=[], query=query, total_results=0,
                                 page=1, total_pages=0, has_prev=False, has_next=False,
                                 error="Search functionality temporarily unavailable")
        
        # Perform search
        if query:
            results = search_engine.fuzzy_search(query, limit=1000)
        else:
            results = search_engine.filter_by_category(division, department, class_name, limit=1000)
        
        # Handle empty results
        if results.empty:
            return render_template('search.html',
                                 results=[], query=query, total_results=0,
                                 page=1, total_pages=0, has_prev=False, has_next=False,
                                 division=division, department=department, class_name=class_name)
        
        # Pagination
        total_results = len(results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = results.iloc[start_idx:end_idx]
        
        # Calculate pagination info
        total_pages = max(1, (total_results + per_page - 1) // per_page)
        has_prev = page > 1
        has_next = page < total_pages
        
        return render_template('search.html',
                             results=paginated_results.to_dict('records'),
                             query=query,
                             total_results=total_results,
                             page=page,
                             total_pages=total_pages,
                             has_prev=has_prev,
                             has_next=has_next,
                             division=division,
                             department=department,
                             class_name=class_name)
        
    except Exception as e:
        print(f"Error in search route: {e}")
        return render_template('search.html', 
                             results=[], query=query, total_results=0,
                             page=1, total_pages=0, has_prev=False, has_next=False,
                             error=f"Search error: {str(e)}")

@main_bp.route('/health')
def health_check():
    """Health check endpoint"""
    from backend.services.data_manager import get_ml_predictor, get_dataframe, get_search_engine
    
    ml_predictor = get_ml_predictor()
    df = get_dataframe()
    search_engine = get_search_engine()
    
    status = {
        'status': 'healthy',
        'ml_loaded': ml_predictor.is_loaded if ml_predictor else False,
        'dataset_size': len(df) if df is not None else 0,
        'search_available': search_engine is not None and not search_engine.df.empty
    }
    return status
