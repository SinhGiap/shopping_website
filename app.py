"""
Enhanced Milestone II: Flask Shopping Website
A comprehensive online shopping website with ML-powered review prediction
Enhanced version with improved error handling and features
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import re
from collections import Counter
from itertools import chain
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data if not present"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]
    
    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading {name}...")
            nltk.download(name)

# Initialize NLTK data
download_nltk_data()

app = Flask(__name__)
app.secret_key = 'enhanced_milestone2_secret_key_for_shopping_website'

class TextProcessor:
    """Enhanced text preprocessing pipeline from Milestone I Task 1"""
    
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords with fallback
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self):
        """Load stopwords with multiple fallback options"""
        # Try custom stopwords file
        stopwords_paths = [
            "stopwords_en.txt",
            "../task1/stopwords_en.txt",
            "../../task1/stopwords_en.txt"
        ]
        
        for path in stopwords_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return set(f.read().splitlines())
            except FileNotFoundError:
                continue
        
        # Fallback to NLTK stopwords
        try:
            return set(stopwords.words('english'))
        except:
            # Final fallback to basic stopwords
            return set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
                       'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
                       'that', 'the', 'to', 'was', 'will', 'with', 'the'])
    
    def preprocess_text(self, text):
        """Apply the same preprocessing pipeline from Task 1"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Tokenization
            tokens = self.tokenizer.tokenize(text)
            
            # Lowercase
            tokens = [token.lower() for token in tokens]
            
            # Remove short words
            tokens = [token for token in tokens if len(token) >= 2]
            
            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stopwords]
            
            # Lemmatization
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return " ".join(tokens)
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return text.lower()

class MLPredictor:
    """Enhanced Machine Learning prediction system using models from Milestone I"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
        self.is_loaded = False
        self.df = None
        
    def load_models_and_data(self):
        """Load trained models and prepare data with enhanced error handling"""
        try:
            # Try multiple paths for the dataset
            dataset_paths = [
                "assignment3_II.csv",
                "../assignment3_II.csv",
                "../../assignment3_II.csv"
            ]
            
            for path in dataset_paths:
                try:
                    self.df = pd.read_csv(path)
                    print(f"Dataset loaded from: {path}")
                    break
                except FileNotFoundError:
                    continue
            
            if self.df is None:
                raise FileNotFoundError("Could not find assignment3_II.csv in any expected location")
            
            # Clean the dataset
            self.df = self._clean_dataset(self.df)
            
            # Prepare features for different models
            self._prepare_bow_model()
            self._prepare_title_struct_model()
            self._prepare_ensemble_model()
            
            self.is_loaded = True
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Create a minimal fallback dataset
            self._create_fallback_dataset()
            self.is_loaded = False
    
    def _clean_dataset(self, df):
        """Clean and validate the dataset"""
        # Fill missing values
        df['Review Text'] = df['Review Text'].fillna('')
        df['Title'] = df['Title'].fillna('')
        df['Clothes Title'] = df['Clothes Title'].fillna('Product')
        df['Clothes Description'] = df['Clothes Description'].fillna('Description not available')
        df['Rating'] = df['Rating'].fillna(df['Rating'].median())
        df['Recommended IND'] = df['Recommended IND'].fillna(1)
        
        # Fill categorical columns
        for col in ['Division Name', 'Department Name', 'Class Name']:
            df[col] = df[col].fillna('General')
        
        return df
    
    def _create_fallback_dataset(self):
        """Create a minimal fallback dataset for demo purposes"""
        self.df = pd.DataFrame({
            'Clothing ID': [1, 2, 3, 4, 5],
            'Title': ['Great product', 'Good quality', 'Nice fit', 'Comfortable', 'Stylish'],
            'Review Text': ['Very good', 'High quality', 'Perfect fit', 'Very comfortable', 'Love the style'],
            'Rating': [5, 4, 5, 4, 5],
            'Recommended IND': [1, 1, 1, 1, 1],
            'Clothes Title': ['Basic T-Shirt', 'Cotton Pants', 'Summer Dress', 'Wool Sweater', 'Denim Jacket'],
            'Clothes Description': ['Comfortable cotton t-shirt', 'Classic cotton pants', 'Light summer dress', 'Warm wool sweater', 'Classic denim jacket'],
            'Division Name': ['General', 'General', 'General', 'General', 'General'],
            'Department Name': ['Tops', 'Bottoms', 'Dresses', 'Tops', 'Jackets'],
            'Class Name': ['T-Shirts', 'Pants', 'Dresses', 'Sweaters', 'Jackets'],
            'Age': [25, 30, 35, 40, 45]
        })
    
    def _prepare_bow_model(self):
        """Prepare Bag-of-Words model with error handling"""
        try:
            # Process review texts
            processed_texts = [self.text_processor.preprocess_text(str(text)) 
                             for text in self.df["Review Text"]]
            
            # Remove empty texts
            processed_texts = [text if text.strip() else "good product" for text in processed_texts]
            
            # Create CountVectorizer for BoW
            self.vectorizers['bow'] = CountVectorizer(
                max_features=min(5000, len(processed_texts)), 
                min_df=1,
                max_df=0.95
            )
            X_bow = self.vectorizers['bow'].fit_transform(processed_texts)
            
            # Train logistic regression
            y = self.df["Recommended IND"].values
            self.models['bow'] = LogisticRegression(
                max_iter=1000, 
                solver="liblinear", 
                random_state=42, 
                class_weight="balanced"
            )
            self.models['bow'].fit(X_bow, y)
            
            print("BoW model prepared successfully")
            
        except Exception as e:
            print(f"Error preparing BoW model: {e}")
    
    def _prepare_title_struct_model(self):
        """Prepare Title + Structured features model with error handling"""
        try:
            # Prepare title features
            titles = [str(title) for title in self.df["Title"]]
            self.vectorizers['title'] = CountVectorizer(
                max_features=min(2000, len(titles)), 
                min_df=1,
                max_df=0.95
            )
            X_title = self.vectorizers['title'].fit_transform(titles)
            
            # Prepare structured features
            struct_df = self.df[["Rating", "Division Name", "Department Name", "Class Name"]].copy()
            struct_df["Rating"] = struct_df["Rating"].fillna(struct_df["Rating"].median())
            for col in ["Division Name", "Department Name", "Class Name"]:
                struct_df[col] = struct_df[col].fillna("General")
            
            # Scale numeric features
            self.scalers['struct'] = StandardScaler()
            X_num = self.scalers['struct'].fit_transform(struct_df[["Rating"]])
            
            # Encode categorical features
            self.encoders['struct'] = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            X_cat = self.encoders['struct'].fit_transform(
                struct_df[["Division Name", "Department Name", "Class Name"]]
            )
            
            # Combine features
            X_struct = hstack([csr_matrix(X_num), X_cat])
            X_combined = hstack([X_title, X_struct]).tocsr()
            
            # Train model
            y = self.df["Recommended IND"].values
            self.models['title_struct'] = LogisticRegression(
                max_iter=1000, 
                solver="liblinear", 
                random_state=42, 
                class_weight="balanced"
            )
            self.models['title_struct'].fit(X_combined, y)
            
            print("Title + Structured features model prepared successfully")
            
        except Exception as e:
            print(f"Error preparing Title + Struct model: {e}")
    
    def _prepare_ensemble_model(self):
        """Prepare ensemble prediction weights"""
        # Based on Milestone I results, BoW performed best
        self.ensemble_weights = {
            'bow': 0.7,
            'title_struct': 0.3
        }
    
    def predict_recommendation(self, review_title, review_text, rating=5, 
                             division="General", department="Tops", class_name="Blouses"):
        """Predict recommendation for a new review with enhanced error handling"""
        if not self.is_loaded or not self.models:
            return {'prediction': 1, 'confidence': 0.75, 'model_predictions': {}, 'status': 'fallback'}
        
        try:
            predictions = {}
            confidences = {}
            
            # BoW prediction
            if 'bow' in self.models and 'bow' in self.vectorizers:
                try:
                    processed_text = self.text_processor.preprocess_text(str(review_text))
                    if not processed_text.strip():
                        processed_text = "good product"
                    
                    X_bow = self.vectorizers['bow'].transform([processed_text])
                    pred_bow = self.models['bow'].predict(X_bow)[0]
                    prob_bow = self.models['bow'].predict_proba(X_bow)[0]
                    predictions['bow'] = int(pred_bow)
                    confidences['bow'] = float(max(prob_bow))
                except Exception as e:
                    print(f"Error in BoW prediction: {e}")
            
            # Title + Struct prediction
            if 'title_struct' in self.models and 'title' in self.vectorizers:
                try:
                    # Title features
                    X_title = self.vectorizers['title'].transform([str(review_title)])
                    
                    # Structured features
                    struct_data = pd.DataFrame({
                        'Rating': [float(rating)],
                        'Division Name': [str(division)],
                        'Department Name': [str(department)],
                        'Class Name': [str(class_name)]
                    })
                    
                    X_num = self.scalers['struct'].transform(struct_data[["Rating"]])
                    X_cat = self.encoders['struct'].transform(
                        struct_data[["Division Name", "Department Name", "Class Name"]]
                    )
                    X_struct = hstack([csr_matrix(X_num), X_cat])
                    X_combined = hstack([X_title, X_struct]).tocsr()
                    
                    pred_title_struct = self.models['title_struct'].predict(X_combined)[0]
                    prob_title_struct = self.models['title_struct'].predict_proba(X_combined)[0]
                    predictions['title_struct'] = int(pred_title_struct)
                    confidences['title_struct'] = float(max(prob_title_struct))
                except Exception as e:
                    print(f"Error in Title+Struct prediction: {e}")
            
            # Ensemble prediction
            if predictions:
                weighted_pred = sum(
                    predictions[model] * self.ensemble_weights.get(model, 0.5)
                    for model in predictions
                )
                ensemble_pred = 1 if weighted_pred >= 0.5 else 0
                
                # Calculate ensemble confidence
                weighted_conf = sum(
                    confidences[model] * self.ensemble_weights.get(model, 0.5)
                    for model in confidences
                ) / len(confidences) if confidences else 0.75
                
                return {
                    'prediction': int(ensemble_pred),
                    'confidence': round(float(weighted_conf), 3),
                    'model_predictions': predictions,
                    'model_confidences': confidences,
                    'status': 'success'
                }
            else:
                # Fallback prediction based on rating
                fallback_pred = 1 if rating >= 4 else 0
                return {
                    'prediction': fallback_pred, 
                    'confidence': 0.6, 
                    'model_predictions': {},
                    'status': 'fallback'
                }
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Smart fallback based on rating
            fallback_pred = 1 if rating >= 4 else 0
            return {
                'prediction': fallback_pred, 
                'confidence': 0.5, 
                'model_predictions': {},
                'error': str(e),
                'status': 'error'
            }

class SearchEngine:
    """Enhanced search functionality with improved fuzzy matching"""
    
    def __init__(self, df):
        self.df = df if df is not None else pd.DataFrame()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        
    def fuzzy_search(self, query, limit=50):
        """Enhanced search with better fuzzy matching - returns unique products"""
        if self.df.empty:
            return pd.DataFrame()
            
        if not query or not query.strip():
            # Return unique products for empty query
            unique_products = self._get_unique_products(self.df.head(limit * 10))
            return unique_products.head(limit)
        
        query = str(query).lower().strip()
        
        # Tokenize and lemmatize query
        try:
            query_tokens = self.tokenizer.tokenize(query)
            query_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in query_tokens]
        except:
            query_tokens = query.split()
        
        # Search ONLY in product name field
        search_fields = {
            'Clothes Title': 10       # Product name ONLY - no other fields
        }
        
        # Score products, not individual reviews
        product_scores = {}
        for idx, row in self.df.iterrows():
            clothing_id = row['Clothing ID']
            score = 0
            
            # Search in each field with weighted scoring
            for field, weight in search_fields.items():
                if field not in row or pd.isna(row[field]):
                    continue
                    
                field_text = str(row[field]).lower()
                
                # Direct substring match (highest score)
                if query in field_text:
                    score += weight * 10
                    continue
                
                # Token-based matching
                try:
                    field_tokens = self.tokenizer.tokenize(field_text)
                    field_tokens = [self.lemmatizer.lemmatize(token) for token in field_tokens]
                except:
                    field_tokens = field_text.split()
                
                # Calculate match score for this field
                field_score = 0
                for query_token in query_tokens:
                    # Exact token match
                    if query_token in field_tokens:
                        field_score += 3
                    # Partial match
                    elif any(query_token in token or token in query_token 
                           for token in field_tokens if len(token) > 2):
                        field_score += 2
                    # Fuzzy match for common variations
                    elif self._fuzzy_match(query_token, field_tokens):
                        field_score += 1
                
                score += field_score * weight
            
            # Only include products with a meaningful score (filter out weak matches)
            if score >= 5:  # Minimum score threshold
                # Keep the highest score for each product
                if clothing_id not in product_scores or score > product_scores[clothing_id][1]:
                    product_scores[clothing_id] = (idx, score)
        
        # Sort products by score and get unique products
        if not product_scores:
            return pd.DataFrame()
        
        sorted_products = sorted(product_scores.values(), key=lambda x: x[1], reverse=True)
        result_indices = [idx for idx, score in sorted_products[:limit]]
        matched_reviews = self.df.loc[result_indices].copy()
        
        # Return unique products with aggregated data
        return self._get_unique_products(matched_reviews)
    
    def _fuzzy_match(self, query_token, field_tokens):
        """Enhanced fuzzy matching for fashion terms"""
        variations = {
            'dress': ['dresses', 'dress', 'gown', 'frock'],
            'shoe': ['shoes', 'shoe', 'footwear', 'sneaker', 'boot', 'sandal'],
            'pant': ['pants', 'pant', 'trouser', 'trousers', 'jean', 'jeans'],
            'shirt': ['shirts', 'shirt', 'top', 'tops', 'blouse', 'tee'],
            'jean': ['jeans', 'jean', 'denim'],
            'skirt': ['skirts', 'skirt', 'mini', 'maxi'],
            'jacket': ['jackets', 'jacket', 'coat', 'coats', 'blazer'],
            'sweater': ['sweaters', 'sweater', 'jumper', 'pullover', 'cardigan'],
            'bag': ['bags', 'bag', 'purse', 'handbag', 'tote'],
            'accessory': ['accessories', 'accessory', 'jewelry', 'belt', 'scarf']
        }
        
        # Check if query matches any variations
        for base, variants in variations.items():
            if query_token in variants:
                return any(variant in field_tokens for variant in variants)
        
        return False
    
    def _get_unique_products(self, df_subset):
        """Convert reviews dataframe to unique products with aggregated data"""
        if df_subset.empty:
            return pd.DataFrame()
        
        # Group by Clothing ID and aggregate data - focus on product info only
        unique_products = df_subset.groupby('Clothing ID').agg({
            'Clothes Title': 'first',        # Product name
            'Clothes Description': 'first',  # Product description
            'Class Name': 'first',           # Product category
            'Department Name': 'first',      # Department
            'Division Name': 'first',        # Division
            'Rating': 'mean',                # Average rating across all reviews
            'Recommended IND': 'mean'        # Percentage recommended
        }).round({'Rating': 1, 'Recommended IND': 2})
        
        # Reset index to make Clothing ID a column again
        unique_products = unique_products.reset_index()
        
        # Add review count for display purposes
        review_counts = df_subset['Clothing ID'].value_counts()
        unique_products['Review Count'] = unique_products['Clothing ID'].map(review_counts)
        
        # Handle duplicate product names by adding unique identifiers
        unique_products = self._handle_duplicate_names(unique_products)
        
        return unique_products
    
    def _handle_duplicate_names(self, df):
        """Handle duplicate product names by adding distinguishing information"""
        if df.empty:
            return df
        
        # Create a copy to work with
        result_df = df.copy()
        
        # Find products with duplicate names
        name_counts = result_df['Clothes Title'].value_counts()
        duplicate_names = name_counts[name_counts > 1].index
        
        for name in duplicate_names:
            # Get all products with this duplicate name
            mask = result_df['Clothes Title'] == name
            duplicate_products = result_df[mask].copy()
            
            # Sort by Clothing ID for consistent ordering
            duplicate_products = duplicate_products.sort_values('Clothing ID')
            
            # Add distinguishing information to product names
            for idx, (df_idx, product) in enumerate(duplicate_products.iterrows()):
                # Create unique identifier based on ID and characteristics
                clothing_id = product['Clothing ID']
                rating = product['Rating']
                review_count = product['Review Count']
                
                # Create a more descriptive name
                unique_name = f"{name} (ID: {clothing_id}, ⭐{rating}, {review_count} reviews)"
                
                # Update the product name in the main dataframe
                result_df.loc[df_idx, 'Display Title'] = unique_name
                result_df.loc[df_idx, 'Original Title'] = name
        
        # For products without duplicates, keep original name
        no_duplicates_mask = ~result_df['Clothes Title'].isin(duplicate_names)
        result_df.loc[no_duplicates_mask, 'Display Title'] = result_df.loc[no_duplicates_mask, 'Clothes Title']
        result_df.loc[no_duplicates_mask, 'Original Title'] = result_df.loc[no_duplicates_mask, 'Clothes Title']
        
        return result_df
    
    def filter_by_category(self, division=None, department=None, class_name=None, limit=100):
        """Enhanced category filtering"""
        if self.df.empty:
            return pd.DataFrame()
            
        filtered_df = self.df.copy()
        
        try:
            if division and division.lower() != 'all':
                filtered_df = filtered_df[
                    filtered_df['Division Name'].str.contains(str(division), case=False, na=False)
                ]
            if department and department.lower() != 'all':
                filtered_df = filtered_df[
                    filtered_df['Department Name'].str.contains(str(department), case=False, na=False)
                ]
            if class_name and class_name.lower() != 'all':
                filtered_df = filtered_df[
                    filtered_df['Class Name'].str.contains(str(class_name), case=False, na=False)
                ]
        except Exception as e:
            print(f"Error in category filtering: {e}")
        
        # Return unique products instead of individual reviews
        return self._get_unique_products(filtered_df.head(limit * 10)).head(limit)
    
    def get_featured_items(self, limit=6):
        """Get featured items with highest ratings - returns unique products"""
        if self.df.empty:
            return []
            
        try:
            # Get unique products with highest average ratings
            unique_products = self._get_unique_products(self.df)
            featured = unique_products.nlargest(limit, 'Rating')
            
            # Use Display Title if available, otherwise fall back to Clothes Title
            title_column = 'Display Title' if 'Display Title' in featured.columns else 'Clothes Title'
            
            return featured[['Clothing ID', title_column, 'Clothes Description', 
                           'Rating', 'Class Name', 'Department Name', 'Review Count']].rename(
                           columns={title_column: 'Clothes Title'}).to_dict('records')
        except Exception as e:
            print(f"Error getting featured items: {e}")
            return []

# Initialize global objects
ml_predictor = MLPredictor()
search_engine = None
df = None

def initialize_app():
    """Initialize ML models and search engine with enhanced error handling"""
    global search_engine, df
    
    try:
        # Load ML models and data
        ml_predictor.load_models_and_data()
        df = ml_predictor.df
        
        # Initialize search engine
        search_engine = SearchEngine(df)
        
        print("Application initialized successfully!")
        print(f"Dataset loaded with {len(df)} records")
        
    except Exception as e:
        print(f"Error initializing application: {e}")
        # Create fallback search engine
        search_engine = SearchEngine(pd.DataFrame())

@app.route('/')
def home():
    """Enhanced home page with better error handling"""
    try:
        if search_engine and not search_engine.df.empty:
            # Get featured items
            featured_items = search_engine.get_featured_items(6)
            
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

@app.route('/search')
def search():
    """Enhanced search page with better pagination and error handling"""
    query = request.args.get('q', '').strip()
    division = request.args.get('division', '').strip()
    department = request.args.get('department', '').strip()
    class_name = request.args.get('class', '').strip()
    page = int(request.args.get('page', 1))
    per_page = 12
    
    try:
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

@app.route('/product/<int:clothing_id>')
def product_detail(clothing_id):
    """Enhanced product detail page"""
    try:
        if df is None or df.empty:
            flash('Product database not available', 'error')
            return redirect(url_for('home'))
        
        # Get product details
        product_rows = df[df['Clothing ID'] == clothing_id]
        if product_rows.empty:
            flash('Product not found', 'error')
            return redirect(url_for('home'))
        
        product = product_rows.iloc[0]
        
        # Get all reviews for this product
        reviews = df[df['Clothing ID'] == clothing_id][
            ['Title', 'Review Text', 'Rating', 'Recommended IND', 'Age']
        ].to_dict('records')
        
        # Calculate statistics
        ratings = df[df['Clothing ID'] == clothing_id]['Rating']
        avg_rating = float(ratings.mean()) if not ratings.empty else 0
        review_count = len(reviews)
        
        return render_template('product.html',
                             product=product.to_dict(),
                             reviews=reviews,
                             avg_rating=round(avg_rating, 1),
                             review_count=review_count)
        
    except Exception as e:
        print(f"Error in product detail route: {e}")
        flash('Error loading product details', 'error')
        return redirect(url_for('home'))

@app.route('/add_review/<int:clothing_id>')
def add_review_form(clothing_id):
    """Enhanced form to add new review"""
    try:
        if df is None or df.empty:
            flash('Product database not available', 'error')
            return redirect(url_for('home'))
        
        product_rows = df[df['Clothing ID'] == clothing_id]
        if product_rows.empty:
            flash('Product not found', 'error')
            return redirect(url_for('home'))
        
        product = product_rows.iloc[0]
        return render_template('add_review.html', product=product.to_dict())
        
    except Exception as e:
        print(f"Error in add review form: {e}")
        flash('Error loading review form', 'error')
        return redirect(url_for('home'))

@app.route('/predict_recommendation', methods=['POST'])
def predict_recommendation():
    """Enhanced API endpoint for ML prediction"""
    try:
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
        prediction_result = ml_predictor.predict_recommendation(
            review_title, review_text, rating, division, department, class_name
        )
        
        return jsonify(prediction_result)
        
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({
            'prediction': 1 if int(request.form.get('rating', 5)) >= 4 else 0, 
            'confidence': 0.5, 
            'error': str(e),
            'status': 'error'
        })

@app.route('/submit_review', methods=['POST'])
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
            return redirect(url_for('add_review_form', clothing_id=clothing_id))
        
        if rating < 1 or rating > 5:
            flash('Rating must be between 1 and 5', 'error')
            return redirect(url_for('add_review_form', clothing_id=clothing_id))
        
        # In a real application, you would save this to a database
        # For this demo, we'll show success message with prediction
        
        # Get ML prediction for the review
        try:
            if df is not None and not df.empty:
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
                    flash(f'Review submitted! Our AI predicts you would {pred_text} this product (confidence: {confidence:.1%})', 'success')
                else:
                    flash('Review submitted successfully!', 'success')
            else:
                flash('Review submitted successfully!', 'success')
        except:
            flash('Review submitted successfully!', 'success')
        
        return redirect(url_for('product_detail', clothing_id=clothing_id))
        
    except Exception as e:
        print(f"Error submitting review: {e}")
        flash('Error submitting review. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/api/categories')
def get_categories():
    """Enhanced API endpoint to get category data"""
    try:
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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'ml_loaded': ml_predictor.is_loaded,
        'dataset_size': len(df) if df is not None else 0,
        'search_available': search_engine is not None and not search_engine.df.empty
    }
    return jsonify(status)

if __name__ == '__main__':
    print("Starting Enhanced Shopping Website...")
    print("Initializing ML models and search engine...")
    
    # Initialize the application
    initialize_app()
    
    print("Application ready!")
    print("Access the website at: http://127.0.0.1:5002")
    
    app.run(debug=True, port=5002, host='127.0.0.1')
