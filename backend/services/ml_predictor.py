"""
Machine Learning Prediction Service
Enhanced ML prediction system using models from Milestone I
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix

from backend.services.text_processor import TextProcessor
from backend.config.settings import Config

class MLPredictor:
    """Enhanced Machine Learning prediction system"""
    
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
            for path in Config.DATASET_PATHS:
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
                max_features=min(Config.MAX_FEATURES_BOW, len(processed_texts)), 
                min_df=Config.MIN_DF,
                max_df=Config.MAX_DF
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
                max_features=min(Config.MAX_FEATURES_TITLE, len(titles)), 
                min_df=Config.MIN_DF,
                max_df=Config.MAX_DF
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
