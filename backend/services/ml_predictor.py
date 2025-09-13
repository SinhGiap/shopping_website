
"""
Machine Learning Prediction Service
Rewritten: Three models (Logistic Regression, AdaBoost, XGBoost) using BoW of title+text and structured features
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
from backend.services.text_processor import TextProcessor
import os
import joblib

class MLPredictor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.models = {}
        self.vectorizer = None
        self.scaler = None
        self.encoder = None
        self.is_loaded = False
        self.df = None

    def load_models_and_data(self, csv_path="assignment3_II.csv"):
        print("DEBUG: Entered load_models_and_data")
        try:
            # Try to load models and vectorizers from disk
            model_dir = "models"
            logreg_path = os.path.join(model_dir, "logreg_model.pkl")
            adaboost_path = os.path.join(model_dir, "adaboost_model.pkl")
            xgboost_path = os.path.join(model_dir, "xgboost_model.pkl")
            vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            encoder_path = os.path.join(model_dir, "encoder.pkl")
            if (os.path.exists(logreg_path) and os.path.exists(adaboost_path) and os.path.exists(xgboost_path)
                and os.path.exists(vectorizer_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
                self.models['logreg'] = joblib.load(logreg_path)
                self.models['adaboost'] = joblib.load(adaboost_path)
                self.models['xgboost'] = joblib.load(xgboost_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.scaler = joblib.load(scaler_path)
                self.encoder = joblib.load(encoder_path)
                self.is_loaded = True
                print("Models and vectorizers loaded from disk!")
                # Always load dataframe from CSV for downstream use
                if os.path.exists(csv_path):
                    self.df = pd.read_csv(csv_path)
                    self.df = self._clean_dataset(self.df)
                else:
                    print(f"[ERROR] Dataset not found: {csv_path}")
                    self.df = None
                return
            # Load dataset and train if not found
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Dataset not found: {csv_path}")
            self.df = pd.read_csv(csv_path)
            self.df = self._clean_dataset(self.df)
            print("DEBUG: Calling _prepare_features_and_target")
            # Prepare features
            X, y = self._prepare_features_and_target(self.df)
            # Train models
            self.models['logreg'] = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42)
            self.models['logreg'].fit(X, y)
            self.models['adaboost'] = AdaBoostClassifier(random_state=42)
            self.models['adaboost'].fit(X, y)
            self.models['xgboost'] = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
            self.models['xgboost'].fit(X, y)
            # Save models and vectorizers to disk
            joblib.dump(self.models['logreg'], logreg_path)
            joblib.dump(self.models['adaboost'], adaboost_path)
            joblib.dump(self.models['xgboost'], xgboost_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.encoder, encoder_path)
            self.is_loaded = True
            print("Models trained and saved to disk!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False

    def _clean_dataset(self, df):
        df['Review Text'] = df['Review Text'].fillna("")
        df['Title'] = df['Title'].fillna("")
        df['Rating'] = df['Rating'].fillna(df['Rating'].median())
        df['Recommended IND'] = df['Recommended IND'].fillna(1)
        for col in ['Division Name', 'Department Name', 'Class Name']:
            df[col] = df[col].fillna('General')
        return df

    def _prepare_features_and_target(self, df):
        # Combine title and review text for all reviews
        combined_raw = [str(title) + " " + str(text) for title, text in zip(df['Title'], df['Review Text'])]
        print(f"DEBUG: combined_raw sample: {combined_raw[:3]}")
        # Preprocess all reviews together (corpus-wide steps applied)
        processed_reviews = self.text_processor.preprocess_text(combined_raw)
        print(f"DEBUG: processed_reviews type: {type(processed_reviews)}")
        if processed_reviews is None:
            raise ValueError("Processed reviews is None. Check preprocessing pipeline.")
        # Join tokens for each review
        combined_texts = [" ".join(tokens) if tokens else "good product" for tokens in processed_reviews]
        print(f"DEBUG: combined_texts sample: {combined_texts[:3]}")
        # If all combined_texts are empty, fill with default
        if not any(txt.strip() for txt in combined_texts):
            print("DEBUG: All combined_texts are empty. Using fallback.")
            combined_texts = ["good product" for _ in combined_texts]
        # Bag-of-words vectorizer
        self.vectorizer = CountVectorizer(max_features=1000, min_df=1)
        X_bow = self.vectorizer.fit_transform(combined_texts)
        # # Debug: print vocabulary size and sample
        # vocab_size = len(self.vectorizer.vocabulary_)
        # print(f"DEBUG: BoW vocabulary size: {vocab_size}")
        # if vocab_size == 0:
        #     print("DEBUG: BoW vocabulary is EMPTY! Check stopwords and preprocessing.")
        # else:
        #     sample_vocab = list(self.vectorizer.vocabulary_.keys())[:20]
        #     print(f"DEBUG: Sample BoW vocab: {sample_vocab}")
        # Structured features
        struct_df = df[['Rating', 'Division Name', 'Department Name', 'Class Name']].copy()
        self.scaler = StandardScaler()
        X_num = self.scaler.fit_transform(struct_df[['Rating']])
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_cat = self.encoder.fit_transform(struct_df[['Division Name', 'Department Name', 'Class Name']])
        X_struct = hstack([csr_matrix(X_num), X_cat])
        # Combine all features
        X = hstack([X_bow, X_struct]).tocsr()
        y = df['Recommended IND'].values
        return X, y

    def predict_recommendation(self, review_title, review_text, rating=5, division="General", department="Tops", class_name="Blouses"):
        print(f"[DEBUG] predict_recommendation called with review_title: {review_title}")
        print(f"[DEBUG] predict_recommendation called with review_text: {review_text}")
        print(f"[DEBUG] rating: {rating}, division: {division}, department: {department}, class_name: {class_name}")
        if not self.is_loaded:
            return {'prediction': 1, 'confidence': 0.75, 'model_predictions': {}, 'status': 'fallback'}
        try:
            # Prepare input features
            print("[DEBUG] Preprocessing review_title...")
            tokens_title = self.text_processor.preprocess_text(str(review_title))
            print(f"[DEBUG] tokens_title: {tokens_title}")
            print("[DEBUG] Preprocessing review_text...")
            tokens_text = self.text_processor.preprocess_text(str(review_text))
            print(f"[DEBUG] tokens_text: {tokens_text}")
            tokens = []
            if isinstance(tokens_title, list):
                tokens += [t for t in tokens_title if isinstance(t, str) and t.strip()]
            if isinstance(tokens_text, list):
                tokens += [t for t in tokens_text if isinstance(t, str) and t.strip()]
            combined = " ".join(tokens) if tokens else "good product"
            print(f"[DEBUG] Combined tokens: {combined}")
            if not combined.strip():
                combined = "good product"
            X_bow = self.vectorizer.transform([combined])
            print(f"[DEBUG] X_bow shape: {X_bow.shape}")
            struct_data = pd.DataFrame({
                'Rating': [float(rating)],
                'Division Name': [str(division)],
                'Department Name': [str(department)],
                'Class Name': [str(class_name)]
            })
            X_num = self.scaler.transform(struct_data[['Rating']])
            X_cat = self.encoder.transform(struct_data[['Division Name', 'Department Name', 'Class Name']])
            X_struct = hstack([csr_matrix(X_num), X_cat])
            print(f"[DEBUG] X_struct shape: {X_struct.shape}")
            X = hstack([X_bow, X_struct]).tocsr()
            print(f"[DEBUG] Final feature shape: {X.shape}")
            # Predict with all models
            preds = {}
            confs = {}
            for name, model in self.models.items():
                print(f"[DEBUG] Predicting with model: {name}")
                pred = model.predict(X)[0]
                if hasattr(model, "predict_proba"):
                    conf = float(np.max(model.predict_proba(X)[0]))
                else:
                    conf = 0.75
                preds[name] = int(pred)
                confs[name] = conf
                print(f"[DEBUG] Prediction result for {name}: {preds[name]}, confidence: {conf}")
            # Majority vote
            votes = sum(preds.values())
            final_pred = 1 if votes >= 2 else 0
            avg_conf = float(np.mean(list(confs.values())))
            print(f"[DEBUG] Final prediction: {final_pred}, avg_conf: {avg_conf}")
            return {
                'prediction': final_pred,
                'confidence': round(avg_conf, 3),
                'model_predictions': preds,
                'model_confidences': confs,
                'status': 'success'
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'prediction': 1,
                'confidence': 0.5,
                'model_predictions': {},
                'error': str(e),
                'status': 'error'
            }