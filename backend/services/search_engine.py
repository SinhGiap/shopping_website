"""
Search Engine Service
 search functionality with improved fuzzy matching
"""

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from backend.config.settings import Config

class SearchEngine:
    """ search functionality with improved fuzzy matching"""
    
    def __init__(self, df):
        self.df = df if df is not None else pd.DataFrame()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        
    def fuzzy_search(self, query, limit=None):
        """ search with better fuzzy matching - returns unique products"""
        if limit is None:
            limit = Config.SEARCH_LIMIT
            
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
                
                # HIGHEST PRIORITY: Exact word-by-word match (case insensitive)
                if query == field_text:
                    score += weight * 1000  # Extremely high score for exact matches
                    continue
                
                # HIGH PRIORITY: Direct substring match
                elif query in field_text:
                    score += weight * 100
                    continue
                
                # Token-based matching for partial matches
                try:
                    field_tokens = self.tokenizer.tokenize(field_text)
                    field_tokens = [self.lemmatizer.lemmatize(token) for token in field_tokens]
                except:
                    field_tokens = field_text.split()
                
                # Calculate match score for this field
                field_score = 0
                total_query_tokens = len(query_tokens)
                matched_tokens = 0
                
                for query_token in query_tokens:
                    # Exact token match
                    if query_token in field_tokens:
                        field_score += 10
                        matched_tokens += 1
                    # Partial match
                    elif any(query_token in token or token in query_token 
                           for token in field_tokens if len(token) > 2):
                        field_score += 5
                        matched_tokens += 0.5
                    # Fuzzy match for common variations
                    elif self._fuzzy_match(query_token, field_tokens):
                        field_score += 2
                        matched_tokens += 0.3
                
                # Boost score if most/all query tokens are matched
                if total_query_tokens > 0:
                    match_ratio = matched_tokens / total_query_tokens
                    if match_ratio >= 1.0:  # All tokens matched exactly
                        field_score *= 3
                    elif match_ratio >= 0.8:  # Most tokens matched
                        field_score *= 2
                    elif match_ratio >= 0.5:  # Half tokens matched
                        field_score *= 1.5
                
                score += field_score * weight
            
            # Only include products with a meaningful score (filter out weak matches)
            if score >= Config.MIN_SEARCH_SCORE:
                # Keep the highest score for each product
                if clothing_id not in product_scores or score > product_scores[clothing_id][1]:
                    product_scores[clothing_id] = (idx, score)
        
        # Sort products by score and get unique products
        if not product_scores:
            return pd.DataFrame()
        
        sorted_products = sorted(product_scores.values(), key=lambda x: x[1], reverse=True)
        result_indices = [idx for idx, score in sorted_products[:limit]]
        matched_reviews = self.df.loc[result_indices].copy()
        
        # Add search scores to the dataframe to preserve ordering
        score_map = {idx: score for idx, score in sorted_products[:limit]}
        matched_reviews['search_score'] = matched_reviews.index.map(score_map)
        
        # Return unique products with aggregated data, preserving score order
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
        """Convert reviews dataframe to unique products with aggregated data, preserving score order"""
        if df_subset.empty:
            return pd.DataFrame()
        
        # If search scores exist, preserve the highest scoring review per product for ordering
        has_scores = 'search_score' in df_subset.columns
        if has_scores:
            # For each Clothing ID, keep the review with the highest search score
            best_reviews = df_subset.loc[df_subset.groupby('Clothing ID')['search_score'].idxmax()]
            # Sort by search score to maintain ordering
            df_subset_sorted = best_reviews.sort_values('search_score', ascending=False)
        else:
            df_subset_sorted = df_subset
        
        # Group by Clothing ID and aggregate data - focus on product info only
        agg_dict = {
            'Clothes Title': 'first',        # Product name
            'Clothes Description': 'first',  # Product description
            'Class Name': 'first',           # Product category
            'Department Name': 'first',      # Department
            'Division Name': 'first',        # Division
            'Rating': 'mean',                # Average rating across all reviews
            'Recommended IND': 'mean',       # Percentage recommended
        }
        
        # Only add search_score to aggregation if it exists
        if has_scores:
            agg_dict['search_score'] = 'first'
            
        unique_products = df_subset_sorted.groupby('Clothing ID').agg(agg_dict).round({'Rating': 1, 'Recommended IND': 2})
        
        # Reset index to make Clothing ID a column again
        unique_products = unique_products.reset_index()
        
        # Add review count for display purposes
        review_counts = df_subset['Clothing ID'].value_counts()
        unique_products['Review Count'] = unique_products['Clothing ID'].map(review_counts)
        
        # Handle duplicate product names by adding unique identifiers
        unique_products = self._handle_duplicate_names(unique_products)
        
        # If we have scores, maintain the score-based ordering
        if has_scores:
            unique_products = unique_products.sort_values('search_score', ascending=False)
        
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
    
    def filter_by_category(self, division=None, department=None, class_name=None, limit=None):
        """Enhanced category filtering"""
        if limit is None:
            limit = 100
            
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
    
    def get_featured_items(self, limit=None):
        """Get featured items with highest ratings - returns 12 products from 4 specific categories for carousel"""
        if limit is None:
            limit = Config.FEATURED_ITEMS_LIMIT
            
        if self.df.empty:
            return []
            
        try:
            # Get unique products with highest average ratings
            unique_products = self._get_unique_products(self.df)
            
            # Specific 4 categories as requested
            target_categories = ['Blouses', 'Dresses', 'Intimates', 'Jackets']
            
            # Get products from these specific 4 categories only
            products_from_target_categories = unique_products[
                unique_products['Class Name'].isin(target_categories)
            ]
            
            # Get enough products from each of the 4 categories for carousel
            featured_by_category = []
            products_per_category = 12  # Enough products for carousel animation
            
            for class_name in target_categories:
                class_products = products_from_target_categories[
                    products_from_target_categories['Class Name'] == class_name
                ]
                if not class_products.empty:
                    top_in_class = class_products.nlargest(products_per_category, 'Rating')
                    featured_by_category.append(top_in_class)
            
            # Combine all products from the 4 categories
            if featured_by_category:
                combined = pd.concat(featured_by_category, ignore_index=True)
                # Remove duplicates and keep order by category
                combined = combined.drop_duplicates(subset=['Clothing ID'])
                featured = combined
            else:
                featured = unique_products.nlargest(limit, 'Rating')
            
            # Use Display Title if available, otherwise fall back to Clothes Title
            title_column = 'Display Title' if 'Display Title' in featured.columns else 'Clothes Title'
            
            return featured[['Clothing ID', title_column, 'Clothes Description', 
                           'Rating', 'Class Name', 'Department Name', 'Review Count']].rename(
                           columns={title_column: 'Clothes Title'}).to_dict('records')
        except Exception as e:
            print(f"Error getting featured items: {e}")
            return []
