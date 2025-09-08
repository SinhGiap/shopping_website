"""
Text Processing Service
Enhanced text preprocessing pipeline from Milestone I Task 1
"""

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from backend.config.settings import Config

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

class TextProcessor:
    """Enhanced text preprocessing pipeline"""
    
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords with fallback (but preserve important negation words)
        self.stopwords = self._load_stopwords()
        
        # Remove negation words from stopwords to preserve sentiment context
        negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 
                         'neither', 'nor', 'dont', "don't", 'doesnt', "doesn't", 
                         'didnt', "didn't", 'wont', "won't", 'wouldnt', "wouldn't",
                         'cant', "can't", 'couldnt', "couldn't", 'shouldnt', "shouldn't"}
        self.stopwords = self.stopwords - negation_words
    
    def _load_stopwords(self):
        """Load stopwords with multiple fallback options"""
        # Try custom stopwords file
        for path in Config.STOPWORDS_PATHS:
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
        """Apply enhanced preprocessing pipeline with negation handling"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Handle common contractions and negations first
            text = self._handle_negations(text)
            
            # Tokenization
            tokens = self.tokenizer.tokenize(text)
            
            # Lowercase
            tokens = [token.lower() for token in tokens]
            
            # Remove short words (but keep important negations)
            tokens = [token for token in tokens if len(token) >= 2 or token in ['no']]
            
            # Remove stopwords (excluding negation words)
            tokens = [token for token in tokens if token not in self.stopwords]
            
            # Lemmatization
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return " ".join(tokens)
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return text.lower()
    
    def _handle_negations(self, text):
        """Handle negations and contractions to preserve sentiment"""
        # Common negation patterns
        negation_patterns = {
            "don't": "do not",
            "doesn't": "does not", 
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "can not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not"
        }
        
        # Replace contractions
        for contraction, expansion in negation_patterns.items():
            text = text.replace(contraction, expansion)
            text = text.replace(contraction.title(), expansion.title())
        
        return text
