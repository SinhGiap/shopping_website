"""
Text Processing Service - Exact Milestone I Implementation
Advanced text preprocessing matching task1.py from Milestone I
"""

import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from collections import defaultdict, Counter

from backend.services.spell_checker import SimpleSpellChecker
from backend.config.settings import Config

class TextProcessor:
    """Enhanced text processor exactly matching Milestone I task1.py implementation"""
    
    def __init__(self):
        self._ensure_nltk_data()
        
        # Initialize tokenizer exactly as in Milestone I
        self.tokenizer = RegexpTokenizer(r'\b\w+\b')
        
        # Initialize lemmatizer exactly as in Milestone I
        self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords exactly as in Milestone I
        self.stop_words = set(stopwords.words('english'))
        
        # Custom stopwords for clothing domain (Milestone I approach)
        custom_stops = {
            'item', 'product', 'clothing', 'wear', 'wearing', 'worn',
            'buy', 'bought', 'purchase', 'purchased', 'order', 'ordered'
        }
        self.stop_words.update(custom_stops)
        
        # Initialize spell checker exactly as in Milestone I
        if Config.SPELL_CHECK_ENABLED:
            self.spell_checker = SimpleSpellChecker()
        else:
            self.spell_checker = None
        
        # Contraction mapping exactly as in Milestone I
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "what's": "what is",
            "where's": "where is",
            "how's": "how is",
            "who's": "who is",
            "why's": "why is"
        }
        
        # Negation handling exactly as in Milestone I
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'nobody',
            'neither', 'nor', 'without', 'lack', 'lacking', 'lacks',
            'barely', 'hardly', 'scarcely', 'seldom', 'rarely'
        }
        
        # Initialize statistics (Milestone I approach)
        self.stats = defaultdict(int)
        
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded exactly as in Milestone I"""
        required_downloads = [
            'punkt',
            'stopwords', 
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        
        for item in required_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                try:
                    nltk.download(item, quiet=True)
                except Exception:
                    pass
    
    def preprocess_text(self, text, extract_collocations=False):
        """
        Preprocess text exactly as in Milestone I task1.py
        
        Args:
            text (str): Input text to preprocess
            extract_collocations (bool): Whether to extract collocations
            
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Convert to lowercase exactly as in Milestone I
        text = text.lower()
        self.stats['original_length'] += len(text)
        
        # Step 2: Expand contractions exactly as in Milestone I
        text = self._expand_contractions(text)
        
        # Step 3: Handle negations exactly as in Milestone I
        text = self._handle_negations(text)
        
        # Step 4: Tokenize exactly as in Milestone I
        tokens = self.tokenizer.tokenize(text)
        self.stats['tokens_before_filter'] += len(tokens)
        
        # Step 5: Spell correction exactly as in Milestone I
        if self.spell_checker and Config.SPELL_CHECK_ENABLED:
            tokens = [self.spell_checker.correction(token) for token in tokens]
            self.stats['spell_corrections'] += 1
        
        # Step 6: Filter tokens exactly as in Milestone I
        filtered_tokens = []
        for token in tokens:
            # Remove non-alphabetic tokens exactly as in Milestone I
            if not token.isalpha():
                continue
            
            # Remove short tokens exactly as in Milestone I
            if len(token) < 2:
                continue
            
            # Remove stopwords exactly as in Milestone I
            if token in self.stop_words:
                continue
            
            # Remove negation words from final output (after handling)
            if token.startswith('not_') or token in self.negation_words:
                filtered_tokens.append(token)
            else:
                filtered_tokens.append(token)
        
        self.stats['tokens_after_filter'] += len(filtered_tokens)
        
        # Step 7: Lemmatization exactly as in Milestone I
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Step 8: Extract collocations if requested (Milestone I approach)
        if extract_collocations and len(lemmatized_tokens) > 1:
            collocations = self._extract_collocations(lemmatized_tokens)
            lemmatized_tokens.extend(collocations)
        
        # Join tokens back to string exactly as in Milestone I
        result = ' '.join(lemmatized_tokens)
        self.stats['final_length'] += len(result)
        
        return result
    
    def _expand_contractions(self, text):
        """Expand contractions exactly as in Milestone I"""
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text
    
    def _handle_negations(self, text):
        """
        Handle negations exactly as in Milestone I
        Preserve negation context by prefixing following words
        """
        words = text.split()
        processed_words = []
        negate_next = False
        
        for i, word in enumerate(words):
            # Check if current word is a negation
            if any(neg in word.lower() for neg in self.negation_words):
                processed_words.append(word)
                negate_next = True
                continue
            
            # If we should negate this word
            if negate_next:
                # Only negate content words (not function words)
                if (word.lower() not in self.stop_words and 
                    len(word) > 2 and 
                    word.isalpha()):
                    processed_words.append(f"not_{word}")
                    negate_next = False  # Reset after first content word
                else:
                    processed_words.append(word)
            else:
                processed_words.append(word)
            
            # Reset negation after punctuation exactly as in Milestone I
            if word.endswith(('.', '!', '?', ';')):
                negate_next = False
        
        return ' '.join(processed_words)
    
    def _extract_collocations(self, tokens):
        """
        Extract meaningful collocations exactly as in Milestone I
        
        Args:
            tokens (list): List of preprocessed tokens
            
        Returns:
            list: List of significant collocations
        """
        if len(tokens) < 2:
            return []
        
        try:
            # Create bigram finder exactly as in Milestone I
            bigram_finder = BigramCollocationFinder.from_words(tokens)
            
            # Apply frequency filter exactly as in Milestone I
            bigram_finder.apply_freq_filter(Config.MIN_COLLOCATION_FREQ)
            
            # Score collocations using PMI exactly as in Milestone I
            scored_collocations = bigram_finder.score_ngrams(BigramAssocMeasures.pmi)
            
            # Filter by threshold exactly as in Milestone I
            significant_collocations = [
                f"{word1}_{word2}" 
                for (word1, word2), score in scored_collocations
                if score >= Config.COLLOCATION_THRESHOLD
            ]
            
            self.stats['collocations_found'] += len(significant_collocations)
            return significant_collocations[:10]  # Limit to top 10 exactly as in Milestone I
            
        except Exception as e:
            print(f"Error extracting collocations: {e}")
            return []
    
    def get_processing_stats(self):
        """Get processing statistics exactly as in Milestone I"""
        return dict(self.stats)
    
    def reset_stats(self):
        """Reset processing statistics exactly as in Milestone I"""
        self.stats.clear()
    
    def train_spell_checker(self, text_corpus):
        """Train spell checker on text corpus exactly as in Milestone I"""
        if self.spell_checker:
            self.spell_checker.train_from_text(text_corpus)
    
    def batch_preprocess(self, texts, extract_collocations=False):
        """
        Batch preprocess multiple texts exactly as in Milestone I
        
        Args:
            texts (list): List of texts to preprocess
            extract_collocations (bool): Whether to extract collocations
            
        Returns:
            list: List of preprocessed texts
        """
        if not texts:
            return []
        
        # Train spell checker on the corpus first exactly as in Milestone I
        if self.spell_checker and Config.SPELL_CHECK_ENABLED:
            corpus = ' '.join(str(text) for text in texts if text)
            self.train_spell_checker(corpus)
        
        # Process each text exactly as in Milestone I
        processed_texts = []
        for text in texts:
            processed = self.preprocess_text(text, extract_collocations)
            processed_texts.append(processed)
        
        return processed_texts
