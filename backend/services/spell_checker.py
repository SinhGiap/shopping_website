"""
Simple Spell Checker for Milestone I compatibility
Provides basic spell correction functionality
"""

import re
from collections import Counter

class SimpleSpellChecker:
    """Simple spell checker based on frequency and edit distance"""
    
    def __init__(self):
        self.word_freq = Counter()
        self._init_basic_vocabulary()
    
    def _init_basic_vocabulary(self):
        """Initialize with basic vocabulary for common clothing reviews"""
        basic_words = [
            'great', 'good', 'excellent', 'nice', 'perfect', 'love', 'like', 'amazing',
            'beautiful', 'comfortable', 'soft', 'quality', 'fit', 'fits', 'size',
            'color', 'colors', 'material', 'fabric', 'cotton', 'polyester',
            'dress', 'shirt', 'top', 'pants', 'jeans', 'jacket', 'sweater',
            'small', 'medium', 'large', 'tight', 'loose', 'long', 'short',
            'recommend', 'buy', 'purchase', 'return', 'keep', 'wear',
            'not', 'very', 'really', 'quite', 'pretty', 'super', 'so',
            'true', 'size', 'runs', 'little', 'bit', 'much', 'too'
        ]
        
        # Set frequency for basic words
        for word in basic_words:
            self.word_freq[word] = 100
    
    def train_from_text(self, text):
        """Train spell checker on given text corpus"""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        self.word_freq.update(words)
    
    def words(self, text):
        """Extract words from text"""
        return re.findall(r'\b[a-z]+\b', text.lower())
    
    def P(self, word):
        """Probability of a word (frequency based)"""
        return self.word_freq[word] / sum(self.word_freq.values()) if sum(self.word_freq.values()) > 0 else 0
    
    def correction(self, word):
        """Most probable spelling correction for word"""
        candidates = self.candidates(word)
        if candidates:
            return max(candidates, key=self.P)
        return word
    
    def candidates(self, word):
        """Generate possible spelling corrections for word"""
        return (self.known([word]) or 
                self.known(self.edits1(word)) or 
                self.known(self.edits2(word)) or 
                [word])
    
    def known(self, words):
        """The subset of `words` that appear in the dictionary"""
        return set(w for w in words if w in self.word_freq)
    
    def edits1(self, word):
        """All edits that are one edit away from `word`"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word):
        """All edits that are two edits away from `word`"""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    def correct_text(self, text):
        """Correct spelling in a text"""
        words = self.words(text)
        corrected_words = [self.correction(word) for word in words]
        
        # Simple reconstruction - assumes space separation
        result = text.lower()
        for original, corrected in zip(words, corrected_words):
            if original != corrected:
                result = re.sub(r'\b' + re.escape(original) + r'\b', corrected, result)
        
        return result
