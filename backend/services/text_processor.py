import re
from nltk import RegexpTokenizer
from itertools import chain
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from spellchecker import SpellChecker

class TextProcessor:
    def __init__(self, stopwords_path="resources/stopwords_en.txt"):
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()
        
        # Spell checking optimization
        self.spell_cache = {}
        self.spell_check_enabled = True
        self.min_word_length_for_spell_check = 4
        
        with open(stopwords_path, "r") as s:
            self.stopwords = set(s.read().splitlines())
        
        # Keep negators
        self.negators = {"no", "not", "never"}
        self.stopwords = self.stopwords.difference(self.negators)

    def preprocess_text(self, texts):
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Step 1: Negation handling
        texts = [self._handle_negations(str(t)) for t in texts if str(t).strip()]

        # Step 2: Tokenization
        tokenized = [self.tokenizer.tokenize(t) for t in texts]

        # Step 3: Length filtering
        tokenized = [
            [w for w in review if len(w) >= 2 or w in self.negators]
            for review in tokenized
        ]

        # Step 4: Stopword removal
        tokenized = [[w for w in review if w not in self.stopwords] for review in tokenized]

        if single_input:
            # For single text, skip corpus-wide steps
            tokens = tokenized[0] if tokenized else []
            tokens = self._optimized_spell_check(tokens)
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
            return tokens

        # Corpus-wide steps
        return self._process_corpus(tokenized)

    def _process_corpus(self, tokenized_reviews):
        # Remove hapax words
        all_tokens = list(chain.from_iterable(tokenized_reviews))
        tf = Counter(all_tokens)
        hapax = {w for w, c in tf.items() if c == 1}
        reviews_no_hapax = [[w for w in review if w not in hapax] for review in tokenized_reviews]

        # Spell correction
        all_tokens = list(chain.from_iterable(reviews_no_hapax))
        corrected_tokens = self._optimized_spell_check(all_tokens)

        # Restore structure
        reviews_corrected = []
        idx = 0
        for review in reviews_no_hapax:
            length = len(review)
            reviews_corrected.append(corrected_tokens[idx:idx+length])
            idx += length

        # Lemmatization
        reviews_lemmatized = [[self.lemmatizer.lemmatize(w) for w in review] for review in reviews_corrected]

        # Remove top 20 most frequent words by document frequency
        df_counter = Counter()
        for review in reviews_lemmatized:
            df_counter.update(set(review))
        top20_df_words = {w for w, _ in df_counter.most_common(20)}
        reviews_df_filtered = [[w for w in review if w not in top20_df_words] for review in reviews_lemmatized]

        # Collocation extraction (merge frequent bigrams)
        all_tokens = list(chain.from_iterable(reviews_df_filtered))
        bigram_finder = BigramCollocationFinder.from_words(all_tokens)
        bigram_finder.apply_freq_filter(3)
        frequent_bigrams = set(bigram_finder.ngram_fd.keys())

        def merge_bigrams(review):
            merged = []
            i = 0
            while i < len(review):
                if i < len(review) - 1 and (review[i], review[i+1]) in frequent_bigrams:
                    merged.append(f"{review[i]}_{review[i+1]}")
                    i += 2
                else:
                    merged.append(review[i])
                    i += 1
            return merged

        return [merge_bigrams(review) for review in reviews_df_filtered]

    def _handle_negations(self, text):
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
        for contraction, expansion in negation_patterns.items():
            text = text.replace(contraction, expansion)
            text = text.replace(contraction.title(), expansion.title())
        return text

    def _optimized_spell_check(self, tokens):
        if not self.spell_check_enabled:
            return tokens

        corrected = []
        for token in tokens:
            if len(token) < self.min_word_length_for_spell_check:
                corrected.append(token)
                continue
            if token in self.spell_cache:
                corrected.append(self.spell_cache[token])
                continue
            if token in self.spell:
                self.spell_cache[token] = token
                corrected.append(token)
            else:
                correction = self.spell.correction(token)
                if correction and correction != token and len(correction) > 2:
                    self.spell_cache[token] = correction
                    corrected.append(correction)
                else:
                    self.spell_cache[token] = token
                    corrected.append(token)
        return corrected

    def enable_spell_checking(self, enabled=True):
        self.spell_check_enabled = enabled
        print("[INFO] Spell checking " + ("enabled" if enabled else "disabled"))
