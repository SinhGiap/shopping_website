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
        with open(stopwords_path, "r") as s:
            self.stopwords = set(s.read().splitlines())
        
        # Ensure negators are *not* removed
        self.negators = {"no", "not", "never"}
        self.stopwords = self.stopwords.difference(self.negators)

    def preprocess_text(self, texts):
        """
        Apply full preprocessing pipeline. If input is a list, apply corpus-wide steps (hapax, top-20 DF, collocations).
        If input is a single string, skip corpus-wide steps.
        Returns list of tokenized documents (for list input) or list of tokens (for string input).
        """
        single_input = False
        if isinstance(texts, str):
            print(f"[DEBUG] preprocess_text called with input: {texts}")
            if texts is None:
                print("[ERROR] Input text is None!")
                return ""
            single_input = True
            texts = [texts]

        # 1. Negation handling
        texts = [self._handle_negations(str(t)) for t in texts if str(t).strip()]
        # 2. Tokenization
        tokenized = [self.tokenizer.tokenize(t) for t in texts]
        for idx, review in enumerate(tokenized):
            if review is None:
                print(f"[ERROR] tokenized[{idx}] is None before len() check!")
        # 3. Length filtering
        length_filtered = [[w for w in review if (review is not None and len(w) >= 2) or w in self.negators] for review in tokenized]
        # 4. Stopword removal
        stopword_filtered = [[w for w in review if w not in self.stopwords] for review in length_filtered]

        if single_input:
            all_tokens = stopword_filtered[0] if stopword_filtered else []
            print(f"[DEBUG] Tokens after stopword removal: {all_tokens}")
            misspelled = self.spell.unknown(all_tokens)
            correction_dict = {w: self.spell.correction(w) for w in misspelled if self.spell.correction(w)}
            corrected_tokens = [correction_dict.get(token, token) for token in all_tokens]
            print(f"[DEBUG] Tokens after spell correction: {corrected_tokens}")
            lemmatized = [self.lemmatizer.lemmatize(w) for w in corrected_tokens]
            print(f"[DEBUG] Tokens after lemmatization: {lemmatized}")
            return lemmatized
            # Corpus-wide steps
            # 5. Remove hapax (words that appear only once across corpus)
            all_tokens = list(chain.from_iterable(stopword_filtered))
            tf = Counter(all_tokens)
            hapax = {w for w, c in tf.items() if c == 1}
            no_hapax_reviews = [[w for w in review if w not in hapax] for review in stopword_filtered]

            # 6. Spell checking and correction
            all_tokens = [token for review in no_hapax_reviews for token in review]
            misspelled = self.spell.unknown(all_tokens)
            correction_dict = {w: self.spell.correction(w) for w in misspelled if self.spell.correction(w)}
            corrected_tokens = [correction_dict.get(token, token) for token in all_tokens]

            # Restore structure
            corrected_reviews = []
            idx = 0
            for idx2, review in enumerate(no_hapax_reviews):
                if review is None:
                    print(f"[ERROR] no_hapax_reviews[{idx2}] is None before len() check!")
                length = len(review) if review is not None else 0
                corrected_reviews.append(corrected_tokens[idx:idx+length])
                idx += length

            # 7. Lemmatization
            lemmatized_reviews = [[self.lemmatizer.lemmatize(w) for w in review] for review in corrected_reviews]

            # 8. Remove top 20 most frequent words (by document frequency)
            df_counter = Counter()
            for review in lemmatized_reviews:
                df_counter.update(set(review))
            top20_df_words = {w for w, _ in df_counter.most_common(20)}
            df_filtered_reviews = [[w for w in review if w not in top20_df_words] for review in lemmatized_reviews]

            # 9. Collocation extraction (merge frequent bigrams)
            all_tokens = list(chain.from_iterable(df_filtered_reviews))
            bigram_finder = BigramCollocationFinder.from_words(all_tokens)
            bigram_finder.apply_freq_filter(3)
            frequent_bigrams = set(bigram_finder.ngram_fd.keys())

            def merge_bigrams(review, bigrams_set):
                merged = []
                i = 0
                while i < len(review):
                    if i < len(review) - 1 and (review[i], review[i + 1]) in bigrams_set:
                        merged.append(f"{review[i]}_{review[i + 1]}")
                        i += 2
                    else:
                        merged.append(review[i])
                        i += 1
                return merged

            collocation_reviews = [merge_bigrams(review, frequent_bigrams) for review in df_filtered_reviews]
            return collocation_reviews
            all_tokens = list(chain.from_iterable(stopword_filtered))
            tf = Counter(all_tokens)
            hapax = {w for w, c in tf.items() if c == 1}
            no_hapax_reviews = [[w for w in review if w not in hapax] 
                                for review in stopword_filtered]

            # 7. Spell checking and correction
            all_tokens = [token for review in no_hapax_reviews for token in review]
            misspelled = self.spell.unknown(all_tokens)
            correction_dict = {w: self.spell.correction(w) for w in misspelled if self.spell.correction(w)}
            corrected_tokens = [correction_dict.get(token, token) for token in all_tokens]

            # Restore structure
            corrected_reviews = []
            idx = 0
            for review in no_hapax_reviews:
                length = len(review)
                corrected_reviews.append(corrected_tokens[idx:idx+length])
                idx += length

            # 8. Lemmatization
            lemmatized_reviews = [[self.lemmatizer.lemmatize(w) for w in review] 
                                  for review in corrected_reviews]

            # 9. Remove top 20 most frequent words (by document frequency)
            df_counter = Counter()
            for review in lemmatized_reviews:
                df_counter.update(set(review))
            top20_df_words = {w for w, _ in df_counter.most_common(20)}
            df_filtered_reviews = [[w for w in review if w not in top20_df_words] 
                                   for review in lemmatized_reviews]

            # 10. Collocation extraction (merge frequent bigrams)
            all_tokens = list(chain.from_iterable(df_filtered_reviews))
            bigram_finder = BigramCollocationFinder.from_words(all_tokens)
            bigram_finder.apply_freq_filter(3)
            frequent_bigrams = set(bigram_finder.ngram_fd.keys())

            def merge_bigrams(review, bigrams_set):
                merged = []
                i = 0
                while i < len(review):
                    if i < len(review) - 1 and (review[i], review[i + 1]) in bigrams_set:
                        merged.append(f"{review[i]}_{review[i + 1]}")
                        i += 2
                    else:
                        merged.append(review[i])
                        i += 1
                return merged

            collocation_reviews = [merge_bigrams(review, frequent_bigrams) 
                                   for review in df_filtered_reviews]
            return collocation_reviews
        else:
            # Single review: skip corpus-wide steps
            # 7. Spell checking and correction
            all_tokens = [token for review in stopword_filtered for token in review]
            misspelled = self.spell.unknown(all_tokens)
            correction_dict = {w: self.spell.correction(w) for w in misspelled if self.spell.correction(w)}
            corrected_tokens = [correction_dict.get(token, token) for token in all_tokens]
            # Lemmatization
            lemmatized = [self.lemmatizer.lemmatize(w) for w in corrected_tokens]
            return lemmatized

    def _handle_negations(self, text):
        """Expand common negations and contractions"""
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