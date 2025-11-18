"""
Text Preprocessing Module for MovieMind
Handles text cleaning, tokenization, and feature extraction for NLP
"""

import re
import string
import nltk
from typing import List, Dict, Any
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


class TextProcessor:
    """Handles all text preprocessing operations"""

    def __init__(self, language: str = 'english'):
        """
        Initialize text processor

        Args:
            language: Language for stopwords (default: english)
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def remove_html(self, text: str) -> str:
        """
        Remove HTML tags from text

        Args:
            text: Input text with potential HTML

        Returns:
            Clean text without HTML
        """
        if not text:
            return ""

        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ')

    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text

        Args:
            text: Input text

        Returns:
            Text without URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    def remove_special_characters(self, text: str, keep_punctuation: bool = False) -> str:
        """
        Remove special characters from text

        Args:
            text: Input text
            keep_punctuation: Whether to keep basic punctuation

        Returns:
            Cleaned text
        """
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?\'\-]'
        else:
            # Keep only letters, numbers, and spaces
            pattern = r'[^a-zA-Z0-9\s]'

        return re.sub(pattern, ' ', text)

    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower() if text else ""

    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize"""
        return ' '.join(text.split())

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return word_tokenize(text)

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens

        Args:
            tokens: List of tokens

        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens

        Args:
            tokens: List of tokens

        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]

    def clean_text(self,
                   text: str,
                   remove_html_tags: bool = True,
                   remove_urls_flag: bool = True,
                   lowercase: bool = True,
                   remove_special_chars: bool = True,
                   keep_punctuation: bool = False) -> str:
        """
        Complete text cleaning pipeline

        Args:
            text: Input text
            remove_html_tags: Remove HTML tags
            remove_urls_flag: Remove URLs
            lowercase: Convert to lowercase
            remove_special_chars: Remove special characters
            keep_punctuation: Keep basic punctuation

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # HTML removal
        if remove_html_tags:
            text = self.remove_html(text)

        # URL removal
        if remove_urls_flag:
            text = self.remove_urls(text)

        # Lowercase
        if lowercase:
            text = self.to_lowercase(text)

        # Special characters
        if remove_special_chars:
            text = self.remove_special_characters(text, keep_punctuation)

        # Whitespace
        text = self.remove_extra_whitespace(text)

        return text

    def preprocess_text(self,
                       text: str,
                       clean: bool = True,
                       tokenize_text: bool = True,
                       remove_stops: bool = True,
                       lemmatize_tokens: bool = True,
                       stem_tokens: bool = False) -> Any:
        """
        Full preprocessing pipeline

        Args:
            text: Input text
            clean: Apply cleaning
            tokenize_text: Tokenize text
            remove_stops: Remove stopwords
            lemmatize_tokens: Lemmatize tokens
            stem_tokens: Stem tokens (note: don't use with lemmatize)

        Returns:
            Processed text (string if not tokenized, list if tokenized)
        """
        # Clean text
        if clean:
            text = self.clean_text(text)

        # Tokenize
        if tokenize_text:
            tokens = self.tokenize(text)

            # Remove stopwords
            if remove_stops:
                tokens = self.remove_stopwords(tokens)

            # Lemmatize or stem
            if lemmatize_tokens and not stem_tokens:
                tokens = self.lemmatize(tokens)
            elif stem_tokens:
                tokens = self.stem(tokens)

            return tokens
        else:
            return text

    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract various features from text

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        # Clean text first
        clean_text = self.clean_text(text, keep_punctuation=True)
        tokens = self.tokenize(clean_text)

        features = {
            'text_length': len(text),
            'clean_text_length': len(clean_text),
            'word_count': len(tokens),
            'unique_word_count': len(set(tokens)),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'sentence_count': len(nltk.sent_tokenize(text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'punctuation_count': sum(1 for c in text if c in string.punctuation),
            'stopword_count': sum(1 for word in tokens if word.lower() in self.stop_words),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?')
        }

        return features


def preprocess_review_batch(reviews: List[str], processor: TextProcessor = None) -> List[str]:
    """
    Preprocess a batch of reviews

    Args:
        reviews: List of review texts
        processor: TextProcessor instance (creates new if None)

    Returns:
        List of preprocessed reviews
    """
    if processor is None:
        processor = TextProcessor()

    processed = []
    for review in reviews:
        cleaned = processor.clean_text(review)
        processed.append(cleaned)

    return processed


if __name__ == "__main__":
    # Test the processor
    processor = TextProcessor()

    # Test text
    sample_text = """
    <p>This is an AMAZING movie! I loved it so much!!!
    Check out more at https://example.com/review</p>
    The acting was superb, but the pacing could've been better.
    """

    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")

    # Clean text
    cleaned = processor.clean_text(sample_text)
    print("Cleaned text:")
    print(cleaned)
    print("\n" + "="*50 + "\n")

    # Preprocess with tokenization
    tokens = processor.preprocess_text(sample_text, lemmatize_tokens=True)
    print("Processed tokens:")
    print(tokens)
    print("\n" + "="*50 + "\n")

    # Extract features
    features = processor.extract_features(sample_text)
    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
