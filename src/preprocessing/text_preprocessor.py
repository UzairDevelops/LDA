"""
Text Preprocessing Module.
Implements the preprocessing pipeline from Ahmed et al. (2022):
1. Tokenization
2. Normalization (lowercase, punctuation removal)
3. Stop words removal
4. Lemmatization
5. Bigram/Trigram detection

Uses SpaCy for lemmatization (preferred) and NLTK for stop words.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Union
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    import spacy
except ImportError:
    spacy = None

from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

# Add project root to path for config import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PREPROCESSING_CONFIG, CUSTOM_STOPWORDS, SPACY_MODEL

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedDocument:
    """Represents a preprocessed document."""
    doc_id: str
    original_text: str
    tokens: List[str]
    num_tokens: int
    
    def __repr__(self):
        return f"PreprocessedDocument(id='{self.doc_id}', tokens={self.num_tokens})"


class TextPreprocessor:
    """
    Text preprocessing pipeline for LDA topic modeling.
    
    Follows the methodology from Ahmed et al. (2022):
    - Tokenization: Breaking text into words
    - Normalization: Lowercase, remove punctuation/special characters
    - Stop words removal: Using NLTK + custom academic stop words
    - Lemmatization: SpaCy for accurate root form reduction
    - Bigram detection: Gensim Phrases for multi-word terms
    
    Usage:
        preprocessor = TextPreprocessor()
        documents = preprocessor.preprocess_documents(texts)
    """
    
    def __init__(
        self,
        min_token_length: int = None,
        max_token_length: int = None,
        remove_numbers: bool = None,
        remove_stopwords: bool = None,
        lemmatize: bool = None,
        use_bigrams: bool = None,
        use_trigrams: bool = None,
        custom_stopwords: Set[str] = None
    ):
        """
        Initialize preprocessor with configuration.
        
        Args:
            min_token_length: Minimum word length (default from config)
            max_token_length: Maximum word length (default from config)
            remove_numbers: Remove numeric tokens
            remove_stopwords: Remove stop words
            lemmatize: Apply lemmatization
            use_bigrams: Detect bigrams
            use_trigrams: Detect trigrams
            custom_stopwords: Additional stop words to remove
        """
        # Use config defaults if not specified
        self.min_token_length = min_token_length or PREPROCESSING_CONFIG["min_token_length"]
        self.max_token_length = max_token_length or PREPROCESSING_CONFIG["max_token_length"]
        self.remove_numbers = remove_numbers if remove_numbers is not None else PREPROCESSING_CONFIG["remove_numbers"]
        self.do_remove_stopwords = remove_stopwords if remove_stopwords is not None else PREPROCESSING_CONFIG["remove_stopwords"]
        self.do_lemmatize = lemmatize if lemmatize is not None else PREPROCESSING_CONFIG["lemmatize"]
        self.use_bigrams = use_bigrams if use_bigrams is not None else PREPROCESSING_CONFIG["use_bigrams"]
        self.use_trigrams = use_trigrams if use_trigrams is not None else PREPROCESSING_CONFIG["use_trigrams"]
        
        # Download NLTK data if needed
        self._download_nltk_data()
        
        # Initialize stop words
        self.stopwords = self._initialize_stopwords(custom_stopwords)
        
        # Initialize SpaCy for lemmatization
        self.nlp = None
        if self.do_lemmatize:
            self._initialize_spacy()
        
        # Phrase models (will be built from corpus)
        self.bigram_model = None
        self.trigram_model = None
        
        logger.info("TextPreprocessor initialized")
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
        for item in required:
            try:
                nltk.data.find(f'tokenizers/{item}' if 'punkt' in item else f'corpora/{item}')
            except LookupError:
                logger.info(f"Downloading NLTK {item}...")
                nltk.download(item, quiet=True)
    
    def _initialize_stopwords(self, custom_stopwords: Optional[Set[str]] = None) -> Set[str]:
        """Initialize stop words from NLTK + custom list."""
        stop_words = set(stopwords.words('english'))
        
        # Add custom academic stop words
        stop_words.update(CUSTOM_STOPWORDS)
        
        # Add any additional custom stop words
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        
        logger.info(f"Initialized {len(stop_words)} stop words")
        return stop_words
    
    def _initialize_spacy(self):
        """Initialize SpaCy model for lemmatization."""
        if spacy is None:
            logger.warning("SpaCy not installed. Lemmatization will use NLTK fallback.")
            return
        
        try:
            self.nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
            logger.info(f"Loaded SpaCy model: {SPACY_MODEL}")
        except OSError:
            logger.warning(f"SpaCy model '{SPACY_MODEL}' not found. Downloading...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)
            self.nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (optional)
        if self.remove_numbers:
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        else:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Tokenize using NLTK
        tokens = word_tokenize(text)
        
        return tokens
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on length and stop words.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        filtered = []
        
        for token in tokens:
            # Check length
            if len(token) < self.min_token_length:
                continue
            if len(token) > self.max_token_length:
                continue
            
            # Check stop words
            if self.do_remove_stopwords and token in self.stopwords:
                continue
            
            # Check for purely numeric (if removing numbers)
            if self.remove_numbers and token.isdigit():
                continue
            
            filtered.append(token)
        
        return filtered
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their root forms.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.do_lemmatize:
            return tokens
        
        if self.nlp is not None:
            # Use SpaCy for lemmatization
            text = " ".join(tokens)
            doc = self.nlp(text)
            return [token.lemma_ for token in doc if token.lemma_.strip()]
        else:
            # Fallback to NLTK WordNetLemmatizer
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(token) for token in tokens]
    
    def build_phrase_models(self, tokenized_docs: List[List[str]]):
        """
        Build bigram and trigram phrase models from the corpus.
        
        Args:
            tokenized_docs: List of tokenized documents
        """
        if self.use_bigrams:
            logger.info("Building bigram model...")
            bigram_phrases = Phrases(tokenized_docs, min_count=5, threshold=100)
            self.bigram_model = Phraser(bigram_phrases)
        
        if self.use_trigrams and self.bigram_model:
            logger.info("Building trigram model...")
            bigram_docs = [self.bigram_model[doc] for doc in tokenized_docs]
            trigram_phrases = Phrases(bigram_docs, min_count=3, threshold=100)
            self.trigram_model = Phraser(trigram_phrases)
    
    def apply_phrases(self, tokens: List[str]) -> List[str]:
        """
        Apply bigram/trigram phrase models to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with phrases joined (e.g., "machine_learning")
        """
        if self.bigram_model:
            tokens = self.bigram_model[tokens]
        
        if self.trigram_model:
            tokens = self.trigram_model[tokens]
        
        return list(tokens)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Apply full preprocessing pipeline to a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Step 1: Tokenize
        tokens = self.tokenize(text)
        
        # Step 2: Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Step 3: Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Step 4: Filter again after lemmatization
        tokens = self.filter_tokens(tokens)
        
        # Step 5: Apply phrase models (if built)
        tokens = self.apply_phrases(tokens)
        
        return tokens
    
    def preprocess_documents(
        self, 
        documents: List[Union[str, Dict]], 
        build_phrases: bool = True,
        show_progress: bool = True
    ) -> List[PreprocessedDocument]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of texts or dicts with 'text' and 'id' keys
            build_phrases: Build bigram/trigram models from corpus
            show_progress: Show progress bar
            
        Returns:
            List of PreprocessedDocument objects
        """
        # Normalize input format
        doc_list = []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                doc_list.append({"id": f"doc_{i}", "text": doc})
            elif isinstance(doc, dict):
                doc_list.append({
                    "id": doc.get("id", doc.get("filename", f"doc_{i}")),
                    "text": doc.get("text", "")
                })
            else:
                # Assume it has text and filename attributes (ExtractedDocument)
                doc_list.append({
                    "id": getattr(doc, "filename", f"doc_{i}"),
                    "text": getattr(doc, "text", "")
                })
        
        logger.info(f"Preprocessing {len(doc_list)} documents...")
        
        # First pass: tokenize, filter, lemmatize
        tokenized_docs = []
        iterator = tqdm(doc_list, desc="Tokenizing") if show_progress else doc_list
        
        for doc in iterator:
            tokens = self.tokenize(doc["text"])
            tokens = self.filter_tokens(tokens)
            tokens = self.lemmatize(tokens)
            tokens = self.filter_tokens(tokens)
            tokenized_docs.append(tokens)
        
        # Build phrase models
        if build_phrases and (self.use_bigrams or self.use_trigrams):
            self.build_phrase_models(tokenized_docs)
        
        # Second pass: apply phrases
        processed_docs = []
        for i, (doc, tokens) in enumerate(zip(doc_list, tokenized_docs)):
            tokens = self.apply_phrases(tokens)
            
            processed_doc = PreprocessedDocument(
                doc_id=doc["id"],
                original_text=doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
                tokens=tokens,
                num_tokens=len(tokens)
            )
            processed_docs.append(processed_doc)
        
        total_tokens = sum(doc.num_tokens for doc in processed_docs)
        logger.info(f"Preprocessing complete. Total tokens: {total_tokens}")
        
        return processed_docs
    
    def get_corpus_stats(self, processed_docs: List[PreprocessedDocument]) -> Dict:
        """
        Get statistics about the preprocessed corpus.
        
        Args:
            processed_docs: List of preprocessed documents
            
        Returns:
            Dictionary with corpus statistics
        """
        all_tokens = []
        for doc in processed_docs:
            all_tokens.extend(doc.tokens)
        
        unique_tokens = set(all_tokens)
        
        # Token frequency distribution
        from collections import Counter
        token_freq = Counter(all_tokens)
        
        return {
            "num_documents": len(processed_docs),
            "total_tokens": len(all_tokens),
            "unique_tokens": len(unique_tokens),
            "avg_tokens_per_doc": len(all_tokens) / len(processed_docs) if processed_docs else 0,
            "top_20_tokens": token_freq.most_common(20),
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    sample_texts = [
        "Machine learning and deep learning are subfields of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Topic modeling discovers abstract topics in document collections.",
    ]
    
    preprocessor = TextPreprocessor()
    processed = preprocessor.preprocess_documents(sample_texts, build_phrases=False)
    
    print("\nProcessed documents:")
    for doc in processed:
        print(f"  {doc.doc_id}: {doc.tokens[:10]}...")
    
    stats = preprocessor.get_corpus_stats(processed)
    print(f"\nCorpus stats: {stats}")
