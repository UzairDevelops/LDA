"""
LDA Topic Model Implementation.
Based on Blei, Ng, Jordan (2003) - Latent Dirichlet Allocation.
Uses Gensim for efficient LDA training and inference.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

from gensim import corpora
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
from tqdm import tqdm

# Add project root to path for config import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import LDA_CONFIG, OUTPUT_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class TopicResult:
    """Represents a single topic with its top words and probabilities."""
    topic_id: int
    words: List[str]
    probabilities: List[float]
    label: Optional[str] = None
    
    def __repr__(self):
        top_words = ", ".join(self.words[:5])
        return f"Topic {self.topic_id}: [{top_words}...]"


@dataclass
class LDAResult:
    """Contains the complete results of LDA training."""
    model: LdaModel
    dictionary: corpora.Dictionary
    corpus: List
    topics: List[TopicResult]
    num_topics: int
    coherence_score: Optional[float] = None
    perplexity: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class LDATopicModel:
    """
    LDA Topic Model for document analysis.
    
    Implements the generative probabilistic model from Blei et al. (2003).
    Uses Gensim's optimized LDA implementation with variational Bayes inference.
    
    Key concepts (from the original paper):
    - Documents are mixtures of latent topics
    - Topics are probability distributions over words
    - Uses Dirichlet priors for document-topic and topic-word distributions
    
    Usage:
        model = LDATopicModel(num_topics=10)
        result = model.train(processed_documents)
        topics = model.get_topics()
    """
    
    def __init__(
        self,
        num_topics: int = None,
        alpha: Union[str, float, List[float]] = None,
        eta: Union[str, float] = None,
        passes: int = None,
        iterations: int = None,
        chunksize: int = None,
        random_state: int = None,
        use_multicore: bool = True,
        workers: int = None
    ):
        """
        Initialize LDA model with parameters.
        
        Args:
            num_topics: Number of topics (K) - default 10 as per Ahmed et al.
            alpha: Document-topic density ('auto', 'symmetric', or numeric)
            eta: Word-topic density ('auto', 'symmetric', or numeric)
            passes: Number of passes through the corpus
            iterations: Maximum iterations for inference per document
            chunksize: Number of documents per training chunk
            random_state: Random seed for reproducibility
            use_multicore: Use parallel processing
            workers: Number of worker processes (default: CPU count - 1)
        """
        # Use config defaults
        self.num_topics = num_topics or LDA_CONFIG["num_topics"]
        self.alpha = alpha or LDA_CONFIG["alpha"]
        self.eta = eta or LDA_CONFIG["eta"]
        self.passes = passes or LDA_CONFIG["passes"]
        self.iterations = iterations or LDA_CONFIG["iterations"]
        self.chunksize = chunksize or LDA_CONFIG["chunksize"]
        self.random_state = random_state or LDA_CONFIG["random_state"]
        self.use_multicore = use_multicore
        self.workers = workers
        
        # Model components
        self.model: Optional[LdaModel] = None
        self.dictionary: Optional[corpora.Dictionary] = None
        self.corpus: Optional[List] = None
        self.texts: Optional[List[List[str]]] = None
        
        logger.info(f"LDATopicModel initialized with K={self.num_topics}")
    
    def build_dictionary_and_corpus(
        self, 
        tokenized_docs: List[List[str]],
        min_df: int = 5,
        max_df_ratio: float = 0.5
    ) -> Tuple[corpora.Dictionary, List]:
        """
        Build dictionary and bag-of-words corpus from tokenized documents.
        
        Args:
            tokenized_docs: List of tokenized documents
            min_df: Minimum document frequency for words
            max_df_ratio: Maximum document frequency ratio (0-1)
            
        Returns:
            Tuple of (dictionary, corpus)
        """
        logger.info("Building dictionary and corpus...")
        
        # Create dictionary
        dictionary = corpora.Dictionary(tokenized_docs)
        original_size = len(dictionary)
        
        # Filter extremes
        max_df = int(max_df_ratio * len(tokenized_docs))
        dictionary.filter_extremes(no_below=min_df, no_above=max_df_ratio)
        
        logger.info(f"Dictionary: {original_size} -> {len(dictionary)} tokens after filtering")
        
        # Create bag-of-words corpus
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        return dictionary, corpus
    
    def train(
        self, 
        documents: List,
        min_df: int = 5,
        max_df_ratio: float = 0.5,
        show_progress: bool = True
    ) -> LDAResult:
        """
        Train LDA model on documents.
        
        Args:
            documents: List of tokenized documents or PreprocessedDocument objects
            min_df: Minimum document frequency
            max_df_ratio: Maximum document frequency ratio
            show_progress: Show training progress
            
        Returns:
            LDAResult with trained model and topics
        """
        # Extract tokens from documents
        if hasattr(documents[0], 'tokens'):
            self.texts = [doc.tokens for doc in documents]
        else:
            self.texts = documents
        
        # Build dictionary and corpus
        self.dictionary, self.corpus = self.build_dictionary_and_corpus(
            self.texts, min_df, max_df_ratio
        )
        
        if len(self.dictionary) == 0:
            raise ValueError("Dictionary is empty. Check preprocessing or adjust min_df/max_df.")
        
        logger.info(f"Training LDA with {self.num_topics} topics...")
        
        # LdaMulticore doesn't support alpha='auto', so use LdaModel in that case
        use_multicore = self.use_multicore and self.alpha != 'auto'
        
        if use_multicore:
            workers = self.workers or max(1, (os.cpu_count() or 4) - 1)
            self.model = LdaMulticore(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                alpha=self.alpha,
                eta=self.eta,
                passes=self.passes,
                iterations=self.iterations,
                chunksize=self.chunksize,
                random_state=self.random_state,
                workers=workers,
                per_word_topics=LDA_CONFIG.get("per_word_topics", True)
            )
        else:
            self.model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                alpha=self.alpha,
                eta=self.eta,
                passes=self.passes,
                iterations=self.iterations,
                chunksize=self.chunksize,
                random_state=self.random_state,
                per_word_topics=LDA_CONFIG.get("per_word_topics", True)
            )
        
        logger.info("LDA training complete")
        
        # Extract topics
        topics = self.get_topics(num_words=10)
        
        # Calculate metrics
        coherence = self.calculate_coherence()
        perplexity = self.calculate_perplexity()
        
        result = LDAResult(
            model=self.model,
            dictionary=self.dictionary,
            corpus=self.corpus,
            topics=topics,
            num_topics=self.num_topics,
            coherence_score=coherence,
            perplexity=perplexity,
            metadata={
                "num_documents": len(self.texts),
                "vocabulary_size": len(self.dictionary),
                "alpha": str(self.alpha),
                "eta": str(self.eta),
                "passes": self.passes,
            }
        )
        
        logger.info(f"Coherence: {coherence:.4f}, Perplexity: {perplexity:.2f}")
        
        return result
    
    def get_topics(self, num_words: int = 10) -> List[TopicResult]:
        """
        Get all topics with their top words.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List of TopicResult objects
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        topics = []
        
        for topic_id in range(self.num_topics):
            word_probs = self.model.show_topic(topic_id, topn=num_words)
            words = [word for word, prob in word_probs]
            probs = [prob for word, prob in word_probs]
            
            topic = TopicResult(
                topic_id=topic_id,
                words=words,
                probabilities=probs
            )
            topics.append(topic)
        
        return topics
    
    def get_document_topics(
        self, 
        document: Union[List[str], str],
        minimum_probability: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a document.
        
        Args:
            document: Tokenized document or raw text
            minimum_probability: Minimum probability threshold
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if isinstance(document, str):
            # Tokenize if raw text
            document = document.lower().split()
        
        bow = self.dictionary.doc2bow(document)
        return self.model.get_document_topics(bow, minimum_probability=minimum_probability)
    
    def calculate_coherence(self, coherence_type: str = "c_v") -> float:
        """
        Calculate topic coherence score.
        
        Args:
            coherence_type: Type of coherence ('c_v', 'u_mass', 'c_npmi')
            
        Returns:
            Coherence score (higher is better for c_v, c_npmi)
        """
        if self.model is None or self.texts is None:
            raise ValueError("Model not trained. Call train() first.")
        
        coherence_model = CoherenceModel(
            model=self.model,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence=coherence_type
        )
        
        return coherence_model.get_coherence()
    
    def calculate_perplexity(self) -> float:
        """
        Calculate model perplexity on training corpus.
        
        Returns:
            Perplexity score (lower is better)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get log perplexity per word
        log_perplexity = self.model.log_perplexity(self.corpus)
        
        # Convert to perplexity
        return np.exp2(-log_perplexity)
    
    def save(self, filepath: Union[str, Path] = None):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model (default: models directory)
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if filepath is None:
            filepath = MODELS_DIR / f"lda_k{self.num_topics}.model"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(filepath))
        
        # Save dictionary
        dict_path = filepath.with_suffix(".dict")
        self.dictionary.save(str(dict_path))
        
        # Save corpus
        corpus_path = filepath.with_suffix(".corpus")
        corpora.MmCorpus.serialize(str(corpus_path), self.corpus)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to model file
        """
        filepath = Path(filepath)
        
        # Load model
        self.model = LdaModel.load(str(filepath))
        self.num_topics = self.model.num_topics
        
        # Load dictionary
        dict_path = filepath.with_suffix(".dict")
        if dict_path.exists():
            self.dictionary = corpora.Dictionary.load(str(dict_path))
        
        # Load corpus
        corpus_path = filepath.with_suffix(".corpus")
        if corpus_path.exists():
            self.corpus = corpora.MmCorpus(str(corpus_path))
        
        logger.info(f"Model loaded from {filepath}")
    
    def print_topics(self, num_words: int = 10):
        """Print all topics with their top words."""
        topics = self.get_topics(num_words)
        
        print(f"\n{'='*60}")
        print(f"LDA Topics (K={self.num_topics})")
        print(f"{'='*60}\n")
        
        for topic in topics:
            words_str = ", ".join(topic.words)
            print(f"Topic {topic.topic_id}: {words_str}")
        
        print(f"\n{'='*60}")


# Import os for cpu_count
import os


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample tokenized documents
    sample_docs = [
        ["machine", "learning", "algorithm", "data", "model"],
        ["deep", "learning", "neural", "network", "training"],
        ["natural", "language", "processing", "text", "nlp"],
        ["topic", "model", "lda", "document", "word"],
        ["computer", "vision", "image", "recognition", "cnn"],
    ]
    
    # Train model
    lda = LDATopicModel(num_topics=3, passes=5, use_multicore=False)
    result = lda.train(sample_docs, min_df=1)
    
    # Print results
    lda.print_topics()
    print(f"\nCoherence: {result.coherence_score:.4f}")
    print(f"Perplexity: {result.perplexity:.2f}")
