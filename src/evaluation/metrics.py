"""
Topic Model Evaluation Metrics.
Implements coherence, perplexity, and topic diversity metrics.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation results for a topic model."""
    coherence_cv: float
    coherence_umass: float
    perplexity: float
    topic_diversity: float
    avg_topic_entropy: float
    num_topics: int
    vocabulary_size: int
    topic_sizes: List[int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "coherence_cv": self.coherence_cv,
            "coherence_umass": self.coherence_umass,
            "perplexity": self.perplexity,
            "topic_diversity": self.topic_diversity,
            "avg_topic_entropy": self.avg_topic_entropy,
            "num_topics": self.num_topics,
            "vocabulary_size": self.vocabulary_size,
            "topic_sizes": self.topic_sizes,
        }


class TopicEvaluator:
    """
    Comprehensive evaluation for LDA topic models.
    
    Implements multiple evaluation metrics:
    - Coherence Score (C_v): Measures semantic interpretability using word embeddings
    - Coherence Score (UMass): Measures using document co-occurrence statistics
    - Perplexity: Measures predictive ability on held-out data
    - Topic Diversity: Measures uniqueness of topics
    - Topic Entropy: Measures word distribution spread within topics
    
    Based on best practices from:
    - Röder et al. (2015): Coherence measures
    - Blei et al. (2003): Perplexity
    
    Usage:
        evaluator = TopicEvaluator()
        result = evaluator.evaluate(model, texts, dictionary, corpus)
    """
    
    def __init__(self, top_n_words: int = 10):
        """
        Initialize evaluator.
        
        Args:
            top_n_words: Number of top words per topic for diversity/coherence
        """
        self.top_n_words = top_n_words
        logger.info("TopicEvaluator initialized")
    
    def calculate_coherence_cv(
        self,
        model: LdaModel,
        texts: List[List[str]],
        dictionary: corpora.Dictionary
    ) -> float:
        """
        Calculate C_v coherence score.
        
        C_v uses sliding window, word co-occurrence, and NPMI.
        Generally considered the best coherence measure for interpretability.
        
        Args:
            model: Trained LDA model
            texts: Original tokenized documents
            dictionary: Gensim dictionary
            
        Returns:
            C_v coherence score (higher is better)
        """
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v',
            topn=self.top_n_words
        )
        return coherence_model.get_coherence()
    
    def calculate_coherence_umass(
        self,
        model: LdaModel,
        corpus: List,
        dictionary: corpora.Dictionary
    ) -> float:
        """
        Calculate UMass coherence score.
        
        UMass uses document co-occurrence and conditional log-probability.
        Faster than C_v but less correlated with human judgment.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            dictionary: Gensim dictionary
            
        Returns:
            UMass coherence score (less negative is better)
        """
        coherence_model = CoherenceModel(
            model=model,
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass',
            topn=self.top_n_words
        )
        return coherence_model.get_coherence()
    
    def calculate_perplexity(
        self,
        model: LdaModel,
        corpus: List
    ) -> float:
        """
        Calculate perplexity on a corpus.
        
        Perplexity measures how well the model predicts the corpus.
        Lower values indicate better generalization.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            
        Returns:
            Perplexity score (lower is better)
        """
        log_perplexity = model.log_perplexity(corpus)
        return np.exp2(-log_perplexity)
    
    def calculate_topic_diversity(self, model: LdaModel) -> float:
        """
        Calculate topic diversity.
        
        Measures the proportion of unique words across all topics.
        Higher diversity indicates more distinct topics.
        
        Args:
            model: Trained LDA model
            
        Returns:
            Topic diversity score (0-1, higher is better)
        """
        all_words = []
        
        for topic_id in range(model.num_topics):
            topic_words = [word for word, _ in model.show_topic(topic_id, topn=self.top_n_words)]
            all_words.extend(topic_words)
        
        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words) if all_words else 0
        
        return diversity
    
    def calculate_topic_entropy(self, model: LdaModel) -> List[float]:
        """
        Calculate entropy for each topic's word distribution.
        
        Higher entropy indicates more even word distribution.
        Very low entropy might indicate topic collapse.
        
        Args:
            model: Trained LDA model
            
        Returns:
            List of entropy values per topic
        """
        entropies = []
        
        for topic_id in range(model.num_topics):
            topic_dist = model.get_topic_terms(topic_id, topn=1000)
            probs = np.array([prob for _, prob in topic_dist])
            
            # Normalize
            probs = probs / probs.sum()
            
            # Calculate entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            entropies.append(entropy)
        
        return entropies
    
    def calculate_topic_sizes(
        self,
        model: LdaModel,
        corpus: List,
        threshold: float = 0.1
    ) -> List[int]:
        """
        Calculate the number of documents assigned to each topic.
        
        A document is assigned to a topic if its probability exceeds threshold.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            threshold: Minimum probability for topic assignment
            
        Returns:
            List of document counts per topic
        """
        topic_counts = [0] * model.num_topics
        
        for doc_bow in corpus:
            doc_topics = model.get_document_topics(doc_bow, minimum_probability=threshold)
            for topic_id, _ in doc_topics:
                topic_counts[topic_id] += 1
        
        return topic_counts
    
    def evaluate(
        self,
        model: LdaModel,
        texts: List[List[str]],
        dictionary: corpora.Dictionary,
        corpus: List
    ) -> EvaluationResult:
        """
        Run complete evaluation on a topic model.
        
        Args:
            model: Trained LDA model
            texts: Original tokenized documents
            dictionary: Gensim dictionary
            corpus: Bag-of-words corpus
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info("Running comprehensive evaluation...")
        
        # Calculate coherence scores
        coherence_cv = self.calculate_coherence_cv(model, texts, dictionary)
        logger.debug(f"C_v coherence: {coherence_cv:.4f}")
        
        coherence_umass = self.calculate_coherence_umass(model, corpus, dictionary)
        logger.debug(f"UMass coherence: {coherence_umass:.4f}")
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(model, corpus)
        logger.debug(f"Perplexity: {perplexity:.2f}")
        
        # Calculate topic diversity
        diversity = self.calculate_topic_diversity(model)
        logger.debug(f"Topic diversity: {diversity:.4f}")
        
        # Calculate topic entropy
        entropies = self.calculate_topic_entropy(model)
        avg_entropy = np.mean(entropies)
        logger.debug(f"Avg topic entropy: {avg_entropy:.4f}")
        
        # Calculate topic sizes
        topic_sizes = self.calculate_topic_sizes(model, corpus)
        
        result = EvaluationResult(
            coherence_cv=coherence_cv,
            coherence_umass=coherence_umass,
            perplexity=perplexity,
            topic_diversity=diversity,
            avg_topic_entropy=avg_entropy,
            num_topics=model.num_topics,
            vocabulary_size=len(dictionary),
            topic_sizes=topic_sizes
        )
        
        logger.info(f"Evaluation complete: C_v={coherence_cv:.4f}, Diversity={diversity:.4f}")
        
        return result
    
    def generate_report(self, result: EvaluationResult) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            result: EvaluationResult from evaluate()
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("TOPIC MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nModel Configuration:")
        report.append(f"  Number of Topics: {result.num_topics}")
        report.append(f"  Vocabulary Size: {result.vocabulary_size}")
        
        report.append(f"\nCoherence Metrics:")
        report.append(f"  C_v Coherence:    {result.coherence_cv:.4f}  (higher is better)")
        report.append(f"  UMass Coherence:  {result.coherence_umass:.4f}  (less negative is better)")
        
        report.append(f"\nQuality Metrics:")
        report.append(f"  Perplexity:       {result.perplexity:.2f}  (lower is better)")
        report.append(f"  Topic Diversity:  {result.topic_diversity:.4f}  (higher is better)")
        report.append(f"  Avg Entropy:      {result.avg_topic_entropy:.4f}")
        
        report.append(f"\nTopic Document Distribution:")
        for i, count in enumerate(result.topic_sizes):
            bar = "█" * (count // 5) if count > 0 else "▏"
            report.append(f"  Topic {i:2d}: {count:4d} docs {bar}")
        
        report.append("\n" + "=" * 60)
        
        # Quality assessment
        report.append("\nQuality Assessment:")
        
        if result.coherence_cv > 0.5:
            report.append("  ✓ Excellent coherence (C_v > 0.5)")
        elif result.coherence_cv > 0.4:
            report.append("  ✓ Good coherence (C_v > 0.4)")
        else:
            report.append("  ! Low coherence - consider adjusting K or preprocessing")
        
        if result.topic_diversity > 0.8:
            report.append("  ✓ High topic diversity - topics are distinct")
        elif result.topic_diversity > 0.6:
            report.append("  ✓ Moderate topic diversity")
        else:
            report.append("  ! Low diversity - topics may overlap significantly")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def compare_models(
        self,
        models: List[LdaModel],
        model_names: List[str],
        texts: List[List[str]],
        dictionary: corpora.Dictionary,
        corpus: List
    ) -> Dict:
        """
        Compare multiple models on evaluation metrics.
        
        Args:
            models: List of trained LDA models
            model_names: Names for each model
            texts: Original tokenized documents
            dictionary: Gensim dictionary
            corpus: Bag-of-words corpus
            
        Returns:
            Comparison dictionary
        """
        comparison = {}
        
        for model, name in zip(models, model_names):
            result = self.evaluate(model, texts, dictionary, corpus)
            comparison[name] = result.to_dict()
        
        return comparison


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from gensim.models import LdaModel
    from gensim import corpora
    
    # Sample documents
    sample_texts = [
        ["machine", "learning", "algorithm", "data", "model"],
        ["deep", "learning", "neural", "network", "training"],
        ["natural", "language", "processing", "text", "nlp"],
        ["topic", "model", "lda", "document", "word"],
        ["computer", "vision", "image", "recognition", "cnn"],
    ]
    
    # Build dictionary and corpus
    dictionary = corpora.Dictionary(sample_texts)
    corpus = [dictionary.doc2bow(text) for text in sample_texts]
    
    # Train model
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,
        passes=10,
        random_state=42
    )
    
    # Evaluate
    evaluator = TopicEvaluator()
    result = evaluator.evaluate(model, sample_texts, dictionary, corpus)
    
    print(evaluator.generate_report(result))
