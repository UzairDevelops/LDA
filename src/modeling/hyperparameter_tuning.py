"""
Hyperparameter Tuning for LDA Topic Models.
Implements coherence-based selection of optimal number of topics (K).
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm

# Add project root to path for config import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import TUNING_CONFIG, OUTPUT_DIR, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    topic_range: List[int]
    coherence_scores: List[float]
    perplexity_scores: List[float]
    optimal_k: int
    optimal_coherence: float
    best_model: Optional[LdaModel] = None


class HyperparameterTuner:
    """
    Hyperparameter tuning for LDA models.
    
    Implements the elbow method for selecting optimal number of topics
    based on coherence scores, as recommended in LDA best practices.
    
    The coherence score (C_v) measures semantic interpretability of topics.
    Higher coherence indicates more coherent and interpretable topics.
    
    Usage:
        tuner = HyperparameterTuner(min_topics=2, max_topics=20)
        result = tuner.find_optimal_k(documents, dictionary, corpus)
        tuner.plot_results(result)
    """
    
    def __init__(
        self,
        min_topics: int = None,
        max_topics: int = None,
        step: int = None,
        coherence_measure: str = None,
        passes: int = 10,
        iterations: int = 400,
        random_state: int = 42
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            min_topics: Minimum number of topics to test
            max_topics: Maximum number of topics to test
            step: Topic number increment
            coherence_measure: Coherence metric ('c_v', 'u_mass', 'c_npmi')
            passes: LDA training passes
            iterations: LDA iterations per document
            random_state: Random seed for reproducibility
        """
        self.min_topics = min_topics or TUNING_CONFIG["min_topics"]
        self.max_topics = max_topics or TUNING_CONFIG["max_topics"]
        self.step = step or TUNING_CONFIG["step"]
        self.coherence_measure = coherence_measure or TUNING_CONFIG["coherence_measure"]
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state
        
        logger.info(f"HyperparameterTuner initialized: K range [{self.min_topics}, {self.max_topics}]")
    
    def find_optimal_k(
        self,
        texts: List[List[str]],
        dictionary: corpora.Dictionary = None,
        corpus: List = None,
        save_models: bool = False,
        show_progress: bool = True
    ) -> TuningResult:
        """
        Find optimal number of topics using coherence scores.
        
        Args:
            texts: List of tokenized documents
            dictionary: Pre-built dictionary (optional)
            corpus: Pre-built corpus (optional)
            save_models: Save all trained models
            show_progress: Show progress bar
            
        Returns:
            TuningResult with optimal K and all scores
        """
        # Build dictionary and corpus if not provided
        if dictionary is None:
            logger.info("Building dictionary...")
            dictionary = corpora.Dictionary(texts)
            dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        if corpus is None:
            corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Topic range to test
        topic_range = list(range(self.min_topics, self.max_topics + 1, self.step))
        
        coherence_scores = []
        perplexity_scores = []
        models = {}
        
        logger.info(f"Testing {len(topic_range)} topic configurations...")
        
        iterator = tqdm(topic_range, desc="Tuning K") if show_progress else topic_range
        
        for k in iterator:
            # Train LDA model
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=self.passes,
                iterations=self.iterations,
                random_state=self.random_state,
                alpha='auto',
                eta='auto'
            )
            
            # Calculate coherence
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=dictionary,
                coherence=self.coherence_measure
            )
            coherence = coherence_model.get_coherence()
            coherence_scores.append(coherence)
            
            # Calculate perplexity
            log_perplexity = model.log_perplexity(corpus)
            perplexity = np.exp2(-log_perplexity)
            perplexity_scores.append(perplexity)
            
            if save_models:
                models[k] = model
            
            logger.debug(f"K={k}: coherence={coherence:.4f}, perplexity={perplexity:.2f}")
        
        # Find optimal K (highest coherence)
        best_idx = np.argmax(coherence_scores)
        optimal_k = topic_range[best_idx]
        optimal_coherence = coherence_scores[best_idx]
        
        logger.info(f"Optimal K={optimal_k} with coherence={optimal_coherence:.4f}")
        
        # Get best model
        best_model = None
        if save_models:
            best_model = models.get(optimal_k)
        
        return TuningResult(
            topic_range=topic_range,
            coherence_scores=coherence_scores,
            perplexity_scores=perplexity_scores,
            optimal_k=optimal_k,
            optimal_coherence=optimal_coherence,
            best_model=best_model
        )
    
    def plot_results(
        self,
        result: TuningResult,
        save_path: Path = None,
        show_plot: bool = True
    ) -> Path:
        """
        Plot coherence and perplexity scores vs number of topics.
        
        Args:
            result: TuningResult from find_optimal_k
            save_path: Path to save plot (optional)
            show_plot: Display plot
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Coherence plot
        ax1.plot(result.topic_range, result.coherence_scores, 'b-o', linewidth=2, markersize=8)
        ax1.axvline(x=result.optimal_k, color='r', linestyle='--', label=f'Optimal K={result.optimal_k}')
        ax1.set_xlabel('Number of Topics (K)', fontsize=12)
        ax1.set_ylabel(f'Coherence Score ({self.coherence_measure})', fontsize=12)
        ax1.set_title('Topic Coherence vs Number of Topics', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal point
        ax1.scatter([result.optimal_k], [result.optimal_coherence], 
                    color='red', s=150, zorder=5, marker='*')
        
        # Perplexity plot
        ax2.plot(result.topic_range, result.perplexity_scores, 'g-o', linewidth=2, markersize=8)
        ax2.axvline(x=result.optimal_k, color='r', linestyle='--', label=f'Optimal K={result.optimal_k}')
        ax2.set_xlabel('Number of Topics (K)', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Perplexity vs Number of Topics', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = VISUALIZATIONS_DIR / "coherence_tuning.png"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved tuning plot to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def find_elbow_point(self, scores: List[float], topic_range: List[int]) -> int:
        """
        Find the elbow point in a curve using the maximum curvature method.
        
        Args:
            scores: List of scores (coherence)
            topic_range: Corresponding topic numbers
            
        Returns:
            Optimal K at elbow point
        """
        # Normalize scores to [0, 1]
        scores = np.array(scores)
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Calculate second derivative (curvature)
        first_derivative = np.gradient(normalized)
        second_derivative = np.gradient(first_derivative)
        
        # Find point of maximum curvature (elbow)
        elbow_idx = np.argmax(np.abs(second_derivative))
        
        return topic_range[elbow_idx]
    
    def generate_report(self, result: TuningResult) -> str:
        """
        Generate a text report of tuning results.
        
        Args:
            result: TuningResult from find_optimal_k
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("LDA HYPERPARAMETER TUNING REPORT")
        report.append("=" * 60)
        report.append(f"\nTopic Range: {result.topic_range[0]} - {result.topic_range[-1]}")
        report.append(f"Coherence Measure: {self.coherence_measure}")
        report.append(f"\nOptimal Number of Topics: K = {result.optimal_k}")
        report.append(f"Optimal Coherence Score: {result.optimal_coherence:.4f}")
        report.append("\n" + "-" * 40)
        report.append("Detailed Results:")
        report.append("-" * 40)
        
        for k, coh, perp in zip(result.topic_range, result.coherence_scores, result.perplexity_scores):
            marker = " <-- OPTIMAL" if k == result.optimal_k else ""
            report.append(f"K={k:2d}: Coherence={coh:.4f}, Perplexity={perp:8.2f}{marker}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample documents
    sample_texts = [
        ["machine", "learning", "algorithm", "data", "model", "training"],
        ["deep", "learning", "neural", "network", "backpropagation"],
        ["natural", "language", "processing", "text", "nlp", "word"],
        ["topic", "model", "lda", "document", "word", "distribution"],
        ["computer", "vision", "image", "recognition", "cnn", "object"],
        ["reinforcement", "learning", "agent", "reward", "policy"],
        ["data", "science", "analytics", "statistics", "visualization"],
        ["artificial", "intelligence", "machine", "cognitive", "reasoning"],
    ]
    
    # Run tuning
    tuner = HyperparameterTuner(min_topics=2, max_topics=5)
    result = tuner.find_optimal_k(sample_texts, save_models=False)
    
    # Generate report
    print(tuner.generate_report(result))
