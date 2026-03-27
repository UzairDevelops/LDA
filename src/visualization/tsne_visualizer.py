"""
t-SNE Visualization for Topic Modeling.
Based on Van der Maaten & Hinton (2008) - t-SNE dimensionality reduction.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import LdaModel
from gensim import corpora

# Add project root to path for config import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import VISUALIZATION_CONFIG, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)


class TSNEVisualizer:
    """
    t-SNE visualization for topic model analysis.
    
    Implements dimensionality reduction based on:
    Van der Maaten & Hinton (2008) - "Visualizing Data using t-SNE"
    
    Provides:
    - Document clustering visualization (documents colored by dominant topic)
    - Topic word embedding visualization
    - Interactive exploration of topic relationships
    
    Usage:
        visualizer = TSNEVisualizer()
        visualizer.visualize_documents(model, corpus)
    """
    
    def __init__(
        self,
        perplexity: int = None,
        n_iter: int = None,
        random_state: int = 42,
        learning_rate: Union[float, str] = 'auto'
    ):
        """
        Initialize t-SNE visualizer.
        
        Args:
            perplexity: t-SNE perplexity parameter (related to local neighbors)
            n_iter: Number of optimization iterations
            random_state: Random seed for reproducibility
            learning_rate: Learning rate for optimization
        """
        self.perplexity = perplexity or VISUALIZATION_CONFIG["tsne_perplexity"]
        self.n_iter = n_iter or VISUALIZATION_CONFIG["tsne_n_iter"]
        self.random_state = random_state
        self.learning_rate = learning_rate
        
        logger.info("TSNEVisualizer initialized")
    
    def get_document_topic_matrix(
        self,
        model: LdaModel,
        corpus: list
    ) -> np.ndarray:
        """
        Get document-topic distribution matrix.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            
        Returns:
            NumPy array of shape (num_docs, num_topics)
        """
        doc_topics = []
        
        for doc_bow in corpus:
            topic_dist = model.get_document_topics(doc_bow, minimum_probability=0.0)
            # Ensure full vector
            topic_vec = np.zeros(model.num_topics)
            for topic_id, prob in topic_dist:
                topic_vec[topic_id] = prob
            doc_topics.append(topic_vec)
        
        return np.array(doc_topics)
    
    def fit_tsne(
        self,
        data: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """
        Fit t-SNE to data.
        
        Args:
            data: High-dimensional data matrix
            n_components: Target dimensionality (2 or 3)
            
        Returns:
            Reduced dimensionality array
        """
        # Adjust perplexity if needed
        effective_perplexity = min(self.perplexity, len(data) - 1)
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=effective_perplexity,
            n_iter=self.n_iter,
            random_state=self.random_state,
            learning_rate=self.learning_rate
        )
        
        return tsne.fit_transform(data)
    
    def visualize_documents(
        self,
        model: LdaModel,
        corpus: list,
        doc_labels: List[str] = None,
        output_path: Union[str, Path] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Optional[Path]:
        """
        Visualize documents in 2D space, colored by dominant topic.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            doc_labels: Optional document labels
            output_path: Path to save plot
            show_plot: Display the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        logger.info("Generating document t-SNE visualization...")
        
        # Get document-topic matrix
        doc_topics = self.get_document_topic_matrix(model, corpus)
        
        # Get dominant topic for each document
        dominant_topics = np.argmax(doc_topics, axis=1)
        
        # Fit t-SNE
        coords = self.fit_tsne(doc_topics)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, model.num_topics))
        
        # Plot each topic's documents
        for topic_id in range(model.num_topics):
            mask = dominant_topics == topic_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[colors[topic_id]],
                label=f'Topic {topic_id}',
                alpha=0.7,
                s=50
            )
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Document Clustering by Topic (t-SNE)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save
        saved_path = None
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = VISUALIZATIONS_DIR / "tsne_documents.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        saved_path = output_path
        logger.info(f"Saved t-SNE document plot to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return saved_path
    
    def visualize_topic_words(
        self,
        model: LdaModel,
        num_words: int = 20,
        output_path: Union[str, Path] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Optional[Path]:
        """
        Visualize topic keywords in 2D space.
        
        Words are positioned based on their topic distributions.
        
        Args:
            model: Trained LDA model
            num_words: Number of top words per topic
            output_path: Path to save plot
            show_plot: Display the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        logger.info("Generating topic words t-SNE visualization...")
        
        # Collect all unique words and their topic vectors
        word_topic_vectors = {}
        word_topic_assignments = {}
        
        for topic_id in range(model.num_topics):
            topic_words = model.show_topic(topic_id, topn=num_words)
            
            for word, prob in topic_words:
                if word not in word_topic_vectors:
                    # Initialize word's topic vector
                    word_topic_vectors[word] = np.zeros(model.num_topics)
                    word_topic_assignments[word] = topic_id
                
                word_topic_vectors[word][topic_id] = prob
                
                # Update assignment if higher probability
                if prob > word_topic_vectors[word][word_topic_assignments[word]]:
                    word_topic_assignments[word] = topic_id
        
        # Convert to matrix
        words = list(word_topic_vectors.keys())
        word_matrix = np.array([word_topic_vectors[w] for w in words])
        
        # Fit t-SNE
        coords = self.fit_tsne(word_matrix)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, model.num_topics))
        
        # Plot words
        for i, word in enumerate(words):
            topic_id = word_topic_assignments[word]
            ax.scatter(coords[i, 0], coords[i, 1], c=[colors[topic_id]], s=30, alpha=0.7)
            ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.8)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c=[colors[i]], label=f'Topic {i}', s=50)
            for i in range(model.num_topics)
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Topic Keywords Visualization (t-SNE)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        saved_path = None
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = VISUALIZATIONS_DIR / "tsne_topic_words.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        saved_path = output_path
        logger.info(f"Saved t-SNE words plot to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return saved_path
    
    def visualize_topic_centers(
        self,
        model: LdaModel,
        corpus: list,
        output_path: Union[str, Path] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[Path]:
        """
        Visualize topic centers and their relative positions.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            output_path: Path to save plot
            show_plot: Display the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        logger.info("Generating topic centers t-SNE visualization...")
        
        # Get document-topic matrix
        doc_topics = self.get_document_topic_matrix(model, corpus)
        
        # Calculate topic centers (mean document vector per topic)
        dominant_topics = np.argmax(doc_topics, axis=1)
        
        topic_centers = []
        topic_sizes = []
        
        for topic_id in range(model.num_topics):
            mask = dominant_topics == topic_id
            if np.any(mask):
                center = doc_topics[mask].mean(axis=0)
                topic_centers.append(center)
                topic_sizes.append(np.sum(mask))
            else:
                topic_centers.append(np.zeros(model.num_topics))
                topic_sizes.append(0)
        
        topic_centers = np.array(topic_centers)
        topic_sizes = np.array(topic_sizes)
        
        # Fit t-SNE on topic centers
        if len(topic_centers) > 2:
            coords = self.fit_tsne(topic_centers)
        else:
            # Not enough topics for t-SNE, use direct positions
            coords = topic_centers[:, :2]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, model.num_topics))
        
        # Normalize sizes for display
        size_normalized = 100 + 500 * (topic_sizes / topic_sizes.max()) if topic_sizes.max() > 0 else topic_sizes * 0 + 200
        
        # Plot topic centers
        for i in range(model.num_topics):
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                c=[colors[i]],
                s=size_normalized[i],
                alpha=0.7,
                edgecolors='black',
                linewidths=2
            )
            
            # Add topic label
            top_words = [w for w, _ in model.show_topic(i, topn=3)]
            label = f"T{i}: {', '.join(top_words)}"
            ax.annotate(label, (coords[i, 0], coords[i, 1]), 
                       fontsize=9, ha='center', va='bottom',
                       xytext=(0, 10), textcoords='offset points')
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Topic Centers (size = document count)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        saved_path = None
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = VISUALIZATIONS_DIR / "tsne_topic_centers.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        saved_path = output_path
        logger.info(f"Saved t-SNE topic centers plot to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return saved_path


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
        ["reinforcement", "learning", "agent", "reward", "policy"],
        ["data", "science", "analytics", "statistics", "visualization"],
        ["artificial", "intelligence", "machine", "cognitive", "reasoning"],
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
    
    # Generate visualizations
    visualizer = TSNEVisualizer(perplexity=5)  # Low perplexity for small dataset
    visualizer.visualize_documents(model, corpus, show_plot=True)
