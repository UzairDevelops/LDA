"""
Word Cloud Generator for Topic Visualization.
Creates word clouds for each topic based on word probabilities.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import LdaModel
import numpy as np

# Add project root to path for config import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import VISUALIZATION_CONFIG, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)


class WordCloudGenerator:
    """
    Generate word clouds for LDA topics.
    
    Word size is proportional to word probability within each topic.
    Provides visual summary of topic content for quick interpretation.
    
    Usage:
        generator = WordCloudGenerator()
        generator.generate_all_topics(model, "output/wordclouds/")
    """
    
    def __init__(
        self,
        width: int = None,
        height: int = None,
        max_words: int = None,
        background_color: str = "white",
        colormap: str = "viridis",
        prefer_horizontal: float = 0.9
    ):
        """
        Initialize word cloud generator.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            max_words: Maximum words per cloud
            background_color: Background color
            colormap: Matplotlib colormap name
            prefer_horizontal: Ratio of horizontal words
        """
        self.width = width or VISUALIZATION_CONFIG["wordcloud_width"]
        self.height = height or VISUALIZATION_CONFIG["wordcloud_height"]
        self.max_words = max_words or VISUALIZATION_CONFIG["wordcloud_max_words"]
        self.background_color = background_color
        self.colormap = colormap
        self.prefer_horizontal = prefer_horizontal
        
        logger.info("WordCloudGenerator initialized")
    
    def generate_topic_wordcloud(
        self,
        model: LdaModel,
        topic_id: int,
        output_path: Union[str, Path] = None,
        title: str = None,
        show_plot: bool = False
    ) -> Optional[Path]:
        """
        Generate word cloud for a single topic.
        
        Args:
            model: Trained LDA model
            topic_id: Topic ID to visualize
            output_path: Path to save image
            title: Optional title for the plot
            show_plot: Display the plot
            
        Returns:
            Path to saved image or None
        """
        # Get topic words and probabilities
        topic_words = model.show_topic(topic_id, topn=self.max_words)
        word_freq = {word: prob for word, prob in topic_words}
        
        if not word_freq:
            logger.warning(f"No words found for topic {topic_id}")
            return None
        
        # Create word cloud
        wc = WordCloud(
            width=self.width,
            height=self.height,
            max_words=self.max_words,
            background_color=self.background_color,
            colormap=self.colormap,
            prefer_horizontal=self.prefer_horizontal,
            random_state=42
        )
        
        wc.generate_from_frequencies(word_freq)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Topic {topic_id}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        saved_path = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            saved_path = output_path
            logger.debug(f"Saved word cloud to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return saved_path
    
    def generate_all_topics(
        self,
        model: LdaModel,
        output_dir: Union[str, Path] = None,
        topic_labels: Dict[int, str] = None,
        show_plots: bool = False
    ) -> List[Path]:
        """
        Generate word clouds for all topics.
        
        Args:
            model: Trained LDA model
            output_dir: Directory to save images
            topic_labels: Optional custom labels for topics
            show_plots: Display plots
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = VISUALIZATIONS_DIR / "wordclouds"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for topic_id in range(model.num_topics):
            # Get title
            if topic_labels and topic_id in topic_labels:
                title = f"Topic {topic_id}: {topic_labels[topic_id]}"
            else:
                # Generate title from top words
                top_words = [w for w, _ in model.show_topic(topic_id, topn=3)]
                title = f"Topic {topic_id}: {', '.join(top_words)}"
            
            output_path = output_dir / f"topic_{topic_id}_wordcloud.png"
            
            path = self.generate_topic_wordcloud(
                model,
                topic_id,
                output_path=output_path,
                title=title,
                show_plot=show_plots
            )
            
            if path:
                saved_paths.append(path)
        
        logger.info(f"Generated {len(saved_paths)} word clouds in {output_dir}")
        return saved_paths
    
    def generate_grid(
        self,
        model: LdaModel,
        output_path: Union[str, Path] = None,
        cols: int = 4,
        topic_labels: Dict[int, str] = None,
        show_plot: bool = True
    ) -> Optional[Path]:
        """
        Generate a grid of word clouds for all topics.
        
        Args:
            model: Trained LDA model
            output_path: Path to save combined image
            cols: Number of columns in grid
            topic_labels: Optional custom labels
            show_plot: Display the plot
            
        Returns:
            Path to saved image
        """
        num_topics = model.num_topics
        rows = (num_topics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten() if num_topics > 1 else [axes]
        
        for topic_id in range(num_topics):
            ax = axes[topic_id]
            
            # Get topic words
            topic_words = model.show_topic(topic_id, topn=self.max_words)
            word_freq = {word: prob for word, prob in topic_words}
            
            if word_freq:
                wc = WordCloud(
                    width=400,
                    height=300,
                    max_words=30,
                    background_color=self.background_color,
                    colormap=self.colormap,
                    random_state=42
                )
                wc.generate_from_frequencies(word_freq)
                ax.imshow(wc, interpolation='bilinear')
            
            ax.axis('off')
            
            # Set title
            if topic_labels and topic_id in topic_labels:
                title = f"Topic {topic_id}: {topic_labels[topic_id]}"
            else:
                top_words = [w for w, _ in model.show_topic(topic_id, topn=2)]
                title = f"Topic {topic_id}: {', '.join(top_words)}"
            ax.set_title(title, fontsize=10)
        
        # Hide empty subplots
        for idx in range(num_topics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('LDA Topic Word Clouds', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        saved_path = None
        if output_path is None:
            output_path = VISUALIZATIONS_DIR / "topic_wordclouds_grid.png"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        saved_path = output_path
        logger.info(f"Saved word cloud grid to {output_path}")
        
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
    
    # Generate word clouds
    generator = WordCloudGenerator()
    generator.generate_grid(model, show_plot=True)
