"""
pyLDAvis Interactive Visualization Generator.
Based on Sievert & Shirley (2014) - LDAvis methodology.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models

# Add project root to path for config import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import VISUALIZATION_CONFIG, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)


class PyLDAvisGenerator:
    """
    Interactive topic visualization using pyLDAvis.
    
    Implements the visualization methodology from:
    Sievert & Shirley (2014) - "LDAvis: A method for visualizing and interpreting topics"
    
    Features:
    - Global topic view showing topic prevalence and similarity
    - Per-topic term bar charts with corpus and topic frequencies
    - Adjustable λ (lambda) for relevance metric tuning
    
    Usage:
        generator = PyLDAvisGenerator()
        generator.generate(model, corpus, dictionary)
    """
    
    def __init__(
        self,
        sort_topics: bool = None,
        lambda_step: float = 0.01,
        mds: str = 'mmds'
    ):
        """
        Initialize visualization generator.
        
        Args:
            sort_topics: Sort topics by prevalence
            lambda_step: Step size for λ slider
            mds: Multidimensional scaling method ('mmds', 'pcoa', 'tsne')
        """
        self.sort_topics = sort_topics if sort_topics is not None else VISUALIZATION_CONFIG["pyldavis_sort_topics"]
        self.lambda_step = lambda_step
        self.mds = mds
        
        logger.info("PyLDAvisGenerator initialized")
    
    def prepare_visualization(
        self,
        model: LdaModel,
        corpus: list,
        dictionary: corpora.Dictionary
    ):
        """
        Prepare pyLDAvis visualization data.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            dictionary: Gensim dictionary
            
        Returns:
            pyLDAvis prepared data
        """
        logger.info("Preparing pyLDAvis visualization...")
        
        vis_data = pyLDAvis.gensim_models.prepare(
            model,
            corpus,
            dictionary,
            sort_topics=self.sort_topics,
            mds=self.mds
        )
        
        return vis_data
    
    def generate_html(
        self,
        model: LdaModel,
        corpus: list,
        dictionary: corpora.Dictionary,
        output_path: Union[str, Path] = None,
        open_browser: bool = False
    ) -> Path:
        """
        Generate interactive HTML visualization.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            dictionary: Gensim dictionary
            output_path: Path to save HTML file
            open_browser: Open in default browser
            
        Returns:
            Path to saved HTML file
        """
        # Prepare visualization data
        vis_data = self.prepare_visualization(model, corpus, dictionary)
        
        # Determine output path
        if output_path is None:
            output_path = VISUALIZATIONS_DIR / f"lda_topics_k{model.num_topics}.html"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save HTML
        pyLDAvis.save_html(vis_data, str(output_path))
        logger.info(f"Saved pyLDAvis visualization to {output_path}")
        
        # Open in browser if requested
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{output_path.absolute()}")
        
        return output_path
    
    def display_notebook(
        self,
        model: LdaModel,
        corpus: list,
        dictionary: corpora.Dictionary
    ):
        """
        Display visualization in Jupyter notebook.
        
        Args:
            model: Trained LDA model
            corpus: Bag-of-words corpus
            dictionary: Gensim dictionary
            
        Returns:
            IPython display object
        """
        vis_data = self.prepare_visualization(model, corpus, dictionary)
        
        # Enable notebook mode
        pyLDAvis.enable_notebook()
        
        return pyLDAvis.display(vis_data)


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
    
    # Generate visualization
    generator = PyLDAvisGenerator()
    output_path = generator.generate_html(model, corpus, dictionary, open_browser=True)
    print(f"Visualization saved to: {output_path}")
