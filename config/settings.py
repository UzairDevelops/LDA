"""
Configuration settings for LDA Topic Modeling project.
Based on Ahmed et al. (2022) methodology and Blei et al. (2003) LDA parameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Preprocessing settings (following Ahmed et al., 2022)
PREPROCESSING_CONFIG = {
    "min_token_length": 3,          # Minimum word length
    "max_token_length": 50,         # Maximum word length
    "remove_numbers": True,         # Remove numeric tokens
    "remove_punctuation": True,     # Remove punctuation
    "lowercase": True,              # Convert to lowercase
    "remove_stopwords": True,       # Remove English stop words
    "lemmatize": True,              # Apply lemmatization
    "use_bigrams": True,            # Detect and use bigrams
    "use_trigrams": False,          # Detect and use trigrams
    "min_word_frequency": 5,        # Minimum document frequency
    "max_word_frequency_pct": 0.5,  # Maximum document frequency (50%)
}

# LDA Model settings (based on Blei et al., 2003 and reference paper)
LDA_CONFIG = {
    "num_topics": 10,               # K=10 as per Ahmed et al. (2022)
    "alpha": "auto",                # Document-topic density (auto-learn)
    "eta": "auto",                  # Word-topic density (auto-learn)
    "passes": 15,                   # Training passes over corpus
    "iterations": 400,              # Inference iterations per document
    "eval_every": 10,               # Perplexity evaluation frequency
    "chunksize": 100,               # Documents per training chunk
    "random_state": 42,             # Reproducibility seed
    "per_word_topics": True,        # Enable per-word topic probabilities
}

# Hyperparameter tuning settings
TUNING_CONFIG = {
    "min_topics": 2,                # Minimum K to test
    "max_topics": 20,               # Maximum K to test
    "step": 1,                      # K increment step
    "coherence_measure": "c_v",     # Coherence metric (c_v recommended)
}

# Visualization settings (based on Sievert & Shirley, 2014)
VISUALIZATION_CONFIG = {
    "pyldavis_lambda": 0.6,         # Relevance metric (0.6 optimal per research)
    "pyldavis_sort_topics": True,   # Sort topics by prevalence
    "wordcloud_max_words": 50,      # Maximum words per cloud
    "wordcloud_width": 800,         # Word cloud width in pixels
    "wordcloud_height": 400,        # Word cloud height in pixels
    "tsne_perplexity": 30,          # t-SNE perplexity parameter
    "tsne_n_iter": 1000,            # t-SNE iterations
}

# SpaCy model for lemmatization
SPACY_MODEL = "en_core_web_sm"

# Custom stop words (extend NLTK defaults)
CUSTOM_STOPWORDS = {
    "et", "al", "fig", "figure", "table", "ref", "references",
    "doi", "http", "https", "www", "pdf", "vol", "pp", "page",
    "journal", "conference", "proceedings", "abstract", "keywords",
    "introduction", "conclusion", "results", "discussion", "methods",
    "methodology", "approach", "study", "research", "paper", "article",
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
