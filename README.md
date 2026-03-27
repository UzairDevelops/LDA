# LDA Topic Modeling for Research Papers

A comprehensive topic modeling system using Latent Dirichlet Allocation (LDA) for analyzing research papers.

**Based on:** Ahmed et al. (2022) - "Topic Modeling of the Pakistani Economy in English Newspapers via LDA" ([Sage Open](https://journals.sagepub.com/doi/full/10.1177/21582440221079931))

## Features

- **PDF Text Extraction**: Automatic extraction from research paper PDFs
- **Smart Preprocessing**: Tokenization, lemmatization, stop word removal, bigram detection
- **LDA Training**: Gensim-based implementation with configurable parameters
- **Automatic K Selection**: Find optimal number of topics using coherence scores
- **Comprehensive Evaluation**: Coherence (C_v, UMass), perplexity, diversity metrics
- **Rich Visualizations**:
  - pyLDAvis interactive HTML
  - Word clouds per topic
  - t-SNE document clustering

## Installation

```bash
# Clone or navigate to project directory
cd c:\Users\AL AZIZ TECH\Desktop\LDA

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Basic Usage

```bash
# With PDFs in data/raw/
python main.py --input data/raw/ --topics 10

# Auto-select optimal K
python main.py --input data/raw/ --auto-k

# With text file (one document per line)
python main.py --input papers.txt --topics 10 --text-file
```

### 2. Python API

```python
from src.extraction import PDFExtractor
from src.preprocessing import TextPreprocessor
from src.modeling import LDATopicModel

# Extract PDFs
extractor = PDFExtractor()
documents = extractor.extract_from_directory("data/raw/")

# Preprocess
preprocessor = TextPreprocessor()
processed_docs = preprocessor.preprocess_documents(documents)

# Train LDA
lda = LDATopicModel(num_topics=10)
result = lda.train(processed_docs)

# View topics
lda.print_topics()
print(f"Coherence: {result.coherence_score:.4f}")
```

## Project Structure

```
LDA/
├── config/
│   └── settings.py          # Configuration parameters
├── data/
│   ├── raw/                  # Input PDFs
│   ├── processed/            # Extracted text
│   └── output/               # Results
│       ├── models/           # Saved LDA models
│       └── visualizations/   # Charts and HTML
├── src/
│   ├── extraction/           # PDF processing
│   ├── preprocessing/        # Text cleaning
│   ├── modeling/             # LDA implementation
│   ├── evaluation/           # Metrics
│   └── visualization/        # Charts
├── main.py                   # CLI entry point
└── requirements.txt
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_topics` | 10 | Number of topics (K) |
| `alpha` | auto | Document-topic density |
| `eta` | auto | Word-topic density |
| `passes` | 15 | Training iterations |

See `config/settings.py` for all configurable parameters.

## Methodology

Following the approach from Ahmed et al. (2022):

1. **Preprocessing**: Tokenization → Normalization → Stop words → Lemmatization
2. **LDA Training**: Generative probabilistic model (Blei et al., 2003)
3. **Visualization**: pyLDAvis with relevance metric (Sievert & Shirley, 2014)
4. **Evaluation**: Coherence scores for topic quality assessment

## Output Files

After running the pipeline:

- `models/lda_k{K}.model` - Trained LDA model
- `visualizations/pyldavis_topics.html` - Interactive visualization
- `visualizations/wordcloud_grid.png` - Topic word clouds
- `visualizations/tsne_documents.png` - Document clustering
- `evaluation_report.txt` - Metrics summary

## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. JMLR.
- Sievert, C., & Shirley, K. (2014). *LDAvis: A method for visualizing and interpreting topics*. ACL Workshop.
- Van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. JMLR.

## License

MIT License
