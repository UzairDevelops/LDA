#!/usr/bin/env python
"""
LDA Topic Modeling Pipeline for Research Papers.

Main execution script that orchestrates the complete topic modeling workflow:
1. PDF text extraction
2. Text preprocessing
3. LDA model training
4. Evaluation and visualization

Based on methodology from Ahmed et al. (2022) - Sage Open.

Usage:
    python main.py --input data/raw/ --topics 10 --output data/output/
    python main.py --input data/raw/ --auto-k --output data/output/
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, 
    MODELS_DIR, VISUALIZATIONS_DIR, LDA_CONFIG
)
from src.extraction import PDFExtractor
from src.preprocessing import TextPreprocessor
from src.modeling import LDATopicModel, HyperparameterTuner
from src.evaluation import TopicEvaluator
from src.visualization import PyLDAvisGenerator, WordCloudGenerator, TSNEVisualizer


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / 'lda_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LDA Topic Modeling for Research Papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/raw/ --topics 10
  python main.py --input data/raw/ --auto-k --min-k 2 --max-k 20
  python main.py --input papers.txt --topics 10 --text-file
        """
    )
    
    # Input options
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=str(RAW_DATA_DIR),
        help='Path to input PDFs directory or text file (default: data/raw/)'
    )
    
    parser.add_argument(
        '--text-file',
        action='store_true',
        help='Input is a text file with one document per line'
    )
    
    # Topic options
    parser.add_argument(
        '--topics', '-k',
        type=int,
        default=LDA_CONFIG['num_topics'],
        help=f'Number of topics K (default: {LDA_CONFIG["num_topics"]})'
    )
    
    parser.add_argument(
        '--auto-k',
        action='store_true',
        help='Automatically find optimal K using coherence scores'
    )
    
    parser.add_argument(
        '--min-k',
        type=int,
        default=2,
        help='Minimum K for auto-selection (default: 2)'
    )
    
    parser.add_argument(
        '--max-k',
        type=int,
        default=20,
        help='Maximum K for auto-selection (default: 20)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory (default: data/output/)'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    
    # Processing options
    parser.add_argument(
        '--passes',
        type=int,
        default=LDA_CONFIG['passes'],
        help=f'LDA training passes (default: {LDA_CONFIG["passes"]})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_documents(input_path: str, is_text_file: bool, logger) -> list:
    """Load documents from PDFs or text file."""
    input_path = Path(input_path)
    
    if is_text_file:
        logger.info(f"Loading documents from text file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    else:
        logger.info(f"Extracting PDFs from: {input_path}")
        extractor = PDFExtractor()
        
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            docs = [extractor.extract_text_from_pdf(input_path)]
        else:
            docs = extractor.extract_from_directory(input_path, recursive=True)
        
        logger.info(f"Extracted {len(docs)} documents")
        return docs


def run_pipeline(args, logger):
    """Run the complete topic modeling pipeline."""
    start_time = datetime.now()
    
    # Ensure output directories exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    vis_dir = output_dir / 'visualizations'
    models_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    # Step 1: Load documents
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Documents")
    logger.info("=" * 60)
    
    documents = load_documents(args.input, args.text_file, logger)
    
    if not documents:
        logger.error("No documents found. Please check input path.")
        return None
    
    # Step 2: Preprocess
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)
    
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    
    # Get tokens
    texts = [doc.tokens for doc in processed_docs]
    
    stats = preprocessor.get_corpus_stats(processed_docs)
    logger.info(f"Corpus stats: {stats['num_documents']} docs, {stats['unique_tokens']} unique tokens")
    
    # Step 3: Determine K
    num_topics = args.topics
    
    if args.auto_k:
        logger.info("=" * 60)
        logger.info("STEP 3: Finding Optimal K")
        logger.info("=" * 60)
        
        tuner = HyperparameterTuner(
            min_topics=args.min_k,
            max_topics=args.max_k
        )
        tuning_result = tuner.find_optimal_k(texts)
        num_topics = tuning_result.optimal_k
        
        # Save tuning plot
        tuner.plot_results(tuning_result, save_path=vis_dir / 'coherence_tuning.png', show_plot=False)
        
        logger.info(f"Optimal K = {num_topics} (coherence = {tuning_result.optimal_coherence:.4f})")
    
    # Step 4: Train LDA
    logger.info("=" * 60)
    logger.info(f"STEP 4: Training LDA (K={num_topics})")
    logger.info("=" * 60)
    
    lda = LDATopicModel(
        num_topics=num_topics,
        passes=args.passes
    )
    result = lda.train(texts)
    
    # Print topics
    lda.print_topics()
    
    # Save model
    model_path = models_dir / f'lda_k{num_topics}.model'
    lda.save(model_path)
    
    # Step 5: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluation")
    logger.info("=" * 60)
    
    evaluator = TopicEvaluator()
    eval_result = evaluator.evaluate(
        lda.model, texts, lda.dictionary, lda.corpus
    )
    
    report = evaluator.generate_report(eval_result)
    print(report)
    
    # Save report
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Step 6: Visualizations
    if not args.no_visualizations:
        logger.info("=" * 60)
        logger.info("STEP 6: Generating Visualizations")
        logger.info("=" * 60)
        
        # pyLDAvis
        try:
            pyldavis_gen = PyLDAvisGenerator()
            pyldavis_gen.generate_html(
                lda.model, lda.corpus, lda.dictionary,
                output_path=vis_dir / 'pyldavis_topics.html'
            )
        except Exception as e:
            logger.warning(f"pyLDAvis generation failed: {e}")
        
        # Word clouds
        try:
            wordcloud_gen = WordCloudGenerator()
            wordcloud_gen.generate_grid(
                lda.model,
                output_path=vis_dir / 'wordcloud_grid.png',
                show_plot=False
            )
            wordcloud_gen.generate_all_topics(
                lda.model,
                output_dir=vis_dir / 'wordclouds'
            )
        except Exception as e:
            logger.warning(f"Word cloud generation failed: {e}")
        
        # t-SNE
        try:
            tsne_vis = TSNEVisualizer(perplexity=min(30, len(texts) - 1))
            tsne_vis.visualize_documents(
                lda.model, lda.corpus,
                output_path=vis_dir / 'tsne_documents.png',
                show_plot=False
            )
        except Exception as e:
            logger.warning(f"t-SNE visualization failed: {e}")
    
    # Summary
    elapsed = datetime.now() - start_time
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Topics discovered: {num_topics}")
    logger.info(f"Coherence (C_v): {eval_result.coherence_cv:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Elapsed time: {elapsed}")
    
    return {
        'model': lda,
        'result': result,
        'evaluation': eval_result,
        'output_dir': output_dir
    }


def main():
    """Main entry point."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    try:
        results = run_pipeline(args, logger)
        
        if results:
            print("\n✓ Topic modeling complete!")
            print(f"  See results in: {results['output_dir']}")
        else:
            print("\n✗ Pipeline failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
