#!/usr/bin/env python
"""
Download research articles from arXiv for topic modeling.
Downloads recent papers in Machine Learning/AI/NLP categories.
"""

import arxiv
import os
import time
from pathlib import Path
from tqdm import tqdm


def download_arxiv_papers(
    output_dir: str,
    num_papers: int = 200,
    categories: list = None,
    search_query: str = None
):
    """
    Download papers from arXiv.
    
    Args:
        output_dir: Directory to save PDFs
        num_papers: Number of papers to download
        categories: arXiv categories to search
        search_query: Custom search query
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default: AI/ML/NLP topics for diverse topic modeling
    if search_query is None:
        search_query = (
            "cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR "  # NLP, ML, AI
            "cat:cs.IR OR cat:stat.ML"  # Information Retrieval, Stats ML
        )
    
    print(f"Searching arXiv for: {search_query}")
    print(f"Target: {num_papers} papers")
    print(f"Output: {output_path}")
    print("-" * 50)
    
    # Create arXiv client
    client = arxiv.Client(
        page_size=50,
        delay_seconds=3.0,  # Be respectful to arXiv servers
        num_retries=3
    )
    
    # Search for papers
    search = arxiv.Search(
        query=search_query,
        max_results=num_papers + 50,  # Request extra in case of failures
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    downloaded = 0
    failed = 0
    
    # Download papers
    print("\nDownloading papers...")
    
    try:
        results = list(client.results(search))
        print(f"Found {len(results)} papers")
        
        for paper in tqdm(results, desc="Downloading"):
            if downloaded >= num_papers:
                break
            
            try:
                # Create safe filename
                paper_id = paper.get_short_id().replace("/", "_")
                filename = f"{paper_id}.pdf"
                filepath = output_path / filename
                
                # Skip if already downloaded
                if filepath.exists():
                    downloaded += 1
                    continue
                
                # Download PDF
                paper.download_pdf(dirpath=str(output_path), filename=filename)
                downloaded += 1
                
                # Small delay to be nice to arXiv
                time.sleep(0.5)
                
            except Exception as e:
                failed += 1
                print(f"\nFailed to download {paper.title[:50]}...: {e}")
                continue
    
    except Exception as e:
        print(f"Search failed: {e}")
        return downloaded, failed
    
    print("\n" + "=" * 50)
    print(f"Download Complete!")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {output_path}")
    print("=" * 50)
    
    # Save metadata
    metadata_path = output_path / "papers_metadata.txt"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Downloaded Papers Metadata\n")
        f.write("=" * 50 + "\n\n")
        
        for i, paper in enumerate(results[:downloaded]):
            f.write(f"{i+1}. {paper.title}\n")
            f.write(f"   ID: {paper.get_short_id()}\n")
            f.write(f"   Authors: {', '.join(a.name for a in paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}\n")
            f.write(f"   Date: {paper.published.date()}\n")
            f.write(f"   Categories: {', '.join(paper.categories)}\n")
            f.write("\n")
    
    print(f"Metadata saved to: {metadata_path}")
    
    return downloaded, failed


if __name__ == "__main__":
    import sys
    
    # Configuration
    OUTPUT_DIR = "data/Articles"
    NUM_PAPERS = 200
    
    # Check for command line args
    if len(sys.argv) > 1:
        NUM_PAPERS = int(sys.argv[1])
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    
    # Download papers
    downloaded, failed = download_arxiv_papers(
        output_dir=OUTPUT_DIR,
        num_papers=NUM_PAPERS
    )
    
    if downloaded > 0:
        print(f"\nReady to run LDA pipeline on {downloaded} papers!")
        print(f"Run: python main.py --input {OUTPUT_DIR} --topics 10")
