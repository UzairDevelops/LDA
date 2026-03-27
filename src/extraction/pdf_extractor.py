"""
PDF Text Extraction Module.
Extracts text content from research paper PDFs using PyMuPDF.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    """Represents an extracted document with metadata."""
    filename: str
    filepath: str
    text: str
    title: Optional[str] = None
    num_pages: int = 0
    metadata: Optional[Dict] = None
    
    def __repr__(self):
        return f"ExtractedDocument(filename='{self.filename}', pages={self.num_pages}, chars={len(self.text)})"


class PDFExtractor:
    """
    PDF text extraction for research papers.
    
    Based on best practices from PDF-Extract-Kit and PyMuPDF documentation.
    Handles full text extraction with optional section isolation.
    
    Usage:
        extractor = PDFExtractor()
        documents = extractor.extract_from_directory("data/raw/")
    """
    
    def __init__(self, extract_metadata: bool = True):
        """
        Initialize PDF extractor.
        
        Args:
            extract_metadata: Whether to extract PDF metadata (title, author, etc.)
        """
        self.extract_metadata = extract_metadata
        logger.info("PDFExtractor initialized")
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> ExtractedDocument:
        """
        Extract text content from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractedDocument with text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            # Extract metadata if requested
            metadata = None
            title = None
            if self.extract_metadata:
                metadata = doc.metadata
                title = metadata.get("title", "").strip() or None
            
            extracted_doc = ExtractedDocument(
                filename=pdf_path.name,
                filepath=str(pdf_path.absolute()),
                text=full_text,
                title=title,
                num_pages=len(doc),
                metadata=metadata
            )
            
            doc.close()
            logger.debug(f"Extracted {len(full_text)} chars from {pdf_path.name}")
            
            return extracted_doc
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            raise
    
    def extract_from_directory(
        self, 
        directory: Union[str, Path], 
        recursive: bool = False,
        show_progress: bool = True
    ) -> List[ExtractedDocument]:
        """
        Extract text from all PDFs in a directory.
        
        Args:
            directory: Path to directory containing PDFs
            recursive: Search subdirectories recursively
            show_progress: Show progress bar
            
        Returns:
            List of ExtractedDocument objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        # Find all PDF files
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        # Extract text from each PDF
        documents = []
        iterator = tqdm(pdf_files, desc="Extracting PDFs") if show_progress else pdf_files
        
        for pdf_path in iterator:
            try:
                doc = self.extract_text_from_pdf(pdf_path)
                if doc.text.strip():  # Only add if text was extracted
                    documents.append(doc)
                else:
                    logger.warning(f"No text extracted from {pdf_path.name}")
            except Exception as e:
                logger.error(f"Skipping {pdf_path.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully extracted {len(documents)} documents")
        return documents
    
    def extract_abstract(self, text: str) -> Optional[str]:
        """
        Attempt to extract abstract section from document text.
        
        Args:
            text: Full document text
            
        Returns:
            Abstract text if found, None otherwise
        """
        text_lower = text.lower()
        
        # Find abstract start
        abstract_markers = ["abstract", "summary"]
        abstract_start = -1
        
        for marker in abstract_markers:
            pos = text_lower.find(marker)
            if pos != -1:
                abstract_start = pos + len(marker)
                break
        
        if abstract_start == -1:
            return None
        
        # Find abstract end (next section)
        end_markers = ["introduction", "keywords", "1.", "1 "]
        abstract_end = len(text)
        
        for marker in end_markers:
            pos = text_lower.find(marker, abstract_start)
            if pos != -1 and pos < abstract_end:
                abstract_end = pos
        
        # Extract and clean abstract
        abstract = text[abstract_start:abstract_end].strip()
        
        # Basic cleaning
        abstract = abstract.lstrip(":.-–—\n\r\t ")
        
        # Limit length (abstracts are typically 150-300 words)
        words = abstract.split()
        if len(words) > 500:
            abstract = " ".join(words[:500])
        
        return abstract if len(abstract) > 50 else None
    
    def save_extracted_texts(
        self, 
        documents: List[ExtractedDocument], 
        output_dir: Union[str, Path],
        format: str = "txt"
    ) -> List[Path]:
        """
        Save extracted texts to files.
        
        Args:
            documents: List of extracted documents
            output_dir: Directory to save text files
            format: Output format ('txt' or 'json')
            
        Returns:
            List of saved file paths
        """
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for doc in documents:
            base_name = Path(doc.filename).stem
            
            if format == "txt":
                output_path = output_dir / f"{base_name}.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(doc.text)
            elif format == "json":
                output_path = output_dir / f"{base_name}.json"
                data = {
                    "filename": doc.filename,
                    "title": doc.title,
                    "num_pages": doc.num_pages,
                    "text": doc.text,
                    "metadata": doc.metadata
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            saved_files.append(output_path)
            logger.debug(f"Saved {output_path}")
        
        logger.info(f"Saved {len(saved_files)} files to {output_dir}")
        return saved_files


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        extractor = PDFExtractor()
        
        if os.path.isfile(pdf_path):
            doc = extractor.extract_text_from_pdf(pdf_path)
            print(f"Extracted: {doc}")
            print(f"First 500 chars:\n{doc.text[:500]}...")
        elif os.path.isdir(pdf_path):
            docs = extractor.extract_from_directory(pdf_path)
            print(f"Extracted {len(docs)} documents")
    else:
        print("Usage: python pdf_extractor.py <pdf_file_or_directory>")
