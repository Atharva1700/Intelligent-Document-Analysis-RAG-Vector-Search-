"""
Document Processor - Handles ingestion of 10K+ documents
Supports PDF, DOCX, TXT, HTML, Markdown
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    DirectoryLoader,
)
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    total_docs: int = 0
    processed: int = 0
    failed: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class DocumentProcessor:
    """
    High-performance document processor optimized for 10K+ documents.
    
    Key optimizations:
    - Parallel processing with ThreadPoolExecutor
    - Content deduplication via MD5 hashing
    - Smart chunking with overlap for context preservation
    - Metadata enrichment for better retrieval
    """

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        max_workers: int = 8,
    ):
        """
        Args:
            chunk_size: Token size per chunk (512 balances context vs. precision)
            chunk_overlap: Overlap between chunks (128 preserves cross-chunk context)
            max_workers: Parallel processing threads
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self._seen_hashes: set = set()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def _load_document(self, file_path: str) -> List[Document]:
        """Load a single document using the appropriate loader."""
        path = Path(file_path)
        ext = path.suffix.lower()
        loader_cls = self.LOADER_MAP.get(ext)

        if not loader_cls:
            raise ValueError(f"Unsupported file type: {ext}")

        loader = loader_cls(file_path)
        docs = loader.load()

        # Enrich metadata
        for doc in docs:
            doc.metadata.update({
                "source": str(path),
                "filename": path.name,
                "file_type": ext,
                "file_size": path.stat().st_size,
            })

        return docs

    def _process_single_file(self, file_path: str) -> tuple[List[Document], Optional[str]]:
        """Process a single file: load → deduplicate → chunk."""
        try:
            docs = self._load_document(file_path)
            all_chunks = []

            for doc in docs:
                # Deduplication check
                content_hash = self._compute_hash(doc.page_content)
                if content_hash in self._seen_hashes:
                    logger.debug(f"Duplicate skipped: {file_path}")
                    continue
                self._seen_hashes.add(content_hash)

                # Split into chunks
                chunks = self.text_splitter.split_documents([doc])

                # Add chunk-level metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["chunk_total"] = len(chunks)
                    chunk.metadata["chunk_hash"] = self._compute_hash(chunk.page_content)

                all_chunks.extend(chunks)

            return all_chunks, None

        except Exception as e:
            return [], str(e)

    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> tuple[List[Document], ProcessingStats]:
        """
        Process all documents in a directory with parallel execution.
        
        Returns:
            Tuple of (all_chunks, processing_stats)
        """
        stats = ProcessingStats()
        start_time = time.time()

        # Collect all files
        dir_path = Path(directory)
        pattern = "**/*" if recursive else "*"
        all_files = [
            str(f) for f in dir_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.LOADER_MAP
        ]

        stats.total_docs = len(all_files)
        logger.info(f"Found {stats.total_docs} documents to process")

        all_chunks: List[Document] = []

        # Parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, f): f
                for f in all_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                chunks, error = future.result()

                if error:
                    stats.failed += 1
                    stats.errors.append(f"{file_path}: {error}")
                    logger.error(f"Failed: {file_path} → {error}")
                else:
                    stats.processed += 1
                    stats.total_chunks += len(chunks)
                    all_chunks.extend(chunks)

                if stats.processed % 100 == 0:
                    logger.info(f"Progress: {stats.processed}/{stats.total_docs}")

        stats.processing_time = time.time() - start_time
        logger.info(
            f"Processing complete: {stats.processed} docs, "
            f"{stats.total_chunks} chunks in {stats.processing_time:.2f}s"
        )

        return all_chunks, stats

    def process_files(self, file_paths: List[str]) -> tuple[List[Document], ProcessingStats]:
        """Process a specific list of files."""
        stats = ProcessingStats()
        stats.total_docs = len(file_paths)
        start_time = time.time()
        all_chunks: List[Document] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, f): f
                for f in file_paths
            }
            for future in as_completed(future_to_file):
                chunks, error = future.result()
                if error:
                    stats.failed += 1
                    stats.errors.append(error)
                else:
                    stats.processed += 1
                    stats.total_chunks += len(chunks)
                    all_chunks.extend(chunks)

        stats.processing_time = time.time() - start_time
        return all_chunks, stats
