# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `requirements.txt` file for dependency installation.
- Datasets preprocessing scripts for huggingface medical datasets.
- Integrated Chroma vector database for efficient document retrieval
- Implemented HuggingFaceEmbeddings with the "sentence-transformers/all-MiniLM-L6-v2" model for improved text representation
- Created comprehensive web scraper for Malaysian tax information acquisition
- Added support for table extraction and conversion to markdown format
- Implemented intelligent document chunking using RecursiveCharacterTextSplitter
- Added capability to combine multiple document sources into a unified knowledge base

### Fixed

- Resolved timeout issues in web scraping by implementing proper error handling
- Fixed embedding generation for documents with special characters
- Corrected metadata handling for PDF content extraction
- Addressed issues with duplicate content in web scraping results
- Proper scripting in `finetune/medical.py` file.

### Changed

- Improved chunking logic with optimized chunk size (800 chars) and overlap (100 chars)
- Enhanced table extraction logic for better handling of complex table structures
- Migrated from basic text splitter to tiktoken-based RecursiveCharacterTextSplitter
- Upgraded document metadata schema for better source tracking
- `datasets` package dependency in `requirements.txt`.

### Removed

- Deprecated direct string-based document chunking
- Removed dependency on older embedding models
- Eliminated redundant content preprocessing steps
- Discontinued support for legacy vector storage formats
- Unnecessary packages in `requirements.txt`.
