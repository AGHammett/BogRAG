# base_processor.py - Framework
from abc import ABC, abstractmethod
from typing import List, Dict
import os
from pathlib import Path
import pdfplumber
import PyPDF2
import pytesseract
import chromadb

class BaseAcademicProcessor:
    """
    Abstract base class for academic document processing
    Handles common academic document operations, delegates domain-specific logic
    """
    
    def __init__(self, domain_name: str, base_path: str):
        self.domain_name = domain_name
        self.base_path = setup_paths(base_path)
        self.vector_db = setup_vector_database(domain_name)
        self.embedding_model = self.get_embedding_model()  # Abstract method
    
    # ===== ABSTRACT METHODS (must be implemented by subclasses) =====
    
    @abstractmethod
    def get_embedding_model(self) -> EmbeddingModel:
        """Return domain-specific embedding model"""
        pass
    
    @abstractmethod
    def extract_domain_metadata(self, text_content: List[Dict]) -> DomainMetadata:
        """Extract domain-specific metadata (authors, citations, etc.)"""
        pass
    
    @abstractmethod
    def identify_domain_sections(self, text: str) -> List[Section]:
        """Identify domain-specific document sections"""
        pass
    
    @abstractmethod
    def classify_chunk_type(self, content: str) -> str:
        """Classify content type (theorem/definition/paragraph for math, etc.)"""
        pass
    
    # ===== CONCRETE METHODS (shared across all domains) =====
    
    def extract_text_from_pdf(self, file_path: str) -> ExtractedText:
        """Universal PDF text extraction with academic optimizations"""
        # Use pdfplumber for tables/equations
        # Fallback to PyPDF2
        # Handle OCR for scanned documents
        # Return structured text + basic metadata
    
    def extract_base_metadata(self, text_content: List[Dict], file_path: str) -> BaseMetadata:
        """Extract universal academic metadata"""
        # File hash, page count, processing date
        # Basic title extraction (first meaningful line)
        # Basic year extraction (regex patterns)
        # Document type detection (paper vs textbook heuristics)
    
    def smart_chunk_text(self, text_content: List[Dict], metadata: DocumentMetadata) -> List[Chunk]:
        """Orchestrate the chunking process"""
        chunks = []
        for page in text_content:
            # 1. Call abstract method: identify_domain_sections()
            sections = self.identify_domain_sections(page.content)
            
            for section in sections:
                # 2. Universal sentence-level chunking
                section_chunks = self.chunk_section_by_sentences(section)
                
                # 3. Call abstract method: classify_chunk_type() for each chunk
                for chunk in section_chunks:
                    chunk.type = self.classify_chunk_type(chunk.content)
                    chunks.append(chunk)
        
        return chunks
    
    def chunk_section_by_sentences(self, section: Section) -> List[Chunk]:
        """Universal sentence-based chunking (same across domains)"""
        # Split into sentences
        # Group into ~400 word chunks
        # Preserve sentence boundaries
        # Return list of chunks with section context
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[Vector]:
        """Generate embeddings using domain-specific model"""
        texts = [chunk.content for chunk in chunks]
        return self.embedding_model.encode(texts)
    
    def store_chunks(self, chunks: List[Chunk], embeddings: List[Vector]):
        """Store chunks in vector database"""
        # Prepare metadata for ChromaDB
        # Add to domain-specific collection
        # Log storage results
    
    # ===== MAIN WORKFLOW METHODS =====
    
    def process_document(self, file_path: str, doc_type: str) -> bool:
        """Main document processing workflow - TEMPLATE METHOD PATTERN"""
        try:
            # 1. Universal: Extract text from PDF
            extracted_text = self.extract_text_from_pdf(file_path)
            
            # 2. Universal: Extract base metadata
            base_metadata = self.extract_base_metadata(extracted_text, file_path)
            
            # 3. Domain-specific: Extract specialized metadata
            domain_metadata = self.extract_domain_metadata(extracted_text)
            
            # 4. Combine metadata
            full_metadata = combine_metadata(base_metadata, domain_metadata)
            
            # 5. Universal + Domain-specific: Smart chunking
            chunks = self.smart_chunk_text(extracted_text, full_metadata)
            
            # 6. Domain-specific: Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # 7. Universal: Store in vector DB
            self.store_chunks(chunks, embeddings)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return False
    
    def process_directory(self, directory_path: str, doc_type: str) -> ProcessingResults:
        """Process all documents in directory"""
        # Find all PDFs
        # Call process_document() for each
        # Aggregate results and logging
    
    # ===== OPTIONAL HOOKS (can be overridden but have defaults) =====
    
    def preprocess_text(self, text: str) -> str:
        """Optional preprocessing hook"""
        return text  # Default: no preprocessing
    
    def postprocess_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optional post-processing hook"""
        return chunks  # Default: no post-processing
    
    def get_chunking_strategy(self) -> ChunkingConfig:
        """Optional: customize chunking parameters"""
        return ChunkingConfig(
            target_words=400,
            max_words=600,
            overlap_sentences=1
        )
    
def setup_paths(base_path: str) -> Path:
    """Helper to create RAG directory structure"""
    base = Path(base_path).expanduser()
    #Define necessary directory structure 
    required_dirs = [ # Update here if need to add new databases!!!!
        'data/economics_db/papers',
        'data/economics_db/textbooks', 
        'data/mathematics_db/papers',
        'data/mathematics_db/textbooks',
        'data/raw_documents',
        'models/embeddings',
        'logs'
    ]

    # Check all exist, create if not
    for dir_path in required_dirs:
        (base / dir_path).mkdir(parents = True, exist_ok = True)

    return base

def setup_vector_database(domain_name: str, base_path: Path) -> dict:
    """Setup ChromaDB collections for a specific domain"""
    # Create chroma client on disk
    client = chromadb.PersistentClient(path=str(base_path / 'data'))
    # Each domain gets 2 collections: papers and textbooks
    collections = {
        'papers': client.get_or_create_collection(
            name=f"{domain_name}_papers"
        ),
        'textbooks': client.get_or_create_collection(
            name=f"{domain_name}_textbooks"
        )
    }
    
    return collections  # Returns dict of collections for this domain
