# base_processor.py - Framework
from abc import ABC, abstractmethod
from typing import List, Dict
import os
import re
from pathlib import Path
import pdfplumber
import PyPDF2
import pytesseract
import chromadb
import fitz
import hashlib
from datetime import datetime
from pix2text import Pix2Text
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class BaseAcademicProcessor:
    """
    Abstract base class for academic document processing
    Handles common academic document operations, delegates domain-specific logic
    """
    
    def __init__(self, domain_name: str, base_path: str):
        self.domain_name = domain_name
        self.base_path = setup_paths(base_path)
        self.vector_db = setup_vector_database(domain_name)
        self.embedding_model = self.get_embedding_model()  # Abstract method - expected in form (tokeniser, model)
        self.chunk_config = None
    
    # ===== ABSTRACT METHODS (must be implemented by subclasses) =====
    
    @abstractmethod
    def get_embedding_model(self) -> (AutoTokenizer, AutoModel):
        """Return domain-specific embedding model"""
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
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """Universal PDF text extraction with academic optimizations"""
        
        p2t = Pix2Text() #Initialise OCR for image processing

        extracted_content = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_data = {
                    "page" : page_num + 1,
                    "text" : "",
                    "tables" : [],
                    "images" : [] 
                }

                text = page.extract_text()
                if text:
                    page_data["text"] = text

                tables = page.extract_table()
                if tables:
                    page_data["tables"] = tables

                extracted_content.append(page_data) # Add page to list - use enumerate to index later

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                images = page.get_images()

                for img_index, img in enumerate(images):
                    cross_ref = img[0]
                    pixel_map = fitz.Pixmap(doc, cross_ref)

                    if pixel_map.n - pixel_map.alpha > 3: #Convert CMYK images into RGB for consistency
                        pixel_map = fitz.Pixmap(fitz.csRGB, pixel_map)

                    img_data = pixel_map.tobytes("png") # Convert to png

                    ocr_img = p2t.recognize(img_data)

                    extracted_content[page_num]["images"].append({ # Add images their numers to each page's dictionary
                        "index": img_index,
                        "ocr_text": ocr_img})
                    
        return extracted_content    

    
    def extract_base_metadata(self, text_content: List[Dict], file_path: str) -> dict:
        # File hash, page count, processing date
        # Basic title extraction (first meaningful line)
        # Basic year extraction (regex patterns)
        # Document type detection (paper vs textbook heuristics)
        
        # File hash for duplication checking/tracking
        with open(file_path, "rb") as f: # read binary data
            file_bytes = f.read() # for hash
            pdf2_reader = PyPDF2.PdfReader(f) # for metadata
        file_hash = hashlib.sha256(file_bytes).hexdigest() # hash data and convert to str

        metadata = pdf2_reader.metadata
        
        page_count = len(text_content)
        
        process_date = datetime.now().utctime().isoformat() # Use universal time (same as GMT)
        
        if metadata and "/Title" in metadata: # Try metadata
            title_guess = metadata["/Title"]
            if title_guess.type() == str:
                title = title_guess # Only assign if title is correct type
            
        if not title: # fallback method - might work for papers probably need improving
            first_page_text = text_content[0]["text"].splitlines() if text_content and text_content[0]["text"] else []
            title_guess = ""
            for line in first_page_text:
                if line.strip() and len(line.strip()) > 5: 
                    title_guess = line.strip()
                    break
            title = title_guess

        if metadata and "/CreationDate" in metadata:
            try:
                creation_date = datetime(metadata["CreationDate"]).utctime().isoformat()
            except TypeError:
                pass

        if not creation_date:# Year guess (regex over first 2 pages)
            combined_text = " ".join([p["text"] for p in text_content[:2] if p["text"]])
            year_match = re.search(r"(19|20)\d{2}", combined_text)
            creation_date = year_match.group(0) if year_match else None

        # Document type heuristic
        doc_type = "paper"
        if any("chapter" in (p["text"] or "").lower() for p in text_content[:3]):
            doc_type = "book"
        elif "abstract" in combined_text.lower():
            doc_type = "paper"

        base_metadata = {
            "file_hash": file_hash,
            "page_count": page_count,
            "process_date": process_date,
            "title": title,
            "year": creation_date,
            "doc_type": doc_type
        }
        
        return base_metadata

    
    def smart_chunk_text(self, text_content: List[Dict], metadata: Dict) -> List[Chunk]:
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
    
    def chunk_section_by_sentences(self, section: Dict) -> List[Dict]:
        """Universal sentence-based chunking (same across domains)"""
        #Assume section comes like {"content", "title", "type", "page"} Might change later
        
        chunks = []
        
        # Use regex to split sentences and clean 
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])' 
        sentences = re.split(sentence_pattern, section["content"])
        sentences = [s.strip() for s in sentences if s.strip()] 

        if not sentences: # Error handling
            return []
        
        current_chunk_text = ""
        current_word_count = 0

        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())

            if current_word_count + sentence_words > self.chunk_config["max_words"] and current_chunk_text:
                toggle_new_chunk = 1

            if toggle_new_chunk:
                chunk = {
                    "content": current_chunk_text.strip(),
                    "metadata": {
                        "section_title": section["title"],
                        "section_type": section["type"],
                        "page": section["page"],
                        "word_count": current_word_count,
                        "chunk_index": len(chunks),
                        "source_sentences": f"{i-len(current_chunk_text.split('. '))}:{i}"}
                }
                chunks.append(chunk)
    
    def generate_embeddings(self, chunks: List[Chunk], batch_size = 8) -> List[Vector]:
        """Generate embeddings using domain-specific model"""
        texts = [chunk.content for chunk in chunks]
        tokenizer, model = self.embedding_model # Unpack tokeniser and model from tuple
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch,return_tensors="pt", padding=True, truncation=True, max_length=512) # tokenise batch returnings as torch tensors

            with torch.no_grad():
                outputs = model(**inputs)
            
            batch_embeddings = outputs.last_hidden_state.mean(dim = 1).cpu().numpy() # convert shape of last hidden state to [batch_size, hidden_dim] before storing
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
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
            
            # 5. Universal + Domain-specific: Smart chunking
            chunks = self.smart_chunk_text(extracted_text, base_metadata)
            
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
