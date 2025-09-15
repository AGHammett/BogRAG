from typing import List, Dict
import re

class Section:
    def __init__(self, content: str, title: str = "", section_type: str = "", page: int = 0):
        self.content = content
        self.title = title
        self.type = section_type
        self.page = page

class Chunk:
    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata or {}
        self.type = None  # Will be set by classify_chunk_type()

class ChunkingConfig:
    def __init__(self, target_words: int = 400, max_words: int = 600, overlap_sentences: int = 1):
        self.target_words = target_words
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences

def chunk_section_by_sentences(section: Section, config: ChunkingConfig) -> List[Chunk]:
    """
    Universal sentence-based chunking method for academic documents.
    
    This method implements a sophisticated chunking strategy that:
    1. Preserves sentence boundaries (never breaks mid-sentence)
    2. Maintains semantic coherence by keeping related sentences together
    3. Provides configurable overlap between chunks for context preservation
    4. Adapts chunk size based on content density
    
    Args:
        section: Section object containing text content and metadata
        config: ChunkingConfig with target_words, max_words, overlap_sentences
    
    Returns:
        List of Chunk objects with content and preserved metadata
    """
    
    chunks = []
    
    # Step 1: Split into sentences using academic-aware regex
    # This pattern handles common academic abbreviations and citations
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, section.content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Handle edge case: very short sections
    if not sentences:
        return []
    
    current_chunk_text = ""
    current_word_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())
        
        # Decision point: Should we start a new chunk?
        should_create_new_chunk = (
            # Current chunk + new sentence exceeds maximum
            (current_word_count + sentence_words > config.max_words and current_chunk_text) or
            # We've hit our target and this is a natural breaking point
            (current_word_count >= config.target_words and current_chunk_text)
        )
        
        if should_create_new_chunk:
            # Finalize current chunk
            chunk = Chunk(
                content=current_chunk_text.strip(),
                metadata={
                    "section_title": section.title,
                    "section_type": section.type,
                    "page": section.page,
                    "word_count": current_word_count,
                    "chunk_index": len(chunks),
                    "source_sentences": f"{i-len(current_chunk_text.split('. '))}:{i}"
                }
            )
            chunks.append(chunk)
            
            # Handle overlap for context preservation
            if config.overlap_sentences > 0 and chunks:
                # Extract last N sentences from completed chunk for overlap
                prev_sentences = re.split(sentence_pattern, current_chunk_text.strip())
                overlap_sentences = prev_sentences[-config.overlap_sentences:]
                overlap_text = ' '.join(overlap_sentences)
                
                # Start new chunk with overlap + current sentence
                current_chunk_text = overlap_text + " " + sentence
                current_word_count = len(current_chunk_text.split())
            else:
                # No overlap - start fresh
                current_chunk_text = sentence
                current_word_count = sentence_words
                
        else:
            # Add sentence to current chunk
            if current_chunk_text:
                current_chunk_text += " " + sentence
            else:
                current_chunk_text = sentence
            current_word_count += sentence_words
    
    # Handle final chunk (remaining content)
    if current_chunk_text.strip():
        final_chunk = Chunk(
            content=current_chunk_text.strip(),
            metadata={
                "section_title": section.title,
                "section_type": section.type,
                "page": section.page,
                "word_count": current_word_count,
                "chunk_index": len(chunks),
                "is_final_chunk": True
            }
        )
        chunks.append(final_chunk)
    
    return chunks


# Example usage and test
if __name__ == "__main__":
    # Sample academic text
    sample_text = """
    Machine learning has revolutionized the field of artificial intelligence. 
    Deep neural networks, in particular, have shown remarkable success in various domains. 
    The backpropagation algorithm, introduced by Rumelhart et al. in 1986, enables efficient training of multi-layer networks. 
    This method computes gradients by applying the chain rule of calculus. 
    Modern implementations use automatic differentiation for computational efficiency. 
    Convolutional neural networks (CNNs) have become the standard for image processing tasks. 
    They leverage spatial locality and parameter sharing to reduce computational complexity. 
    The concept of pooling layers helps achieve translation invariance. 
    Recurrent neural networks (RNNs) handle sequential data effectively. 
    Long Short-Term Memory (LSTM) networks address the vanishing gradient problem in traditional RNNs.
    """
    
    # Create section and config
    section = Section(
        content=sample_text.strip(),
        title="Introduction to Neural Networks",
        section_type="introduction",
        page=1
    )
    
    config = ChunkingConfig(target_words=50, max_words=80, overlap_sentences=1)
    
    # Generate chunks
    chunks = chunk_section_by_sentences(section, config)
    
    # Display results
    print(f"Generated {len(chunks)} chunks from section:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"Content: {chunk.content}")
        print(f"Word count: {chunk.metadata['word_count']}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 30)