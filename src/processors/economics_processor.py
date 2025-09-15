from base_processor import BaseAcademicProcessor, Chunk
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

class EconomicsProcessor(BaseAcademicProcessor):

    def __init__(self, domain_name: str, base_path: str):
        super().__init__(domain_name, base_path)

    def get_embedding_model(self) -> Tuple[AutoTokenizer, AutoModel]: 
        """Fills abstract method to return chosen models
        Economcis Model to use FinBERT model - specialised to financial and economics text"""
        
        tokeniser = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModel.from_pretrained("ProsusAI/finbert")
        
        return (tokeniser, model)
    
    def generate_embeddings(self, chunks: List[Chunk], batch_size = 8) -> List[Vector]:
        """Generate embeddings using domain-specific model"""
        texts = [chunk["content"] for chunk in chunks]
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
    
