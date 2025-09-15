from base_processor import BaseAcademicProcessor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

class EconomicsProcessor(BaseAcademicProcessor):

    def get_embedding_model(self):

        # Economcis Model to use FinBERT model - specialised to financial and economics text
        tokeniser = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModel.from_pretrained("ProsusAI/finbert")
        return (tokeniser, model)
    
    
