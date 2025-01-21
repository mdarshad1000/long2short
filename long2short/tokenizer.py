# long2short/tokenizer.py
from typing import List
import tiktoken

class Tokenizer:
    """Handles text tokenization for different models."""
    
    def __init__(self, model: str = "gpt-4-turbo"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to tokens."""
        return self.encoding.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text."""
        return self.encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenize(text))