# long2short/models.py
from abc import ABC, abstractmethod
from typing import List, Dict
import os

from openai import OpenAI
from anthropic import Anthropic

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion from the LLM provider."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', 0),
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-opus-20240229"):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": f"Summarize the following text:\n\n{prompt}"}],
            temperature=kwargs.get('temperature', 0),
        )
        return response.content[0].text