# long2short/long2short.py
from typing import Optional
from tqdm import tqdm

from .models import LLMProvider
from .tokenizer import Tokenizer
from .chunker import TextChunker

class Long2Short:
    """Main class for text summarization."""
    
    def __init__(self, 
        llm_provider: LLMProvider,
        tokenizer: Optional[Tokenizer] = None):
        """
        Initialize summarizer with LLM provider and tokenizer.
        
        Args:
            llm_provider: Provider for language model completions
            tokenizer: Optional tokenizer (will create default if None)
        """
        self.llm_provider = llm_provider
        self.tokenizer = tokenizer or Tokenizer()
        self.chunker = TextChunker(self.tokenizer)
    
    def summarize(self,
        text: str,
        detail: float = 0,
        additional_instructions: Optional[str] = None,
        minimum_chunk_size: int = 500,
        chunk_delimiter: str = ".",
        header: Optional[str] = None,
        summarize_recursively: bool = False,
        verbose: bool = False,
        **kwargs) -> str:
        """
        Summarize text with controllable level of detail.
        
        Args:
            text: Input text to summarize
            detail: Level of detail (0 to 1), higher means more detailed summary
            additional_instructions: Extra instructions for the LLM
            minimum_chunk_size: Minimum size for text chunks
            chunk_delimiter: Delimiter for splitting text into chunks
            header: Optional header to prepend to each chunk
            summarize_recursively: Whether to use previous summaries as context
            verbose: Print detailed information about the process
            **kwargs: Additional arguments passed to the LLM provider
        
        Returns:
            Summarized text
        """
        assert 0 <= detail <= 1, "Detail must be between 0 and 1"
        
        # Calculate number of chunks based on detail level
        document_length = len(self.tokenizer.tokenize(text))
        
        # Get initial chunk count to determine max_chunks
        initial_chunks, _, _ = self.chunker.chunk_text(
            text=text,
            max_tokens=minimum_chunk_size,
            delimiter=chunk_delimiter,
            header=header
        )
        max_chunks = len(initial_chunks)
        min_chunks = 1
        
        # Calculate target number of chunks based on detail level
        num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))
        
        # Calculate final chunk size
        chunk_size = max(minimum_chunk_size, document_length // num_chunks)
        
        # Get final chunks with calculated size
        text_chunks, chunk_indices, dropped_chunks = self.chunker.chunk_text(
            text=text,
            max_tokens=chunk_size,
            delimiter=chunk_delimiter,
            header=header
        )
        
        if verbose:
            print(f"Document length: {document_length} tokens")
            print(f"Target chunks: {num_chunks}")
            print(f"Actual chunks: {len(text_chunks)}")
            print(f"Chunk size: {chunk_size} tokens")
            print(f"Dropped chunks: {dropped_chunks}")
            print(f"Chunk lengths: {[self.tokenizer.count_tokens(x) for x in text_chunks]}")
        
        # Build system message
        system_message = "Rewrite this text in summarized form."
        if additional_instructions is not None:
            system_message += f"\n\n{additional_instructions}"
        
        # Generate summaries
        accumulated_summaries = []
        for chunk in tqdm(text_chunks):
            if summarize_recursively and accumulated_summaries:
                context = "\n\n".join(accumulated_summaries)
                prompt = f"Previous summaries:\n\n{context}\n\nText to summarize next:\n\n{chunk}"
            else:
                prompt = chunk

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]

            summary = self.llm_provider.generate_completion(messages, **kwargs)
            accumulated_summaries.append(summary)
        
        return "\n\n".join(accumulated_summaries)