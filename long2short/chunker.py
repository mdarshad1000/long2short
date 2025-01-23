# long2short/chunker.py
from typing import List, Tuple, Optional
from .tokenizer import Tokenizer


class TextChunker:
    """Handles splitting text into manageable chunks."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def chunk_text(
        self,
        text: str,
        max_tokens: int,
        delimiter: str = ".",
        header: Optional[str] = None,
        add_ellipsis: bool = True,
    ) -> Tuple[List[str], List[List[int]], int]:
        """
        Split text into chunks based on token limit.

        Args:
            text: Text to split into chunks
            max_tokens: Maximum tokens per chunk
            delimiter: Delimiter to split text on
            header: Optional header to prepend to each chunk
            add_ellipsis: Whether to add ellipsis for overflow chunks

        Returns:
            Tuple of (chunks, chunk_indices, dropped_chunk_count)
        """
        chunks = text.split(delimiter)
        combined_chunks, indices, dropped = self._combine_chunks(
            chunks=chunks,
            max_tokens=max_tokens,
            chunk_delimiter=delimiter,
            header=header,
            add_ellipsis=add_ellipsis,
        )
        if dropped > 0:
            print(f"warning: {dropped} chunks were dropped due to overflow")

        # Add back delimiters
        combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
        return combined_chunks, indices, dropped

    def _combine_chunks(
        self,
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter: str = "\n\n",
        header: Optional[str] = None,
        add_ellipsis: bool = True,
    ) -> Tuple[List[str], List[List[int]], int]:
        """
        Combine text chunks while respecting token limits.

        Args:
            chunks: List of text chunks to combine
            max_tokens: Maximum tokens per combined chunk
            chunk_delimiter: Delimiter to join chunks with
            header: Optional header to prepend to each chunk
            add_ellipsis: Whether to add ellipsis for overflow chunks
        """
        dropped_chunk_count = 0
        output = []
        output_indices = []
        candidate = [] if header is None else [header]

        candidate_indices = []
        for chunk_i, chunk in enumerate(chunks):
            chunk_with_header = [chunk] if header is None else [header, chunk]

            # Check if single chunk exceeds token limit
            if (len(self.tokenizer.tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens):
                print(f"Warning: chunk overflow")
                if add_ellipsis and len(self.tokenizer.tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens:
                    candidate.append("...")
                    dropped_chunk_count += 1
                continue  # this case would break downstream assumptions

            # estimate token count with the current chunk added
            extended_candidate_token_count = len(self.tokenizer.tokenize(chunk_delimiter.join(candidate + [chunk])))
            # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
            if extended_candidate_token_count > max_tokens:
                output.append(chunk_delimiter.join(candidate))
                output_indices.append(candidate_indices)
                candidate = chunk_with_header  # re-initialize candidate
                candidate_indices = [chunk_i]
            # otherwise keep extending the candidate
            else:
                candidate.append(chunk)
                candidate_indices.append(chunk_i)
        # add the remaining candidate to output if it's not empty
        if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
        return output, output_indices, dropped_chunk_count