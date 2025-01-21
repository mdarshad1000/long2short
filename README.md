# long2short

**long2short** is a flexible Python library for long document text summarization that supports multiple Language Model (LLM) providers. It allows you to summarize long documents with fine-grained control over the level of detail. With an extensible architecture, it’s easy to integrate with various LLMs and customize its behavior.

---

## Features

- **Multi-LLM Support**: Compatible with OpenAI, Anthropic, and custom LLM providers.
- **Detail Control**: Adjust the level of detail in the summary with a simple parameter.
- **Smart Chunking**: Automatically splits and processes large texts based on token limits.
- **Recursive Summarization**: Uses previous summaries as context for summarizing subsequent sections.
- **Custom Instructions**: Add domain-specific instructions for tailored summarization.
- **Progress Tracking**: Visualize progress with `tqdm`.
- **Extensible Design**: Add new LLM providers or customize existing ones with ease.

---

## Installation

Install the library using pip:

```bash
pip install long2short
```

---

## Quick Start

Here’s how to get started with **long2short** using OpenAI as the LLM provider:

```python
from long2short import Long2Short, OpenAIProvider

# Initialize the provider
provider = OpenAIProvider(api_key="your-api-key")
summarizer = Long2Short(provider)

# Summarize text
text = "Your long text here..."
summary = summarizer.summarize(text, detail=0.5)
print(summary)
```

---

## Using Different Providers

### OpenAI

To use OpenAI’s GPT models:

```python
from long2short import Long2Short, OpenAIProvider

provider = OpenAIProvider(
    api_key="your-openai-api-key",
    model="gpt-4-turbo"  # Specify your preferred model
)
summarizer = Long2Short(provider)
```

### Anthropic (Claude)

To use Anthropic’s Claude models:

```python
from long2short import Long2Short, AnthropicProvider

provider = AnthropicProvider(
    api_key="your-anthropic-api-key",
    model="claude-3-opus-20240229"  # Specify your preferred model
)
summarizer = Long2Short(provider)
```

---

## Controlling Summary Detail

The `detail` parameter allows you to adjust how detailed the summary should be:

```python
# Generate a brief, high-level summary
brief_summary = summarizer.summarize(text, detail=0)

# Generate a detailed, in-depth summary
detailed_summary = summarizer.summarize(text, detail=1)
```

---

## Advanced Features

### Recursive Summarization

Enable recursive summarization to use previous summaries as context for generating new ones:

```python
summary = summarizer.summarize(
    text,
    detail=0.5,
    summarize_recursively=True
)
```

### Custom Instructions

Tailor the summary with additional instructions:

```python
summary = summarizer.summarize(
    text,
    detail=0.5,
    additional_instructions="Focus on numerical data and statistics."
)
```

### Smart Text Chunking

Large texts are automatically split into manageable chunks based on token limits, ensuring efficient processing. You can control:

- Minimum chunk size (`minimum_chunk_size`)
- Chunk delimiters (`chunk_delimiter`)
- Headers for each chunk (`header`)

Example:

```python
summary = summarizer.summarize(
    text,
    detail=0.7,
    minimum_chunk_size=500,
    chunk_delimiter=".",
    header="Section Summary"
)
```

### Verbose Output

Enable detailed logging to track the summarization process:

```python
summary = summarizer.summarize(
    text,
    detail=0.5,
    verbose=True
)
```

### Handling Dropped Chunks

The library ensures that excessively large chunks are skipped, and any dropped chunks are logged (if verbose mode is enabled). This prevents token overflow issues while maintaining efficient processing.

---

## Creating Custom Providers

You can implement custom LLM providers by extending the `LLMProvider` abstract base class:

```python
from long2short import LLMProvider

class CustomProvider(LLMProvider):
    def __init__(self, **kwargs):
        # Initialize your provider
        pass

    def generate_completion(self, messages: list, **kwargs) -> str:
        # Implement completion generation logic
        return "Custom completion response"
```

Integrate the custom provider into Long2Short:

```python
custom_provider = CustomProvider()
summarizer = Long2Short(custom_provider)
```

---

## Progress Tracking

The summarization process supports `tqdm` for real-time progress tracking:

```python
summary = summarizer.summarize(
    text,
    detail=0.5,
    verbose=True
)
```

---

## Extensibility

### Adding New Features
- Extend functionality by overriding or extending the `Long2Short` class.
- Customize tokenization or chunking behavior by modifying `Tokenizer` or `TextChunker` classes.

---

## Contributing

Contributions are welcome! Whether it’s reporting a bug, suggesting new features, or submitting a pull request, your help is appreciated.

To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## Example Usage

```python
from long2short import Long2Short, OpenAIProvider

# Initialize with OpenAI
provider = OpenAIProvider(api_key="your-api-key")
summarizer = Long2Short(provider)

# Summarize with custom instructions
text = "Your long document here..."
summary = summarizer.summarize(
    text,
    detail=0.8,
    additional_instructions="Focus on the key takeaways and technical details."
)

print("Summary:")
print(summary)
```

### Attribution
This project heavily references code and ideas from the [OpenAI Cookbook](https://cookbook.openai.com/examples/summarizing_long_documents).
