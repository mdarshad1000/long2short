from long2short import Long2Short, OpenAIProvider

provider = OpenAIProvider(api_key="your-api-key")
summarizer = Long2Short(provider)

text = """
Enter a Long Text Here
"""

summary = summarizer.summarize(
    text=text,
    detail=0.25,
    verbose=True,
    summarize_recursively=True,
    additional_instructions="Write additional instructions here."
)
print(summary)
