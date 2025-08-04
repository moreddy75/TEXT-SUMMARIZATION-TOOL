from transformers import pipeline

def summarize_article(text, max_length=130, min_length=30):
    # Load summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Example input: Replace with any long article or paste one
    long_text = """
    Climate change is the defining crisis of our time and it is happening even more quickly than we feared. 
    But we are far from powerless in the face of this global threat. As United Nations Secretary-General António Guterres noted, 
    “The climate emergency is a race we are losing, but it is a race we can win.” Tackling climate change requires 
    unprecedented international cooperation, policy changes, and individual responsibility. The science is clear: 
    greenhouse gas emissions are driving global warming, leading to more frequent and intense weather events, 
    sea level rise, and disruption of ecosystems. Solutions lie in transitioning to renewable energy, protecting forests, 
    transforming agriculture, and rethinking our consumption patterns.
    """

    print("\nOriginal Text:\n")
    print(long_text)
    print("\n---\n")

    summary = summarize_article(long_text)
    print("Summary:\n")
    print(summary)
