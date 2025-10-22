from transformers import pipeline
import torch

def device_id():
    # use GPU if available, else CPU
    return 0 if torch.cuda.is_available() else -1

def main():
    # pick a summarization model (smaller + faster than bart-large)
    model_name = "sshleifer/distilbart-cnn-12-6"

    summarizer = pipeline("summarization", model=model_name, device=device_id())

    # 👉 Paste any article / text you want summarized here
    ARTICLE = """    """

    summary = summarizer(
        ARTICLE,
        max_length=120,   # cap on output size (tokens)
        min_length=25,    # minimum size (tokens)
        do_sample=False   # deterministic (no random sampling)
    )

    print("INPUT:\n", ARTICLE.strip())
    print("\nSUMMARY:\n", summary[0]["summary_text"])

if __name__ == "__main__":
    main()
