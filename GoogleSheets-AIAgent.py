#Step-1
# Colab cell 1 (version-pinned safe installs)
!pip install -q \
    transformers>=4.30.0,<5.0.0 \
    beautifulsoup4 lxml requests pandas \
    "spacy>=3.0.0,<4.0.0" \
    gspread "gspread-dataframe>=3.2.1,<5.0.0" \
    google-auth oauth2client
# download spaCy English model
!python -m spacy download en_core_web_sm


#Step-2
import transformers, torch
import spacy
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import pandas as pd
print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("spaCy:", spacy.__version__)
print("gspread:", gspread.__version__)
print("gspread_dataframe:", getattr(set_with_dataframe, "__module__", "<unknown> module>"))
print("pandas:", pd.__version__)



Step-3
# Colab cell 2 — imports and helper functions
import requests
from bs4 import BeautifulSoup
import re
import time
import json
import logging
import pandas as pd
# transformers + device check
import torch
from transformers import pipeline
# spaCy (ensure model installed by Cell 1)
import spacy
# load spaCy model safely (downloads if missing)
def _ensure_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")
nlp = _ensure_spacy_model()
# initialize summarizer pipeline (uses GPU if available)
_device = 0 if torch.cuda.is_available() else -1
# Using a reliable model: facebook/bart-large-cnn (good summary quality for news)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=_device)

# --- Scraper: robust fallback for many news pages ---
def fetch_article(url, timeout=15):    
    """
    Returns dict: {'url':..., 'title':..., 'text':...}
    Attempts: article tag -> common div selectors -> fallback to all <p> text.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # title best-effort
    title_tag = soup.find('h1') or soup.title
    title = title_tag.get_text().strip() if title_tag else ""
    selectors = [
        "article",
        "div[itemprop='articleBody']",
        "div[class*='caas-body']",
        "div[class*='article']",
        "div[class*='content']",
        "section[class*='article']",
    ]
    text = ""
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            ps = node.find_all('p')
            if ps:
                text = "\n".join(p.get_text().strip() for p in ps if p.get_text().strip())
                if len(text.split()) > 50:
                    break

    # final fallback: join all paragraph text (may include noisy content)
    if not text:
        ps = soup.find_all('p')
        text = "\n".join(p.get_text().strip() for p in ps if p.get_text().strip())

    return {"url": url, "title": title, "text": text}
# --- spaCy NER extractor ---
def extract_entities(text, top_k_per_label=10):
    """
    Returns a dict mapping entity label -> list of unique entity strings.
    Keeps at most top_k_per_label occurrences (sorted).
    """
    doc = nlp(text)
    ent_map = {}
    for ent in doc.ents:
        ent_map.setdefault(ent.label_, set()).add(ent.text)
    # convert to lists
    return {label: sorted(list(vals))[:top_k_per_label] for label, vals in ent_map.items()}


# --- chunking and summarization (for long articles) ---
def _chunk_text_sentences(text, max_chars=1000):
    # split into sentence-like chunks (simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def summarize_long_text(text, max_chunk_chars=1000, min_length=40, max_length=160):
    """
    Summarizes long text by chunking, summarizing chunks, then doing a final pass.
    Returns a single summary string.
    """
    if not text or len(text.split()) < 10:
        return ""  # nothing meaningful to summarize

    chunks = _chunk_text_sentences(text, max_chars=max_chunk_chars)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        # safe call to the summarizer
        try:
            out = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            chunk_summaries.append(out[0]['summary_text'].strip())
        except Exception as e:
            logging.warning(f"Summarizer failed on chunk {i}: {e}")
            # fallback: use the chunk head
            chunk_summaries.append(chunk[:max_length])

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    # final condensation
    joined = " ".join(chunk_summaries)
    try:
        final = summarizer(joined, max_length=max_length, min_length=min_length, do_sample=False)
        return final[0]['summary_text'].strip()
    except Exception as e:
        logging.warning(f"Final summarization failed: {e}")
        return " ".join(chunk_summaries)


# --- Google Sheets helper (optional) ---
def push_dataframe_to_sheet(df, sheet_url):
    """
    Push a pandas.DataFrame to an existing Google Sheet (first worksheet).
    In Colab it uses your Google account for auth (will prompt).
    - sheet_url: full URL of the Google Sheet you created.
    """
    # lazy import to avoid forcing auth unless user uses sheets
    from google.colab import auth
    auth.authenticate_user()
    import gspread
    from google.auth import default
    from gspread_dataframe import set_with_dataframe

    creds, _ = default()
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(sheet_url)
    worksheet = sh.sheet1
    set_with_dataframe(worksheet, df)
    return True


# --- high-level pipeline: process URLs and return DF ---
def process_urls_to_dataframe(urls, pause_seconds=2):
    """
    Accepts list of article URLs.
    Returns pandas.DataFrame with columns:
    ['fetched_at_utc','source_url','title','summary','entities_json']
    """
    rows = []
    for url in urls:
        try:
            art = fetch_article(url)
        except Exception as e:
            logging.error(f"Failed to fetch {url}: {e}")
            continue

        text = art.get("text", "")
        title = art.get("title", "") or ""
        entities = extract_entities(text)
        summary = summarize_long_text(text)

        rows.append({
            "fetched_at_utc": pd.Timestamp.utcnow(),
            "source_url": url,
            "title": title,
            "summary": summary,
            # keep entities as JSON-string for easy CSV/gsheet storage
            "entities_json": json.dumps(entities, ensure_ascii=False)
        })

        time.sleep(pause_seconds)  # polite pause

    df = pd.DataFrame(rows)
    return df

#Step-4
# Colab cell 3 — run this with your URLs
# Replace with your target articles (example placeholders)
EXAMPLE_URLS = [
    "https://timesofindia.indiatimes.com/business/india-business/household-fin-assets-growth-at-8-yr-high/articleshow/124132274.cms?utm_source=chatgpt.com",
    "https://timesofindia.indiatimes.com/business/india-business/it-growth-lifts-accenture-revenue-rises-7-per-cent-new-bookings-reach-21-31-billion/articleshow/124126024.cms?utm_source=chatgpt.com",
    
    
]

# run pipeline
df = process_urls_to_dataframe(EXAMPLE_URLS)

# show what we produced
if df.empty:
    print("No rows produced — check the URLs or scraper output.")
else:
    display(df)  # Jupyter pretty display

# save to a local CSV in the Colab VM
csv_path = "financial_summaries.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"Saved local CSV to: {csv_path}")

# Optional: mount Drive and copy CSV there for persistence (uncomment to use)
# from google.colab import drive
# drive.mount('/content/drive')
# df.to_csv('/content/drive/MyDrive/financial_summaries.csv', index=False, encoding='utf-8')


#Step-5
# Colab cell 4 — push to Google Sheets (optional)
SHEET_URL = "PASTE_YOUR_GOOGLE_SHEET_URL_HERE"  # REQUIRED if you want Sheets

if 'df' not in globals() or df.empty:
    raise ValueError("No dataframe available. Run Cell 3 first and ensure df has rows.")

# this will prompt you to authenticate in Colab (one-click)
push_dataframe_to_sheet(df, SHEET_URL)
print("Pushed dataframe to Google Sheet.")