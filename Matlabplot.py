#Step-1
# Colab: install required packages (will install latest compatible versions)
# Note: this may take 1-3 minutes.
!pip install -q -U spacy transformers torch sentencepiece pandas plotly matplotlib beautifulsoup4 gspread oauth2client openpyxl

# Download spaCy small English model
!python -m spacy download en_core_web_sm


#Step-2
import re
import math
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.matcher import Matcher
import pandas as pd
from transformers import pipeline
import plotly.express as px
import matplotlib.pyplot as plt
from google.colab import files
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials

# load spaCy model
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

#Step-3
def fetch_article_text(url, timeout=15):
    """Fetch a URL and return cleaned text. Basic, robust HTML trimming."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside"]):
        tag.decompose()
    # prefer <article> if present
    article = soup.find("article")
    text = article.get_text(separator=" ") if article else soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Step-4
# small regex-based numeric normalizer for money, percents, EPS
def parse_money_string(s):
    """Return a float value in absolute units using common suffixes (k, m, bn)."""
    s = s.replace(',', '').strip()
    # find first number + optional suffix
    m = re.search(r'(-?\d+(?:\.\d+)?)(?:\s)?([kKmMbB]|bn|million|billion|thousand|k)?', s)
    if not m:
        return None
    val = float(m.group(1))
    suf = (m.group(2) or '').lower()
    if suf in ('m', 'million'):
        val *= 1e6
    elif suf in ('b', 'bn', 'billion'):
        val *= 1e9
    elif suf in ('k', 'thousand'):
        val *= 1e3
    return val

def find_money_in_text(s):
    """Return list of money-like substrings (e.g., '$1.2 billion', '₹ 200 crore', '120 million')."""
    # capture currency symbol + number + optional suffix words
    patterns = [
        r'[\$₹€]\s?\d{1,3}(?:[0-9,\.]*)\s?(?:million|billion|bn|m|k|thousand)?',
        r'\d{1,3}(?:[0-9,\,\.]*)\s?(?:million|billion|crore|lakh|bn|m|k|thousand)'
    ]
    results = []
    for p in patterns:
        for m in re.finditer(p, s, flags=re.I):
            results.append(m.group(0))
    return results

def parse_percent(s):
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*%', s)
    return float(m.group(1)) if m else None

#Step-5
# EPS detection: look for "EPS" or "earnings per share" then a nearby number
eps_pattern = [
    [{"LOWER": {"IN": ["eps", "eps:"]}}],
    # allow an optional punctuation and number
]
# simpler: we'll use regex later; keep matcher for generic patterns if needed
matcher.add("EPS_PATTERN", [[{"LOWER":"eps"}]])

# Quarter token (Q1, Q2, Q3, Q4) via regex matching in sentence processing (we'll use python regex on text)


#Step-6
def extract_financial_facts(text):
    """
    Returns:
      entities_df: dataframe of simple extracted facts (entity_type, span_text, numeric_value, unit, context_sentence)
      quarter_table: pandas DataFrame with columns ['Quarter', 'Revenue', 'EPS'] if found
    """
    doc = nlp(text)
    rows = []
    # collect spaCy entities: ORG, MONEY, PERCENT, DATE
    for ent in doc.ents:
        if ent.label_ in ("ORG", "MONEY", "PERCENT", "DATE", "CARDINAL", "QUANTITY"):
            rows.append({
                "entity_type": ent.label_,
                "span": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "sentence": ent.sent.text
            })

    # match EPS via regex on whole text and per-sentence
    eps_matches = []
    for sent in doc.sents:
        s = sent.text
        # direct EPS patterns
        em = re.search(r'(?:EPS|Earnings per share|earnings-per-share)[\s:\-]*([-\d\.]+)', s, flags=re.I)
        if em:
            val = float(em.group(1))
            eps_matches.append({"span": em.group(0), "value": val, "sentence": s})
        else:
            # sometimes written like "EPS was $1.25" or "diluted EPS of Rs. 3.5"
            em2 = re.search(r'(?:EPS|earnings per share).{0,30}?([\$₹]?\s?-?\d[\d,\.]*(?:\s?(?:million|billion|m|bn|k))?)', s, flags=re.I)
            if em2:
                parsed = parse_money_string(em2.group(1))
                eps_matches.append({"span": em2.group(0), "value": parsed, "sentence": s})

    # quarter-based revenue extraction: scan sentences that mention Q1/Q2/Q3/Q4 or 'quarter ended' + month
    quarter_map = {}
    month_to_q = {
        'jan':'Q1','feb':'Q1','mar':'Q1','apr':'Q2','may':'Q2','jun':'Q2',
        'jul':'Q3','aug':'Q3','sep':'Q3','sept':'Q3','oct':'Q4','nov':'Q4','dec':'Q4'
    }
    for sent in doc.sents:
        s = sent.text
        Qm = re.search(r'\b(Q[1-4])\b', s, flags=re.I)
        q_label = None
        if Qm:
            q_label = Qm.group(1).upper()
        else:
            m = re.search(r'quarter (?:ended|ending|to)\s*(?:on\s*)?([A-Za-z]{3,9})', s, flags=re.I)
            if m:
                month = m.group(1)[:3].lower()
                q_label = month_to_q.get(month)
        # if this sentence looks like a quarter sentence, extract first money
        if q_label:
            monies = find_money_in_text(s)
            if monies:
                val = parse_money_string(monies[0])
                quarter_map.setdefault(q_label, {})['Revenue'] = val
                quarter_map[q_label].setdefault('Context', []).append(s)
        else:
            # fallback: company-level revenue mentions without quarter label -> ignore for quarter table
            pass

    # Build entities_df and quarter_table
    entities_df = pd.DataFrame(rows)
    # include EPS matches into entities_df
    for em in eps_matches:
        entities_df = pd.concat([entities_df, pd.DataFrame([{"entity_type":"EPS","span":em["span"], "numeric": em["value"], "sentence":em["sentence"]}])], ignore_index=True)

    # Build quarter_table (sorted Q1->Q4)
    q_order = ['Q1','Q2','Q3','Q4']
    qt_rows = []
    for q in q_order:
        entry = quarter_map.get(q, {})
        qt_rows.append({"Quarter": q, "Revenue": entry.get('Revenue', None)})
    quarter_table = pd.DataFrame(qt_rows)

    return entities_df, quarter_table


#Step-7
# prepare the summarizer pipeline (this will download the model)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # default summarizer model. :contentReference[oaicite:3]{index=3}

def plot_and_caption(df_quarter, company_name="Company", metric='Revenue'):
    # drop rows with no metric
    df_plot = df_quarter.dropna(subset=[metric]).reset_index(drop=True)
    if df_plot.empty:
        print("No time-series numeric data found for plotting.")
        return None, None

    # Plotly interactive
    fig = px.line(df_plot, x='Quarter', y=metric, markers=True,
                  title=f"{company_name} — {metric} by Quarter")
    fig.update_traces(mode='lines+markers')
    fig.show()

    # Matplotlib static (in case you want a PNG)
    plt.figure(figsize=(8,4))
    plt.plot(df_plot['Quarter'], df_plot[metric], marker='o')
    plt.title(f"{company_name} — {metric} by Quarter")
    plt.xlabel('Quarter'); plt.ylabel(metric)
    plt.grid(True)
    plt.show()

    # simple textual highlights to feed to summarizer
    values = df_plot[metric].tolist()
    quarters = df_plot['Quarter'].tolist()
    raw = f"{company_name} reported {metric} of " + ", ".join([f"{q}: {v:,.0f}" for q,v in zip(quarters, values)]) + "."
    if len(values) >= 2 and values[0] != 0:
        pct = (values[-1] - values[0]) / abs(values[0]) * 100
        raw += f" This is a {pct:.1f}% change from {quarters[0]} to {quarters[-1]}."
    # refine with summarizer (short caption)
    # keep input concise, summarizer does best with a short but informative prompt
    prompt = raw
    s = summarizer(prompt, max_length=60, min_length=15)[0]['summary_text']
    return fig, s


#Step-8
# Put your URL here (or swap to a manual text variable)
ARTICLE_URL = "https://timesofindia.indiatimes.com/business/india-business/household-fin-assets-growth-at-8-yr-high/articleshow/124132274.cms?utm_source=chatgpt.com"  # <- replace with a real article URL

# Option A: fetch directly (uncomment to fetch real article)
text = fetch_article_text(ARTICLE_URL)

# Option B: paste text manually (useful for paywalled content)
#text = """
#The technology giant reported on Thursday that quarterly earnings exceeded analyst expectations, driven by strong demand for cloud services and AI products. Stock prices rose modestly after the announcement, while investors expressed caution about rising operational costs. Meanwhile, competitors in the sector are expected to release their own earnings reports later this week, which could influence broader market sentiment.
#"""

entities_df, quarter_table = extract_financial_facts(text)
print("Entities (sample):")
display(entities_df.head())

print("\nQuarter table:")
display(quarter_table)

fig, caption = plot_and_caption(quarter_table, company_name="Acme Corp", metric="Revenue")
print("\nAI-generated caption:")
print(caption)



#Step-9
# Example: supply the company name here
company_name = "AI Company"

# Build the values list with an extra title row
title_row = [[f"Company: {company_name}"]]
header_row = [quarter_table.columns.values.tolist()]
data_rows = quarter_table.fillna("").values.tolist()

values = title_row + header_row + data_rows

# Clear the sheet first (optional, keeps things tidy)
ws.clear()

# Update with title + table
ws.update(values)

print(f"✅ Wrote {company_name} quarter table to Google Sheet:", SHEET_URL)