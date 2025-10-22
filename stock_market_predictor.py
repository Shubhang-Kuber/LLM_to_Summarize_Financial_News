
# FinBERT + Live Stock Sentiment Pipeline (Daily Auto-Update)
!pip install transformers yfinance requests schedule tqdm -q
#transformers: loads FinBERT model
#yfinance: gets stock data
#requests: fetches news
#schedule: automates daily updates
#tqdm: adds progress bars
import requests
import pandas as pd
import yfinance as yf
import datetime as dt
import torch
import schedule
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# Load FinBERT (Finance-specific BERT model)
#tokenizer converts the text into tokens which helps the model to read
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Parameters
#API_Key is used to fetch the articles 
API_KEY = "a907970697ea4a6585cca194bbb1330c"   
TICKER = "TCS.NS"                # Stock symbol
DAYS_BACK = 10                  # Days of news to fetch


# Function: Fetch latest stock data like open, close, low, high, volume
def fetch_stock_data(ticker=TICKER):
    end = dt.datetime.now()
    start = end - dt.timedelta(days=DAYS_BACK)
    stock_df = yf.download(ticker, start=start, end=end)
    stock_df.reset_index(inplace=True)
    stock_df["Date"] = stock_df["Date"].astype(str).str[:10]
    return stock_df


# Function: Fetch latest financial news
def fetch_news():
  #Request the articles from the (x) no of  days back
    from_date = (dt.datetime.now() - dt.timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q=stock market OR {TICKER}&from={from_date}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    
    news_df = pd.DataFrame([{
        #Going to extract the date, time and the Description 
        "Date": a["publishedAt"][:10],
        "Title": a["title"],
        "Description": a["description"]
    } for a in articles if a["description"]])
    
    return news_df


# Function: FinBERT sentiment scoring
def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[torch.argmax(probs)]
    score = probs.max().item()
    return sentiment, score

# ------------------------------------------------------------
# Function: Analyze sentiment and merge with stock data
# ------------------------------------------------------------

# ============================================================
# 🧠 FINAL FIXED analyze_sentiment() — Handles MultiIndex Columns
# ============================================================

def analyze_sentiment():
    import datetime as dt
    from tqdm import tqdm
    import torch
    import pandas as pd
    import yfinance as yf
    import requests

    print(f"\n[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching new data...")

    # ------------------------------------------------------------
    # 1️⃣ Fetch latest stock data
    # ------------------------------------------------------------
    end = dt.datetime.now()
    start = end - dt.timedelta(days=DAYS_BACK)
    stock_df = yf.download(TICKER, start=start, end=end)

    # ---- Handle MultiIndex for both index and columns ----
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = ['_'.join(col).strip() for col in stock_df.columns.values]

    stock_df.reset_index(inplace=True)
    stock_df.rename(columns={"Date": "Date"}, inplace=True)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------
    # 2️⃣ Fetch financial news
    # ------------------------------------------------------------
    from_date = (dt.datetime.now() - dt.timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q=stock market OR {TICKER}&from={from_date}&language=en&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    articles = response.json().get("articles", [])

    news_df = pd.DataFrame([{
        "Date": a["publishedAt"][:10],
        "Title": a["title"],
        "Description": a["description"]
    } for a in articles if a["description"]])

    if news_df.empty:
        print("⚠️ No news articles found for this period.")
        return

    # ------------------------------------------------------------
    # 3️⃣ FinBERT Sentiment Analysis
    # ------------------------------------------------------------
    sentiments, scores = [], []
    for desc in tqdm(news_df["Description"], desc="Analyzing Sentiment"):
        try:
            inputs = tokenizer(desc, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = ["negative", "neutral", "positive"]
            sentiment = labels[torch.argmax(probs)]
            score = probs.max().item()
            sentiments.append(sentiment)
            scores.append(score)
        except Exception as e:
            sentiments.append("neutral")
            scores.append(0.0)

    news_df["Sentiment_Label"] = sentiments
    news_df["Sentiment_Score"] = scores
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    news_df["Sentiment"] = news_df["Sentiment_Label"].map(mapping)

    sentiment_df = news_df.groupby("Date", as_index=False)["Sentiment"].mean()
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------
    # 4️⃣ Safe Merge (Handles MultiIndex columns + clean Date)
    # ------------------------------------------------------------
    stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.strftime("%Y-%m-%d")
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.strftime("%Y-%m-%d")

    merged = pd.merge(stock_df, sentiment_df, on="Date", how="left")
    merged["Sentiment"].fillna(0, inplace=True)

    # ------------------------------------------------------------
    # 5️⃣ Save + Display
    # ------------------------------------------------------------
    merged.to_csv("daily_stock_sentiment.csv", index=False)
    print("✅ Data successfully updated and saved as 'daily_stock_sentiment.csv'!")

    display(merged.tail())


# ------------------------------------------------------------
# Run immediately + schedule for daily updates
# ------------------------------------------------------------
analyze_sentiment()

# Optional: Run automatically every day at 9 AM
#schedule.every().day.at("09:00").do(analyze_sentiment)

# Uncomment this loop if you want continuous daily refresh:
# while True:
#     schedule.run_pending()
#     time.sleep(60)
