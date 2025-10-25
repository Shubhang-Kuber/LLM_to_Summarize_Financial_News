# 🔥 LLM to Summarize Financial News

An AI-driven project that scrapes financial news, uses NLP models for summarization and entity extraction, and automates structured data storage in Google Sheets for analysis.

## 🌟 Features

**AI-Powered Summarization**: Transformer-based model (BART) that generates concise summaries of complex financial news articles.  
- **Automated Web Scraping**: Extracts news data from sites like Times of India and Business Today using BeautifulSoup and Requests.  
- **Named Entity Recognition (NER)**: Detects and categorizes entities such as companies, organizations, and financial figures using spaCy.  
- **Structured Data Output**: Stores processed summaries and entities in organized tabular format using pandas.  
- **Google Sheets Sync**: Automatically updates summaries and key data points to Google Sheets through gspread API.  
- **Local CSV Export**: Enables offline access and data archival through CSV generation.  
- **Step-wise Modular Pipeline**: Clear stages from article extraction to summarization and cloud storage for transparency and debugging.  
- **Cloud-Optimized Execution**: Seamlessly deployable on Google Colab with GPU support for faster inference.  
- **Scalable & Extensible**: Easily extendable for sentiment analysis, trend prediction, or investment-focused data insights.
  
  ## 🏗️ Output Images

![Web-Scrapping Image](https://github.com/Shubhang-Kuber/LLM_to_Summarize_Financial_News/blob/main/Images/Screenshot%202025-10-25%20214304.png)
![Stock-Market-Predictor Image](https://github.com/Shubhang-Kuber/LLM_to_Summarize_Financial_News/blob/main/Images/Screenshot%202025-10-25%20214502.png)

### **Web Scraping Module (Data Extraction)**
- **Purpose**: Collects raw financial news data from online sources.  
- **Libraries Used**: BeautifulSoup and Requests for HTML parsing and data retrieval.  
- **Data Sources**: Financial websites such as Times of India and Business Today.  
- **Functionality**: Identifies article titles, extracts body text, and filters relevant financial content.  
- **Output**: Cleaned text data passed to the AI Summarization module for further processing.

---

### **AI Summarization & Google Sheets Module (Data Processing & Storage)**
- **Summarization Model**: Transformer-based `facebook/bart-large-cnn` for generating concise financial summaries.  
- **Entity Extraction**: Uses spaCy to identify organizations, people, and key financial entities.  
- **Data Structuring**: Organizes results into pandas DataFrame with timestamps, URLs, summaries, and entity data.  
- **Cloud Integration**: Automatically uploads processed summaries to Google Sheets using gspread API.    
- **Execution Platform**: Fully compatible with Google Colab, supporting GPU acceleration for faster inference.

---

### **System Workflow**
1. Extracts financial news articles using the **Web Scraping Module**.  
2. Passes article content to the **AI Summarization Module** for processing.  
3. Generates summaries and extracts financial entities.  
4. Stores structured results both locally (CSV) and on Google Sheets.  
5. Enables future integration for trend detection, sentiment analysis, or investment prediction.

---

## 📋 Software & System Requirements

### Web Scraping Module
| Component | Library/Tool | Purpose |
|------------|--------------|----------|
| BeautifulSoup4 | `bs4` | Parses HTML and extracts financial news content |
| Requests | `requests` | Sends HTTP requests to fetch web pages |
| lxml | `lxml` | Fast and efficient HTML parsing backend |
| re | `re` | Cleans and formats extracted text using regular expressions |
| pandas | `pandas` | Structures data for analysis and export |

---

### AI Summarization & NLP Module
| Component | Library/Tool | Purpose |
|------------|--------------|----------|
| Transformers | `facebook/bart-large-cnn` | Generates concise, context-aware news summaries |
| spaCy | `en_core_web_sm` | Performs Named Entity Recognition (NER) on news articles |
| torch | `PyTorch` | Provides backend for transformer model execution |
| json | `json` | Converts extracted entities into structured JSON format |
| logging | `logging` | Tracks summarization errors and runtime information |

---

### Google Sheets Integration Module
| Component | Library/Tool | Purpose |
|------------|--------------|----------|
| gspread | `gspread` | Connects Python to Google Sheets API for reading and writing data. |
| google-auth | `google.auth` | Handles OAuth 2.0 authentication between Colab and Google account. |
| oauth2client | `oauth2client` | Manages credential authorization for Google APIs. |
| gspread-dataframe | `gspread_dataframe` | Converts pandas DataFrame into a Google Sheet format for seamless upload. |
| google.colab.auth | `auth.authenticate_user()` | Authenticates Colab environment to allow secure access to Google Sheets. |
| pandas | `pandas` | Prepares summarized financial data for transfer and formatting into spreadsheet cells. |
| set_with_dataframe() | Function | Writes the processed DataFrame into a Google Sheet in tabular structure. |
| open_by_url() | Method | Opens the Google Sheet using its unique URL for targeted updates. |
| worksheet.sheet1 | Object | Refers to the first worksheet where summarized financial news data is stored. |

---

### Common Dependencies
- Python 3.9 or higher  
- Google Colab or Jupyter Notebook  
- Active internet connection (for fetching articles and using APIs)  
- Google account (for Google Sheets integration)  
- CSV-compatible viewer (Excel, LibreOffice, etc.)  
## 🔧 Pin Configuration



---

**Made with ❤️ for safer environments**
