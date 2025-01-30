from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
import requests
import re
import nltk
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from transformers import pipeline
from textblob import TextBlob
# Download necessary NLP model
nltk.download('punkt')

# Define RSS feed URL for political news
site = 'https://news.google.com/rss/search?q=politics'

# Open and read the RSS feed
op = urlopen(site)
rd = op.read()
op.close()

# Parse XML content
sp_page = soup(rd, 'xml')
news_list = sp_page.find_all('item')

# Initialize BERT summarization pipeline
summarizer = pipeline("summarization")

# Function to resolve actual article URL from Google News
def get_actual_url(google_news_url):
    match = re.search(r'url=(.*?)&', google_news_url)
    if match:
        return match.group(1)  # Extracted actual URL
    try:
        response = requests.get(google_news_url, allow_redirects=True)
        return response.url  # Resolved URL
    except Exception as e:
        print(f"Could not resolve URL: {e}")
        return google_news_url

# Function to fetch article content using requests and BeautifulSoup
def fetch_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        page_soup = soup(response.text, "html.parser")

        # Extract readable text from common article containers
        paragraphs = page_soup.find_all("p")
        article_text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])

        return article_text if len(article_text) > 300 else None  # Ensure enough content
    except Exception as e:
        print("Could not fetch article text:", str(e))
        return None

# Function to extract article text using Selenium (for JavaScript-heavy sites)
def fetch_article_with_selenium(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(url)
        time.sleep(5)  # Allow page to load
        page_source = driver.page_source
        driver.quit()
        page_soup = soup(page_source, "html.parser")

        # Extract readable text
        paragraphs = page_soup.find_all("p")
        article_text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])

        return article_text if len(article_text) > 300 else None
    except Exception as e:
        print("Could not fetch article via Selenium:", str(e))
        driver.quit()
        return None

# Function to summarize text using BERT
def summarize_text(text, max_length=250):
    try:
        summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print("Could not summarize:", str(e))
        return None

# Loop through each news item and extract details
for news in news_list:
    title = news.title.text
    link = news.link.text
    pub_date = news.pubDate.text

    # Extract and resolve the actual article URL
    actual_url = get_actual_url(link)

    print(f"Title: {title}")
    print(f"News Link: {actual_url}")

    # Fetch article text
    article_text = fetch_article_text(actual_url)

    # If failed, try Selenium
    if not article_text:
        print("Trying Selenium to fetch content...")
        article_text = fetch_article_with_selenium(actual_url)

    # Summarize if content is available
    if article_text:
        news_summary = summarize_text(article_text)
        print(f"News Summary: {news_summary}")
    else:
        print("Could not extract summary.")

    print(f"Published Date: {pub_date}")
    print("-" * 60)  # Separator line
    #sentiment analysis
    analysis = TextBlob(news.text)
    print(f'sentiment: {"positive" if analysis.polarity > 0 else "negetive" if analysis.polarity < 0 else "neutral" }')
