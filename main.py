from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from urllib.parse import urlparse, parse_qs
import requests
import re
import nltk
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from transformers import pipeline, MarianMTModel, MarianTokenizer
from textblob import TextBlob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib

nltk.download('punkt')

def get_rss_feed(category):
    return f'https://news.google.com/rss/search?q={category}'

def parse_date(date_string):
    try:
        return datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")
    except ValueError:
        return None

def is_specific_date(pub_date, target_date):
    pub_datetime = parse_date(pub_date)
    return pub_datetime and pub_datetime.date() == target_date.date()

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)

def get_translation_pipeline(target_language):
    model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

translator_hi = get_translation_pipeline("hi")
translator_mr = get_translation_pipeline("mr")

def get_actual_url(google_news_url):
    parsed_url = urlparse(google_news_url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get('url', [google_news_url])[0]

def fetch_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        page_soup = soup(response.text, "html.parser")
        paragraphs = page_soup.find_all("p")
        article_text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        return article_text if len(article_text) > 300 else None
    except Exception:
        return None

def fetch_article_with_selenium(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(url)
        time.sleep(5)
        page_source = driver.page_source
        driver.quit()
        page_soup = soup(page_source, "html.parser")
        paragraphs = page_soup.find_all("p")
        article_text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        return article_text if len(article_text) > 300 else None
    except Exception:
        driver.quit()
        return None

def summarize_text(text):
    try:
        input_length = len(text.split())
        max_length = max(40, min(150, int(input_length * 0.5)))
        truncated_text = " ".join(text.split()[:1024])
        summary = summarizer(truncated_text, max_length=max_length, min_length=max(20, int(max_length * 0.5)), do_sample=False)
        return summary[0]["summary_text"].strip() if summary else None
    except Exception:
        return None

def translate_text(text, translator):
    try:
        translation = translator(text)
        return translation[0]['translation_text'] if translation else None
    except Exception:
        return None

def generate_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_news_item(news, target_date, seen_hashes, need_translation, translation_language):
    raw_title = news.title.text
    link = news.link.text
    pub_date = news.pubDate.text
    
    if not is_specific_date(pub_date, target_date):
        return
    
    actual_url = get_actual_url(link)
    news_source = news.source.text if news.source else "Unknown"
    title = re.sub(r"\s*-\s*" + re.escape(news_source) + r"$", "", raw_title)
    
    article_text = fetch_article_text(actual_url) or fetch_article_with_selenium(actual_url)
    if article_text:
        article_hash = generate_hash(article_text)
        if article_hash in seen_hashes:
            return
        seen_hashes.add(article_hash)
        
        news_summary = summarize_text(article_text)
        translated_summary = None
        
        if news_summary and need_translation:
            translator = translator_hi if translation_language == 'hi' else translator_mr
            translated_summary = translate_text(news_summary, translator)
        
        sentiment = TextBlob(title).sentiment.polarity
        sentiment_label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        
        print(f"Title: {title}")
        print(f"Source: {news_source}")
        print(f"Published Date: {pub_date}")
        print(f'News Link: \033]8;;{actual_url}\033\\Click Here\033]8;;\033\\')
        print(f"News Summary: {news_summary}")
        if translated_summary:
            print(f"Translated Summary ({translation_language}): {translated_summary}")
        print(f"Sentiment: {sentiment_label}")
        print("-" * 100)

category = input("Enter the news category (e.g., politics, sports, technology): ")
target_date_str = input("Enter the date to fetch news for (YYYY-MM-DD): ")
target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

need_translation = input("Do you need translation? (yes/no): ").strip().lower() == "yes"
translation_language = None
if need_translation:
    translation_language = input("Enter the translation language (hi for Hindi, mr for Marathi): ").strip().lower()
    if translation_language not in ['hi', 'mr']:
        print("Invalid language choice. Translation will be skipped.")
        need_translation = False

site = get_rss_feed(category)
op = urlopen(site)
rd = op.read()
op.close()
sp_page = soup(rd, 'xml')
news_list = sp_page.find_all('item')

seen_hashes = set()
with ThreadPoolExecutor(max_workers=5) as executor:
    for news in news_list:
        executor.submit(process_news_item, news, target_date, seen_hashes, need_translation, translation_language)

