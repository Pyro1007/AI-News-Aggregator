from flask import Flask, render_template, request, jsonify
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
import os
from flask_cors import CORS

# Suppress unnecessary logs
os.environ['WDM_LOG_LEVEL'] = '0'
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Initialize models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
translator_hi = None
translator_mr = None

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

def get_translation_pipeline(target_language):
    model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

def get_actual_url(google_news_url):
    parsed_url = urlparse(google_news_url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get('url', [google_news_url])[0]

def fetch_article_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
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
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    driver = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        driver.get(url)
        time.sleep(3)  # Reduced from 5 seconds
        page_source = driver.page_source
        page_soup = soup(page_source, "html.parser")
        paragraphs = page_soup.find_all("p")
        article_text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        return article_text if len(article_text) > 300 else None
    except Exception as e:
        print(f"Selenium error: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()

def summarize_text(text):
    try:
        input_length = len(text.split())
        max_length = max(40, min(150, int(input_length * 0.5)))
        truncated_text = " ".join(text.split()[:1024])
        summary = summarizer(truncated_text, max_length=max_length, min_length=max(20, int(max_length * 0.5)), do_sample=False)
        return summary[0]["summary_text"].strip() if summary else None
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return None

def translate_text(text, translator):
    try:
        translation = translator(text)
        return translation[0]['translation_text'] if translation else None
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return None

def generate_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_news_item(news, target_date, seen_hashes, need_translation, translation_language):
    global translator_hi, translator_mr
    
    raw_title = news.title.text
    link = news.link.text
    pub_date = news.pubDate.text
    
    if not is_specific_date(pub_date, target_date):
        return None
    
    actual_url = get_actual_url(link)
    news_source = news.source.text if news.source else "Unknown"
    title = re.sub(r"\s*-\s*" + re.escape(news_source) + r"$", "", raw_title)
    
    article_text = fetch_article_text(actual_url) or fetch_article_with_selenium(actual_url)
    if not article_text:
        return None
        
    article_hash = generate_hash(article_text)
    if article_hash in seen_hashes:
        return None
    seen_hashes.add(article_hash)
    
    news_summary = summarize_text(article_text)
    translated_summary = None
    
    if news_summary and need_translation:
        try:
            if translation_language == 'hi':
                if not translator_hi:
                    translator_hi = get_translation_pipeline("hi")
                translator = translator_hi
            elif translation_language == 'mr':
                if not translator_mr:
                    translator_mr = get_translation_pipeline("mr")
                translator = translator_mr
            else:
                translator = None
                
            if translator:
                translated_summary = translate_text(news_summary, translator)
        except Exception as e:
            print(f"Translation setup error: {str(e)}")
    
    sentiment = TextBlob(title).sentiment.polarity
    sentiment_label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
    
    return {
        "title": title,
        "source": news_source,
        "published_date": pub_date,
        "url": actual_url,
        "summary": news_summary,
        "translated_summary": translated_summary,
        "sentiment": sentiment_label,
        "translation_language": translation_language if translated_summary else None
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_news', methods=['POST'])
def get_news():
    data = request.json
    category = data.get('category', 'technology')
    target_date_str = data.get('date', datetime.now().strftime("%Y-%m-%d"))
    need_translation = data.get('translation', False)
    translation_language = data.get('translation_language', 'hi')
    
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    
    site = get_rss_feed(category)
    try:
        op = urlopen(site)
        rd = op.read()
        op.close()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch RSS feed: {str(e)}"}), 500
    
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    
    seen_hashes = set()
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for news in news_list:
            futures.append(executor.submit(
                process_news_item, 
                news, 
                target_date, 
                seen_hashes, 
                need_translation, 
                translation_language
            ))
        
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    
    return jsonify({"news": results})

if __name__ == '__main__':
    app.run(debug=True)
