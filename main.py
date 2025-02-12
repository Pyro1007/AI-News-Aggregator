import torch
import asyncio
import aiohttp
from bs4 import BeautifulSoup as soup
from urllib.parse import urlparse, parse_qs
import re
import nltk
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from textblob import TextBlob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib
from deep_translator import GoogleTranslator
from transformers import pipeline

# Suppress unnecessary logs
logging.getLogger("selenium").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.ERROR)
nltk.data.path.append("nltk_data")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Load Summarizer Model Efficiently
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def get_rss_feed(category):
    return f"https://news.google.com/rss/search?q={category}"

def parse_date(date_string):
    try:
        return datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")
    except ValueError:
        return None

def is_specific_date(pub_date, target_date):
    pub_datetime = parse_date(pub_date)
    return pub_datetime and pub_datetime.date() == target_date.date()

def get_actual_url(google_news_url):
    parsed_url = urlparse(google_news_url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get("url", [google_news_url])[0]

def is_valid_paragraph(text):
    """Filter out cookie notices, disclaimers, and very short paragraphs."""
    invalid_keywords = ["cookie", "advertising", "privacy policy", "terms of service", "agreement"]
    return len(text.split()) > 10 and not any(keyword in text.lower() for keyword in invalid_keywords)

async def fetch_article_text(url, session):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=8) as response:
            if response.status != 200:
                return None
            page_soup = soup(await response.text(), "html.parser")

            # Extract meaningful content (filtering out unwanted sections)
            article_body = page_soup.find("article")
            paragraphs = article_body.find_all("p") if article_body else page_soup.find_all("p")

            article_text = "\n".join([p.get_text() for p in paragraphs if is_valid_paragraph(p.get_text())])
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
        time.sleep(3)
        page_source = driver.page_source
        driver.quit()
        page_soup = soup(page_source, "html.parser")

        article_body = page_soup.find("article")
        paragraphs = article_body.find_all("p") if article_body else page_soup.find_all("p")

        article_text = "\n".join([p.get_text() for p in paragraphs if is_valid_paragraph(p.get_text())])
        return article_text if len(article_text) > 300 else None
    except Exception:
        driver.quit()
        return None

def summarize_text(text):
    try:
        input_length = len(text.split())
        max_length = min(150, max(40, int(input_length * 0.5)))
        truncated_text = " ".join(text.split()[:1024])
        summary = summarizer(truncated_text, max_length=max_length, min_length=max(20, int(max_length * 0.5)), do_sample=False)
        return summary[0]["summary_text"].strip() if summary else None
    except Exception:
        return None

def translate_text(text, target_language):
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(text)
    except Exception:
        return None

def generate_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def process_news_item(news, target_date, seen_hashes, need_translation, translation_language, session):
    raw_title = news.title.text
    link = news.link.text
    pub_date = news.pubDate.text
    if not is_specific_date(pub_date, target_date):
        return None

    actual_url = get_actual_url(link)
    news_source = news.source.text if news.source else "Unknown"
    title = re.sub(r"\s*-\s*" + re.escape(news_source) + r"$", "", raw_title)

    article_text = await fetch_article_text(actual_url, session) or fetch_article_with_selenium(actual_url)
    if not article_text:
        return None  # Do not print failed extraction message

    article_hash = generate_hash(article_text)
    if article_hash in seen_hashes:
        return None
    seen_hashes.add(article_hash)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(summarize_text, article_text)
        news_summary = future.result()

    if not news_summary:
        return None

    translated_summary = translate_text(news_summary, translation_language) if need_translation else None
    sentiment = TextBlob(title).sentiment.polarity
    sentiment_label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"

    print(f"Title: {title}")
    print(f"Source: {news_source}")
    print(f"Published: {pub_date}")
    print(f'News Link: \033]8;;{actual_url}\033\\Click Here\033]8;;\033\\')
    print(f"Summary: {news_summary}")
    
    if translated_summary:
        print(f"Translated ({translation_language}): {translated_summary}")
    
    print(f"Sentiment: {sentiment_label}\n{'-'*100}")

async def main():
    category = input("Enter news category[world, technology, business, sports, health, science, entertainment]: ")
    target_date = datetime.strptime(input("Enter date (YYYY-MM-DD): "), "%Y-%m-%d")
    need_translation = input("Translate? (yes/no): ").strip().lower() == "yes"
    translation_language = input("Enter language code: ").strip().lower() if need_translation else None

    async with aiohttp.ClientSession() as session:
        async with session.get(get_rss_feed(category)) as response:
            xml_content = await response.text()

    news_list = soup(xml_content, "xml").find_all("item")
    #print(f"Fetched {len(news_list)} articles")

    seen_hashes = set()
    tasks = [process_news_item(news, target_date, seen_hashes, need_translation, translation_language, session) for news in news_list]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
