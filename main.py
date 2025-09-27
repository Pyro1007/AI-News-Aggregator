from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen, Request
from urllib.parse import urlparse, parse_qs
import requests
import re
import nltk
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from transformers import pipeline, MarianMTModel, MarianTokenizer
from textblob import TextBlob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from flask_cors import CORS
import logging
import traceback

# Suppress unnecessary logs
os.environ['WDM_LOG_LEVEL'] = '0'
logging.getLogger('transformers').setLevel(logging.ERROR)

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download nltk data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)

# Initialize models (lazy loading to avoid startup delays)
summarizer = None
translator_hi = None
translator_mr = None

def init_summarizer():
    global summarizer
    if summarizer is None:
        try:
            logger.info("Loading summarizer model...")
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True)
            logger.info("Summarizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load summarizer: {str(e)}")
            summarizer = False  # Mark as failed
    return summarizer

def get_rss_feed(category):
    # Multiple RSS feed URLs to try
    feeds = [
        f'https://news.google.com/rss/search?q={requests.utils.quote(category)}&hl=en-US&gl=US&ceid=US:en',
        f'https://news.google.com/rss/topics/{get_topic_code(category)}?hl=en-US&gl=US&ceid=US:en',
        'https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en'  # Fallback to top stories
    ]
    return feeds

def get_topic_code(category):
    # Google News topic codes
    topics = {
        'technology': 'CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB',
        'business': 'CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB',
        'sports': 'CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB',
        'entertainment': 'CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB',
        'health': 'CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ',
        'science': 'CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB',
        'world': 'CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB'
    }
    return topics.get(category.lower(), topics['technology'])

def parse_date(date_string):
    try:
        # Google News typically uses GMT
        date_string = date_string.replace(' GMT', '').replace(' +0000', '')
        
        formats = [
            "%a, %d %b %Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        return None
    except Exception as e:
        logger.error(f"Date parsing error: {str(e)} for date: {date_string}")
        return None

def is_within_date_range(pub_date, target_date, days_range=1):
    """Check if publication date is within range of target date"""
    pub_datetime = parse_date(pub_date)
    if not pub_datetime:
        return True  # Include if we can't parse date
    
    # Allow for some flexibility in date matching
    date_diff = abs((pub_datetime.date() - target_date.date()).days)
    return date_diff <= days_range

def get_translation_pipeline(target_language):
    try:
        model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return pipeline("translation", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.error(f"Translation model error for {target_language}: {str(e)}")
        return None

def get_actual_url(google_news_url):
    try:
        # Google News URLs often have the actual URL as a parameter
        if 'news.google.com' in google_news_url:
            parsed_url = urlparse(google_news_url)
            query_params = parse_qs(parsed_url.query)
            if 'url' in query_params:
                return query_params['url'][0]
        return google_news_url
    except:
        return google_news_url

def fetch_article_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {url}")
            return None
        
        page_soup = soup(response.content, "html.parser")
        
        # Remove unwanted elements
        for element in page_soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
            element.decompose()
        
        article_content = ""
        
        # Try multiple selectors
        selectors = [
            'article',
            '[role="main"]',
            '.story-body',
            '.article-body',
            '.story-content',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content-body',
            'main',
            '.main-content'
        ]
        
        for selector in selectors:
            if selector.startswith('.') or selector.startswith('['):
                elements = page_soup.select(selector)
            else:
                elements = page_soup.find_all(selector)
            
            if elements:
                for element in elements:
                    paragraphs = element.find_all(['p', 'div'], recursive=True)
                    text_parts = []
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 30 and not any(skip in text.lower() for skip in ['cookie', 'subscribe', 'newsletter', 'advertisement']):
                            text_parts.append(text)
                    
                    if text_parts:
                        article_content = " ".join(text_parts)
                        break
            
            if article_content and len(article_content) > 200:
                break
        
        # Fallback: get all paragraphs
        if not article_content or len(article_content) < 200:
            all_paragraphs = page_soup.find_all('p')
            text_parts = []
            for p in all_paragraphs:
                text = p.get_text().strip()
                if len(text) > 50:
                    text_parts.append(text)
            article_content = " ".join(text_parts[:20])  # Limit to first 20 paragraphs
        
        return article_content if len(article_content) > 100 else None
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None

def fetch_article_with_selenium(url):
    """Fallback method using Selenium for JavaScript-heavy sites"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        driver.set_page_load_timeout(15)
        driver.get(url)
        time.sleep(3)  # Wait for JavaScript to load
        
        # Scroll to load lazy content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(1)
        
        page_source = driver.page_source
        page_soup = soup(page_source, "html.parser")
        
        # Remove unwanted elements
        for element in page_soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        paragraphs = page_soup.find_all("p")
        text_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50]
        article_text = " ".join(text_parts[:20])  # Limit paragraphs
        
        return article_text if len(article_text) > 100 else None
        
    except Exception as e:
        logger.error(f"Selenium error for {url}: {str(e)}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

def summarize_text(text):
    summarizer_model = init_summarizer()
    if not summarizer_model or summarizer_model is False:
        # Return first 200 characters as fallback
        return text[:200] + "..." if len(text) > 200 else text
    
    try:
        # Clean and limit text
        text = ' '.join(text.split())  # Normalize whitespace
        words = text.split()
        
        if len(words) < 50:
            return text  # Too short to summarize
        
        if len(words) > 1000:
            text = " ".join(words[:1000])
        
        max_length = min(150, max(50, len(words) // 3))
        min_length = min(30, max(20, len(words) // 6))
        
        summary = summarizer_model(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"].strip() if summary else text[:200] + "..."
        
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return text[:200] + "..." if len(text) > 200 else text

def translate_text(text, translator):
    if not translator:
        return None
    
    try:
        # Handle long texts by chunking
        max_chunk_size = 400
        words = text.split()
        
        if len(words) <= max_chunk_size:
            translation = translator(text)
            return translation[0]['translation_text'] if translation else None
        
        # Split into sentences and chunk
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip() + "."
            if len(current_chunk.split()) + len(sentence.split()) < max_chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        translated_chunks = []
        for chunk in chunks[:3]:  # Limit chunks to avoid timeout
            if chunk:
                translation = translator(chunk)
                if translation:
                    translated_chunks.append(translation[0]['translation_text'])
        
        return " ".join(translated_chunks) if translated_chunks else None
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return None

def generate_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def process_news_item(news, target_date, seen_hashes, need_translation, translation_language):
    global translator_hi, translator_mr
    
    try:
        # Extract basic information
        raw_title = news.title.text if news.title else "No title"
        link = news.link.text if news.link else ""
        pub_date = news.pubDate.text if news.pubDate else ""
        
        if not link:
            return None
        
        # Check date range (be more flexible)
        if pub_date and not is_within_date_range(pub_date, target_date, days_range=2):
            logger.info(f"Skipping article from {pub_date} (target: {target_date})")
            return None
        
        actual_url = get_actual_url(link)
        
        # Extract source
        news_source = "Unknown"
        if hasattr(news, 'source') and news.source:
            news_source = news.source.text
        elif 'source' in news.find_all():
            news_source = news.find('source').text
        else:
            # Try to extract from URL
            try:
                domain = urlparse(actual_url).netloc
                news_source = domain.replace('www.', '').split('.')[0].title()
            except:
                pass
        
        # Clean title
        title = re.sub(r"\s*-\s*" + re.escape(news_source) + r"$", "", raw_title)
        
        # Check for duplicate titles
        title_hash = generate_hash(title)
        if title_hash in seen_hashes:
            logger.info(f"Skipping duplicate: {title[:50]}")
            return None
        seen_hashes.add(title_hash)
        
        # Try to fetch article content
        article_text = fetch_article_text(actual_url)
        
        # Use Selenium as fallback for JS-heavy sites
        if not article_text or len(article_text) < 100:
            logger.info(f"Trying Selenium for {actual_url[:50]}")
            article_text = fetch_article_with_selenium(actual_url)
        
        # If still no content, use description from RSS
        if not article_text:
            if hasattr(news, 'description') and news.description:
                article_text = news.description.text
                logger.info(f"Using RSS description for {title[:50]}")
            else:
                logger.warning(f"No content found for {title[:50]}")
                return None
        
        # Generate summary
        news_summary = summarize_text(article_text)
        translated_summary = None
        
        # Translation if requested
        if news_summary and need_translation and translation_language:
            try:
                if translation_language == 'hi':
                    if not translator_hi:
                        logger.info("Loading Hindi translator...")
                        translator_hi = get_translation_pipeline("hi")
                    translator = translator_hi
                elif translation_language == 'mr':
                    if not translator_mr:
                        logger.info("Loading Marathi translator...")
                        translator_mr = get_translation_pipeline("mr")
                    translator = translator_mr
                else:
                    translator = None
                
                if translator:
                    translated_summary = translate_text(news_summary, translator)
            except Exception as e:
                logger.error(f"Translation setup error: {str(e)}")
        
        # Sentiment analysis
        try:
            blob = TextBlob(title + " " + (news_summary[:200] if news_summary else ""))
            sentiment = blob.sentiment.polarity
            sentiment_label = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
        except:
            sentiment_label = "neutral"
        
        result = {
            "title": title[:200],  # Limit title length
            "source": news_source,
            "published_date": pub_date,
            "url": actual_url,
            "summary": news_summary[:500] if news_summary else "Content not available",
            "translated_summary": translated_summary[:500] if translated_summary else None,
            "sentiment": sentiment_label,
            "translation_language": translation_language if translated_summary else None
        }
        
        logger.info(f"Successfully processed: {title[:50]}")
        return result
    
    except Exception as e:
        logger.error(f"Error processing news item: {str(e)}\n{traceback.format_exc()}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_news', methods=['POST', 'GET'])
def get_news():
    try:
        # Get parameters
        if request.method == 'GET':
            data = request.args
        else:
            data = request.json or {}
        
        category = data.get('category', 'technology').lower()
        target_date_str = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        need_translation = data.get('translation', False) in [True, 'true', 'True', '1']
        translation_language = data.get('translation_language', 'hi')
        
        logger.info(f"Request: category={category}, date={target_date_str}, translation={need_translation}")
        
        # Parse target date
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError:
            target_date = datetime.now()
            logger.warning(f"Invalid date format, using today: {target_date}")
        
        # Try multiple RSS feeds
        rss_feeds = get_rss_feed(category)
        news_list = []
        
        for site in rss_feeds:
            try:
                logger.info(f"Trying RSS feed: {site[:100]}")
                req = Request(site, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                op = urlopen(req, timeout=10)
                rd = op.read()
                op.close()
                
                sp_page = soup(rd, 'xml')
                items = sp_page.find_all('item')
                
                if items:
                    news_list = items
                    logger.info(f"Found {len(items)} news items")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to fetch RSS feed: {str(e)}")
                continue
        
        if not news_list:
            # Try fallback to general news
            try:
                logger.info("Trying fallback to general news feed")
                fallback_url = 'https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en'
                req = Request(fallback_url, headers={'User-Agent': 'Mozilla/5.0'})
                op = urlopen(req, timeout=10)
                rd = op.read()
                op.close()
                sp_page = soup(rd, 'xml')
                news_list = sp_page.find_all('item')
                logger.info(f"Fallback found {len(news_list)} items")
            except:
                pass
        
        if not news_list:
            logger.error("No news items found from any source")
            return jsonify({
                "error": "Unable to fetch news from Google News. The service might be temporarily unavailable.",
                "news": [],
                "debug": "No RSS items found"
            }), 404
        
        seen_hashes = set()
        results = []
        
        # Process news items
        items_to_process = min(len(news_list), 15)  # Process up to 15 items
        logger.info(f"Processing {items_to_process} news items")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for news in news_list[:items_to_process]:
                futures.append(executor.submit(
                    process_news_item,
                    news,
                    target_date,
                    seen_hashes,
                    need_translation,
                    translation_language
                ))
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                        if len(results) >= 10:  # Stop after 10 successful results
                            break
                except Exception as e:
                    logger.error(f"Future execution error: {str(e)}")
                    continue
        
        if not results:
            # Try to provide at least basic information from RSS
            basic_results = []
            for news in news_list[:5]:
                try:
                    title = news.title.text if news.title else "No title"
                    link = news.link.text if news.link else ""
                    pub_date = news.pubDate.text if news.pubDate else ""
                    description = news.description.text if hasattr(news, 'description') and news.description else ""
                    
                    if title and link:
                        basic_results.append({
                            "title": title,
                            "source": "Google News",
                            "published_date": pub_date,
                            "url": get_actual_url(link),
                            "summary": description[:200] + "..." if description else "Click to read more",
                            "translated_summary": None,
                            "sentiment": "neutral",
                            "translation_language": None
                        })
                except:
                    continue
            
            if basic_results:
                return jsonify({
                    "news": basic_results,
                    "count": len(basic_results),
                    "note": "Basic results only - full content extraction failed"
                })
            
            return jsonify({
                "error": "Could not process any news articles. This might be due to rate limiting or connection issues.",
                "news": [],
                "suggestion": "Try again in a few moments or with a different category"
            })
        
        logger.info(f"Successfully returning {len(results)} news items")
        return jsonify({
            "news": results,
            "count": len(results),
            "category": category,
            "date": target_date_str
        })
    
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "news": []
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/test')
def test():
    """Test endpoint to verify the app is working"""
    return jsonify({
        "message": "News aggregator is running",
        "available_categories": ["technology", "business", "sports", "entertainment", "health", "science", "world"],
        "endpoints": {
            "/": "Home page",
            "/get_news": "Get news (POST/GET)",
            "/health": "Health check",
            "/test": "This test endpoint"
        }
    })

if __name__ == '__main__':
    logger.info("Starting Flask News Aggregator...")
    app.run(debug=True, host='0.0.0.0', port=5000)
