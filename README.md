# News Summarizer and Sentiment Analyzer

This Python project fetches, summarizes, translates, and analyzes the sentiment of news articles using AI-powered techniques.

## Features
- Fetches news articles from Google News RSS based on user-specified categories.
- Extracts article content using `aiohttp` (async requests) and `selenium` (as a fallback).
- Summarizes articles using Facebook's `BART` transformer model.
- Provides sentiment analysis on news titles.
- Translates summaries into user-specified languages using `deep_translator`.
- Uses multi-threading for faster execution.

## Technologies Used
- Python
- Selenium
- BeautifulSoup
- Aiohttp
- Transformers (`BART` model for summarization)
- Deep Translator
- TextBlob (Sentiment Analysis)
- NLTK (Natural Language Processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/news-summarizer.git
   cd news-summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script using:
```bash
python main.py
```

Follow the prompts to enter your preferred category and date.

### Example Input:
```
Enter news category[world, technology, business, sports, health, science, entertainment]: technology
Enter date (YYYY-MM-DD): 2025-03-28
Translate? (yes/no): yes
Enter language code: fr
```

### Example Output:
```
Title: Apple Announces New AI Chip for MacBooks
Source: CNN
Published: Fri, 28 Mar 2025 12:30:00 GMT
News Link: Click Here
Summary: Apple unveils its latest AI chip for MacBooks, promising improved performance and efficiency...
Translated (fr): Apple dévoile sa nouvelle puce AI pour MacBooks...
Sentiment: Positive
```

## Notes
- Ensure you have `ChromeDriver` installed for Selenium.
- Make sure `torch` is installed with GPU support if available for better performance.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to fork this project and submit pull requests!

