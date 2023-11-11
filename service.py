from flask import Flask, request, jsonify
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import asyncio
from flask_cors import CORS, cross_origin
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
openai_key = os.getenv("OPEN_AI_KEY")
stocknews_key = os.getenv("STOCKNEWS_KEY")
alphaventage_key = os.getenv("ALPHAVENTAGE_KEY")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


openai_client = OpenAI(api_key=openai_key)


def get_ticker_symbol(company_name):
    # currently filtering only for US listed
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={alphaventage_key}'
    try:
        response = requests.get(url)
        data = response.json()
        for entry in data['bestMatches']:
            if entry['4. region'] != 'United States':
                continue
            return entry['1. symbol']
    except Exception as e:
        print(e)
        pass

    return None


def get_sentiment(ticker: str):
    '''
    Bearish:
        A bearish outlook indicates a belief that prices are likely to fall or that the market is in a downtrend.
        Bearish investors expect a decline in the value of assets and may take actions such as selling stocks or shorting securities (betting that their value will decrease).
    Bullish:
        A bullish outlook indicates a belief that prices are likely to rise or that the market is in an uptrend.
        Bullish investors anticipate an increase in the value of assets and may take actions such as buying stocks or holding onto securities with the expectation of future gains.

    x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish
    '''
    def softmax(x):
        return (np.exp(x)/np.exp(x).sum())
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=20&apikey={alphaventage_key}'
    try:
        relevances = []
        sentiment_scores = []
        response = requests.get(url)
        data = response.json()
        for entry in data['feed']:
            for sentiment in entry['ticker_sentiment']:
                if sentiment['ticker'] != ticker:
                    continue
                relevances.append(float(sentiment['relevance_score']))
                sentiment_scores.append(
                    float(sentiment['ticker_sentiment_score']))
        relevances = np.array(relevances)
        sentiment_scores = np.array(sentiment_scores)
        relevances = softmax(relevances)
        final_score = round(np.dot(relevances, sentiment_scores), 4)
        if final_score <= -0.35:
            label = 'Bearish'
        elif final_score <= -0.15:
            label = 'Somewhat-Bearish'
        elif final_score < 0.15:
            label = 'Neutral'
        elif final_score < 0.35:
            label = 'Somewhat-Bullish'
        else:
            label = 'Bullish'
        return {'score': final_score, 'label': label}

    except Exception as e:
        print(e)
        return None


def extract_text_from_html(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for noise in soup(['script', 'style', 'header', 'footer', 'nav', '[role="navigation"]', '.popup', '#popup']):
        noise.decompose()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def get_html_from_article(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HTML content: {e}")
        return None


def get_text_content(url: str) -> str:
    html = get_html_from_article(url)
    if not html:
        return None
    return extract_text_from_html(html)


class Article:
    def __init__(self, url, title, short_content):
        self.url = url
        self.title = title
        self.short_content = short_content
        self.long_content = get_text_content(self.url)
        self.long_content_summary = None

    async def summarize_long_content(self):
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"I have extracted text content from a news article. The news article title is {self.title}. The content is noisy due to parsing. Please summarize the article given this text data I have extracted:\n{self.long_content}\n\nSummary:",
                }
            ],
            model="gpt-4",
            temperature=0,
            max_tokens=1024,
            n=1,
        )
        summary = response.choices[0].message.content
        if summary and len(summary) > 0:
            self.long_content_summary = summary

    def __str__(self):
        return f"Title: {self.title}\nURL: {self.url}\nShort Content: {self.short_content}\nLong Content: {self.long_content}"


def get_news_from_stocknews(ticker: str, items: int = 3):
    news = []
    news_count = 0
    page = 1
    while news_count < items:
        news_batch = get_news_page_from_stocknews(ticker, page=page)
        news.extend(news_batch)
        news_count += len(news_batch)
        page += 1
    return news


def get_news_page_from_stocknews(ticker: str, items: int = 3, page: int = 1) -> list[Article]:
    # for now, only one ticker, parallelization up to platform
    # https://stocknewsapi.com/api/v1?tickers=AMZN&items=50&page=1&token=GET_API_KEY

    articles = []

    url = "https://stocknewsapi.com/api/v1"
    params = {
        'tickers': ticker,
        'items': items,
        'page': page,
        'token': stocknews_key,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()["data"]
        for entry in data:
            # if url contains youtube continue // TODO: add more filtering conditions, the data is messy
            if "youtube" in entry["news_url"]:
                continue
            article = Article(
                url=entry["news_url"], title=entry["title"], short_content=entry["text"])
            if not article.long_content:
                continue
            articles.append(article)
        return articles
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


async def summarize_news(articles):
    tasks = [article.summarize_long_content() for article in articles]
    await asyncio.gather(*tasks)


def fetch_and_summarize_news(ticker, items=3):
    news = get_news_from_stocknews(ticker, items)
    asyncio.run(summarize_news(news))
    return news


# SERVICE

@app.route('/sentiment', methods=['GET'])
@cross_origin()
def get_sentiment_route():
    try:
        ticker = request.args.get('ticker')

        if ticker:
            sentiment = get_sentiment(ticker)
            return jsonify({'sentiment': sentiment})
        else:
            return jsonify({'error': 'Ticker not provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ticker', methods=['GET'])
@cross_origin()
def get_ticker_route():
    try:
        company_name = request.args.get('company_name')

        if company_name:
            ticker = get_ticker_symbol(company_name)
            if ticker is None:
                return jsonify({'error': 'Company name not found'}), 400
            return jsonify({'ticker': ticker})
        else:
            return jsonify({'error': 'Company name not provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/news', methods=['GET'])
@cross_origin()
def get_news_route():
    # This endpoint returns the news as received from the API
    try:
        ticker = request.args.get('ticker')
        count = request.args.get('count')

        if ticker:
            if count is None:
                news_articles = get_news_from_stocknews(ticker)
            else:
                news_articles = get_news_from_stocknews(ticker, int(count))

            response = []
            for item in news_articles:
                response.append({
                    "url": item.url,
                    "title": item.title,
                    "short_content": item.short_content,
                })
            return jsonify({'news': response})
        else:
            return jsonify({'error': 'Ticker not provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


summarize_dict = {}


@app.route('/news/summarized', methods=['GET'])
@cross_origin()
def get_news_summarized_route():
    # This endpoint returns the news as received from the API, summarized by GPT-4
    try:
        ticker = request.args.get('ticker')
        count = request.args.get('count')

        if ticker:
            if summarize_dict.get(ticker) is not None:
                return jsonify({'news': summarize_dict[ticker]})

            if count is None:
                news_articles = fetch_and_summarize_news(ticker)
            else:
                news_articles = fetch_and_summarize_news(ticker, int(count))

            response = []
            for item in news_articles:
                response.append({
                    "url": item.url,
                    "title": item.title,
                    "summary": item.long_content_summary,
                })

            summarize_dict[ticker] = response

            return jsonify({'news': response})
        else:
            return jsonify({'error': 'Ticker not provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/timeseries/daily', methods=['GET'])
@cross_origin()
def get_daily_time_series():
    # This endpoint returns the news as received from the API, summarized by GPT-4
    try:
        ticker = request.args.get('ticker')

        if ticker:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&apikey={alphaventage_key}'
            response = requests.get(url)
            stock_data = response.json()["Time Series (Daily)"]
            return jsonify({'timeseries': stock_data})
        else:
            return jsonify({'error': 'Ticker not provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
