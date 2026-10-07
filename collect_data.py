import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime

def collect_news_from_source(url, source_name, is_real=True, max_articles=50):
    """
    Collect news articles from a given source URL.
    This is a basic scraper - in production, use proper APIs.
    """
    articles = []

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find article links (this is source-specific and may need adjustment)
        article_links = []

        # For BBC - find news article links
        if 'bbc.com' in url:
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if '/news/' in href and 'live' not in href and len(href) > 20:
                    if href.startswith('/'):
                        href = 'https://www.bbc.com' + href
                    if href not in article_links:
                        article_links.append(href)

        # For CNN
        elif 'cnn.com' in url:
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if '/202' in href and len(href) > 30:  # Contains year
                    if href.startswith('/'):
                        href = 'https://www.cnn.com' + href
                    if href not in article_links:
                        article_links.append(href)

        # For Reuters
        elif 'reuters.com' in url:
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if '/article/' in href and len(href) > 30:
                    if href.startswith('/'):
                        href = 'https://www.reuters.com' + href
                    if href not in article_links:
                        article_links.append(href)

        print(f"Found {len(article_links)} potential article links from {source_name}")

        # Extract text from first max_articles
        for i, article_url in enumerate(article_links[:max_articles]):
            try:
                print(f"Extracting article {i+1}/{min(max_articles, len(article_links))}: {article_url}")

                # Extract article text
                article_response = requests.get(article_url, headers=headers, timeout=10)
                article_response.raise_for_status()

                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                # Remove script and style elements
                for script in article_soup(["script", "style"]):
                    script.extract()

                # Get text from paragraphs
                paragraphs = article_soup.find_all('p')
                text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

                if len(text) > 500:  # Only save substantial articles
                    articles.append({
                        'title': article_soup.title.string if article_soup.title else 'No Title',
                        'text': text,
                        'url': article_url,
                        'source': source_name,
                        'collected_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                time.sleep(1)  # Be respectful to servers

            except Exception as e:
                print(f"Error extracting {article_url}: {e}")
                continue

    except Exception as e:
        print(f"Error collecting from {source_name}: {e}")

    return articles

def main():
    print("📰 Fake News Detection - Data Collection Tool")
    print("=" * 50)

    # Real news sources
    real_sources = [
        ('https://www.bbc.com/news', 'BBC'),
        ('https://www.cnn.com', 'CNN'),
        ('https://www.reuters.com', 'Reuters'),
        ('https://www.nytimes.com', 'New York Times'),
    ]

    # Fake news sources (satire/conspiracy - be careful!)
    fake_sources = [
        ('https://www.theonion.com', 'The Onion'),  # Satire
        ('https://www.infowars.com', 'Infowars'),   # Conspiracy
    ]

    all_real_articles = []
    all_fake_articles = []

    # Collect real news
    print("\n📄 Collecting REAL news articles...")
    for url, name in real_sources:
        print(f"\n🔍 Collecting from {name}...")
        articles = collect_news_from_source(url, name, is_real=True, max_articles=25)
        all_real_articles.extend(articles)
        print(f"✓ Collected {len(articles)} articles from {name}")

    # Collect fake news
    print("\n📄 Collecting FAKE news articles...")
    for url, name in fake_sources:
        print(f"\n🔍 Collecting from {name}...")
        articles = collect_news_from_source(url, name, is_real=False, max_articles=25)
        all_fake_articles.extend(articles)
        print(f"✓ Collected {len(articles)} articles from {name}")

    # Save to CSV
    print("\n💾 Saving collected data...")
    os.makedirs('data_set', exist_ok=True)

    # Load existing data
    try:
        existing_real = pd.read_csv('data_set/True.csv')
        print(f"✓ Loaded {len(existing_real)} existing real articles")
    except:
        existing_real = pd.DataFrame(columns=['title', 'text', 'url', 'source', 'collected_date'])

    try:
        existing_fake = pd.read_csv('data_set/Fake.csv')
        print(f"✓ Loaded {len(existing_fake)} existing fake articles")
    except:
        existing_fake = pd.DataFrame(columns=['title', 'text', 'url', 'source', 'collected_date'])

    # Add new data
    if all_real_articles:
        new_real_df = pd.DataFrame(all_real_articles)
        combined_real = pd.concat([existing_real, new_real_df], ignore_index=True)
        combined_real.to_csv('data_set/True.csv', index=False)
        print(f"✓ Saved {len(combined_real)} total real articles")

    if all_fake_articles:
        new_fake_df = pd.DataFrame(all_fake_articles)
        combined_fake = pd.concat([existing_fake, new_fake_df], ignore_index=True)
        combined_fake.to_csv('data_set/Fake.csv', index=False)
        print(f"✓ Saved {len(combined_fake)} total fake articles")

    print("\n✅ Data collection complete!")
    print(f"📊 Summary:")
    print(f"   Real articles: {len(all_real_articles)} new")
    print(f"   Fake articles: {len(all_fake_articles)} new")
    print("\n🔄 Run 'python train.py' to retrain with new data")

if __name__ == '__main__':
    main()