import re
import string
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources silently
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    # also try punkt_tab for newer nltk versions
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Tokenizing
    4. Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation using string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)


def count_real_news_indicators(text):
    if not isinstance(text, str):
        return 0
    text = text.lower()
    indicators = [
        "announced", "said", "official", "minister", "government", "report",
        "policy", "program", "agency", "statement", "according", "review",
        "committee", "investigation", "project", "development", "infrastructure",
        "service", "community", "president", "minister", "public"
    ]
    return sum(term in text for term in indicators)


def extract_text_from_url(url):
    """
    Extracts the main text content from a given URL.
    Returns a tuple (text, error_message).
    """
    try:
        # Add a more robust user-agent and headers to avoid getting blocked by some websites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code in [401, 403]:
            return None, "This website has security measures that block automated text extraction. Please copy the text manually and use the 'Manual Text Input' mode instead."
            
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from paragraph tags
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        if not text.strip():
            return None, "No text content could be extracted from the provided URL."
            
        return text, None
        
    except requests.exceptions.Timeout:
        return None, "Connection timed out. Please try a different URL."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        return None, f"This website returned an error (HTTP {status}) and likely blocks automated access. Please copy the text manually and use 'Manual Text Input'."
    except requests.exceptions.RequestException as e:
        return None, f"Could not reach the website. Please check the URL or your connection."
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"
