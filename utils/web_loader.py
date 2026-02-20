import requests
from bs4 import BeautifulSoup
from typing import Optional

def extract_text_from_url(url: str) -> Optional[str]:
    """
    Extracts text from a URL.
    Returns the extracted text or None if fetching fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None
