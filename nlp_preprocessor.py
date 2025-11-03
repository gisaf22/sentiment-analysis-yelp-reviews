from bs4 import BeautifulSoup
import contractions
import re
from unidecode import unidecode
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Precompile the URL regex once for speed
URL_PATTERN = re.compile(
    r'https?://\S+|www\.\S+'   # matches http://, https:// or www. links
)

# Helper fn
def remove_html_tags(text):
    try:          
        # Handle non-string inputs gracefully
        if not isinstance(text, str):
            return ""

        # Parse HTML
        soup = BeautifulSoup(text, "html.parser")

        # If any tags are present
        for tag in soup.find_all(['iframe', 'script', 'a']):
            tag.decompose()

        # Extract plain text
        stripped_text = soup.get_text()

        # Remove any remaining raw URLs in the text
        stripped_text = URL_PATTERN.sub('', stripped_text)
        
        # Replace all linebreaks with a single '\n', newline
        stripped_text = re.sub(r'\r\n?|\n', '\n', stripped_text)

        return stripped_text

    except TypeError as e:
        error_message = str(e)
        print(f"Error message: {error_message}")
        return ""
    

def text_preprocessor(text):
    if text is None:
        return ""

    # Remove HTML tags
    # Expand Contractions
    text_cleaned = contractions.fix(remove_html_tags(text))
    
    # Replace accented characters
    # Lower case
    clean_txt = unidecode(text_cleaned).lower()

    # Collapse whitespace
    return re.sub(r'\s+', ' ', clean_txt).strip()

