import re
import logging
class TextCleaner:
    """
    Cleans consumer complaint narratives for NLP embedding.
    """
    def __init__(self):
        # Precompile boilerplate regex patterns for efficiency
        self.boilerplate_patterns = [
            re.compile(r"i am writing to file a complaint", re.IGNORECASE),
            re.compile(r"this complaint is regarding", re.IGNORECASE),
            re.compile(r"i would like to file a complaint", re.IGNORECASE),
            re.compile(r"i am submitting this complaint", re.IGNORECASE),
            re.compile(r"please help me with this issue", re.IGNORECASE),
        ]
        # Precompile URL pattern
        self.url_pattern = re.compile(r'http\S+|www\S+', re.IGNORECASE)
        # Precompile special character pattern
        self.special_char_pattern = re.compile(r"[^a-z0-9\s]")
    def clean(self, text):
        """
        Clean a single complaint narrative.
        """
        try:
            if not isinstance(text,str):
                logging.warning("Non-string input encountered during text cleaning.")
                return ""
            text = text.lower()
            # Remove boilerplate phrases
            for pattern in self.boilerplate_patterns:
                text = pattern.sub('', text)
            # Remove URLs
            text = self.url_pattern.sub('', text)
            # Remove special characters
            text = self.special_char_pattern.sub('', text)
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            return text
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            return ""