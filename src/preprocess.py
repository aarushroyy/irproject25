import re
from typing import List

def basic_clean(text: str) -> str:
    """
    Lowercase, strip extra whitespace.
    For Hindi, we mostly keep characters as-is.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def basic_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.
    """
    text = basic_clean(text)
    return text.split()