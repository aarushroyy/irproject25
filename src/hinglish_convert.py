from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from .preprocess import basic_clean

def hindi_to_hinglish(text: str) -> str:
    """
    Romanize Hindi Devanagari text into Latin script
    using ITRANS scheme.
    This gives you a 'Hinglish-like' version.
    """
    text = basic_clean(text)
    roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    roman = roman.lower().replace("_", " ")
    return roman