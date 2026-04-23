"""
Filename: lang_utils.py
Author: Digital Hub
Date: 2025-06-11
Version: 0.1.0
Description: 
    provides language filtering for accurate evaluation of languages
"""

import re
import string

def filter_string(input_string: str, lang: str = "en") -> str:
    ''' Orchestrator for language filtering '''
    
    input_string = remove_punctuation(input_string=input_string)
    
    lang = lang.lower()
    
    if lang == "en" or lang == "english":
        
        filtered_string = filter_english(input_string=input_string)

    elif lang == "hi" or lang == "hindi":
        
        filtered_string = filter_hindi(input_string=input_string)
    
    elif lang == "th" or lang == "thai": 
        
        filtered_string = filter_thai(input_string=input_string)
    
    elif lang == "vi" or lang == "vietnamese": 
        
        filtered_string = filter_vietnamese(input_string=input_string)
    
    elif lang == "bn" or lang == "bengali": 
        
        filtered_string = filter_bengali(input_string=input_string)
    
    elif lang == "fil" or lang == "tagalog": 
        
        filtered_string = filter_filipino(input_string=input_string)
    
    elif lang == "ms" or lang == "malay":
        
        filtered_string = filter_malay(input_string=input_string)
    
    elif lang == "id" or lang == "indonesian":
        
        filtered_string = filter_indonesian(input_string=input_string)
    
    elif lang == "zh" or lang == "chinese":
        
        filtered_string = filter_simplified_chinese(input_string=input_string)
    
    elif lang == "ar" or lang == "arabic":
        
        filtered_string = filter_arabic(input_string=input_string)
    
    else:
        # Code for other languages
        filtered_string = input_string

    filtered_string = remove_needless_whitespace(input_string=filtered_string)
    return filtered_string

def filter_english(input_string: str) -> str:
    """
    Filter the string such that comparison for WER and evaluation metrics are accurate.
    Current filters:
        - Lowercase all characters
        - Only keep alphabetic characters and whitespaces

    Parameters:
    input_string (str): The string to be filtered.

    Returns:
    String: The filtered string.
    """
    # Convert both texts to lowercase
    input_string = input_string.lower()
    # Use regular expression to find all alphabetic characters and whitespaces
    filtered_string = re.sub(r"[^a-zA-Z\s]", "", input_string)
    
    return filtered_string

def filter_hindi(input_string: str) -> str:
    """ Filtering hindi """
    # Convert input to lowercase (note: Hindi script doesn't have case)
    # Retain only Hindi characters (Devanagari Unicode range) and spaces
    filtered_string = re.sub(r"[^\u0900-\u097F\s]", "", input_string)
    
    return filtered_string

def filter_thai(input_string: str) -> str:
    """ Filtering Thai """
    # Retain only Thai characters and whitespace
    filtered_string = re.sub(r"[^\u0E00-\u0E7F\s]", "", input_string)
    
    return filtered_string

def filter_vietnamese(input_string: str) -> str:
    """ Filtering Vietnamese """
    # This regex retains all Vietnamese letters and whitespace
    input_string = input_string.lower()
    filtered_string = re.sub(r"[^a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂÊÔƠƯăêôơư\s]", "", input_string)
    
    return filtered_string

def filter_bengali(input_string: str) -> str:
    """ Filtering Bengali """
    # Keep only Bengali characters and whitespace
    filtered_string = re.sub(r"[^\u0980-\u09FF\s]", "", input_string)
    return filtered_string

def filter_filipino(input_string: str) -> str:
    """ Filtering Filipino """
    # Retain only Filipino letters (including ñ, é) and whitespace
    input_string = input_string.lower()
    filtered_string = re.sub(r"[^a-zA-ZñÑéÉ\s]", "", input_string)
    return filtered_string

def filter_arabic(input_string: str) -> str:
    """Filtering Arabic letters and whitespace from input string."""
    # Arabic Unicode ranges + whitespace
    filtered_string = re.sub(r"[^\u0600-\u06FF\s]", "", input_string)
    return filtered_string


def filter_simplified_chinese(input_string: str) -> str:
    """Filtering CJK characters (which includes Simplified Chinese) and whitespace."""
    # CJK Unified Ideographs: \u4e00–\u9fff
    filtered_string = re.sub(r"[^\u4e00-\u9fff\s]", "", input_string)
    return filtered_string

def filter_indonesian(input_string: str) -> str:
    """Filtering for only Indonesian alphabet characters and whitespace."""
    from indonesian_number_normalizer import create_normalizer
    ind_normalizer = create_normalizer()
    input_string = ind_normalizer.normalize_text(input_string)
    
    input_string = input_string.lower()
    filtered_string = re.sub(r"[^a-zA-Z\s]", "", input_string)
    return filtered_string

def filter_malay(input_string: str) -> str:
    """Filtering for only Malay characters and whitespace."""
    # Includes standard Latin letters + common Malay accents + whitespace
    input_string = input_string.lower()
    filtered_string = re.sub(r"[^a-zA-ZéÉèÈâÂîÎôÔ\s]", "", input_string)
    return filtered_string

def remove_punctuation(input_string: str) -> str:
    """Remove punctuation characters, keeping letters, digits, and whitespace."""
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", input_string)

# from num2words import num2words
# def convert_numbers_to_words(text: str) -> str:
#     """
#     Convert all integer or decimal numbers in a string to their word form.

#     Args:
#         text (str): Input string containing numbers.

#     Returns:
#         str: String with numbers replaced by words.
#     """
#     # This regex finds integers and decimals (e.g., 123, 45.67)
#     number_pattern = re.compile(r"\d+(?:\.\d+)?")

#     def replacer(match):
#         number_str = match.group(0)
#         try:
#             if "." in number_str:
#                 return num2words(float(number_str))
#             else:
#                 return num2words(int(number_str))
#         except ValueError:
#             return number_str  # fallback in case of unexpected format

#     return number_pattern.sub(replacer, text)

def remove_needless_whitespace(input_string:str):
    '''
    combine consecutive whitespaces into 1
    '''
    
    filtered_string = re.sub(r"\s+", " ", input_string).strip()
    
    return filtered_string