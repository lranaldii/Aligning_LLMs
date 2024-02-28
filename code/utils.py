import difflib
import re

def best_match(target, *candidates):
    """
    Find the best matching string from the candidates based on the SequenceMatcher ratio.
    
    Args:
    - target (str): The target string to match against.
    - candidates (list of str): The list of candidate strings.
    
    Returns:
    - float: The best match ratio.
    - str: The best matching string.
    """
    best_ratio = 0
    best_match_string = None
    
    for candidate in candidates:
        s = difflib.SequenceMatcher(None, target, candidate)
        ratio = s.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_string = candidate
            
    return best_ratio, best_match_string

def extract_answer(text):
    """
    Extract the answer from a given text using regex.
    
    Args:
    - text (str): The input text.
    
    Returns:
    - str: The extracted answer or None if not found.
    """
    match = re.search(r'(?:answer (?:is|from the given options would be)|correct answer to the question .+? is) ([A-Da-d]\) .+?)\.', text)
    if match:
        return match.group(1)
    return None

def check_matching(text, target, threshold=0.6):
    """
    Check if the extracted answer from the text matches the target with a given threshold.
    
    Args:
    - text (str): The input text.
    - target (str): The target answer.
    - threshold (float, optional): The matching threshold. Defaults to 0.6.
    
    Returns:
    - bool: True if the match ratio is above the threshold, False otherwise.
    """
    extracted_answer = extract_answer(text)
    if not extracted_answer:
        return False
    
    ratio, _ = best_match(target, extracted_answer)
    return ratio >= threshold
