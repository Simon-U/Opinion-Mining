import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import spacy
from bs4 import BeautifulSoup
import unidecode
import re
import contractions
import unicodedata

nlp = spacy.load('en_core_web_sm')
stop_words = nltk.corpus.stopwords.words('english')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')


contractions.add("n't", "not")
contractions.add("1st", "first")
contractions.add("2st", "second")
contractions.add("3th", "third")


def lemmatize_text(text):
    """
    Input: Vector of text
    Process: Lemmatize the input vector
    Output: Returns vector of text
    """
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text



def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def text_normalization(text, new_stop = None, rem_stop = None, strip_html=True, extra_whitespace=True, accented_chars=True, 
                       contraction=True, lowercase=True, space_correction=True, remove_single_character = True, text_lemmatization=True, 
                       stopword_removal=True, special_char_removal = True, remove_duplicates = True, remove_digits=True):
   
    """preprocess text with default option set to true for all steps
    Input: Vector with text in each field
    Process: Stripping of HTML, additional whitespace, etc.
    Output: text vector
    """
    
    # Strip HTMl
    if strip_html:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    
    # remove accented characters  
    if accented_chars:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # expand contractions
    if contraction:
        text = contractions.fix(text)
    # convert all characters to lowercase
    if lowercase:
        text = text.lower()
    # The regular expression looks for cases where there is no whitespace after the end of a sentence
    if space_correction:
        text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)
    
    if special_char_removal:
    # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        text = special_char_pattern.sub(" \\1 ", text)
        text = remove_special_characters(text, remove_digits=remove_digits)
    
    if remove_single_character:
        pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
        text=  re.sub("\s+", " ", re.sub(pattern, '', text).strip())

    if text_lemmatization:
        text = lemmatize_text(text)
    
    if stopword_removal:
        [stopword_list.append(st) for st in new_stop]
        [stopword_list.remove(st) for st in rem_stop]
        text = remove_stopwords(text)
        
    if remove_duplicates:
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # remove extra whitespaces
    if extra_whitespace:
        text = text.strip()
        text = " ".join(text.split())
        text = re.sub(' +', ' ', text)
    
    return text
