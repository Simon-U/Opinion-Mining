import contractions
import spacy
import re
from word2number import w2n
import unidecode
from bs4 import BeautifulSoup
import math
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('SpacyTextBlob')


def get_sent(text):
    doc = nlp(text)
    return doc._.sentiment.polarity

def get_subj(text):
    doc = nlp(text)
    return doc._.sentiment.subjectivity
