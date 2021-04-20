import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_lg')
spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)


def get_sent(text):
    doc = nlp(text)
    return doc._.sentiment.polarity

def get_subj(text):
    doc = nlp(text)
    return doc._.sentiment.subjectivity
