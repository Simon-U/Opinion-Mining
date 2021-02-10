import contractions
import spacy
import re
from word2number import w2n
import unidecode
from bs4 import BeautifulSoup
import math
#from spacy import nlp
nlp = spacy.load("en_core_web_lg")



def text_preprocessing(text, strip_html=True, extra_whitespace=True, accented_chars=True, contraction=True, space_correction =True,
                       stop_words=True, punctuations=True, special_chars=True, remove_num=True, convert_num=True, lemmatization=True         
                       ):
    """preprocess text with default option set to true for all steps"""
    
    if strip_html == True: #convert all characters to lowercase
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    if extra_whitespace == True: #remove extra whitespaces
        text = text.strip()
        text = " ".join(text.split())
    if accented_chars == True: #remove accented characters
        text = unidecode.unidecode(text)
    if contraction == True: #expand contractions
        text = contractions.fix(text)
    if space_correction == True:
        text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)

        
    doc = nlp(text)
    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
         # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
         # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
         # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        if len(token) < 2:
            flag = False
         # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag == True:
            flag = False
         # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
         # convert tokens to base form        
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
         # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)
        #data['Review'] = clean_text    
    return clean_text
"""
def token_to_vector(text):
    doc = nlp(text)
    clean_text = []
    for token in doc:
        clean_text.append(token.vector_norm)
    return clean_text
    
def make_sentence(text):
    text =text.apply(lambda x:' '.join([i+' ' for i in x]))
    # Removing double spaces if created
    text =text.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    
    return text
"""

def wordFrequ(doc):
    word_frequencies = {}
    for word in doc:
        if word.text not in word_frequencies.keys():
            word_frequencies[word.text] = 1
        else:
            word_frequencies[word.text] += 1
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    return word_frequencies

def IDF(docum):
    word_frequencies = {}
    for doc in docum:
        doc = nlp(doc)
        for word in doc:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
    return word_frequencies

#Function works and goes through on review
#Missing:
#PoS
#DEP
def word2features(doc,IDF_dic, doc_num):
    """
    Function loops over 
    """
    num_of_token = len(doc)
    features = []
    max_len_token = len(max(doc, key=len))
    
    Word_frequencies = wordFrequ(doc)
    
    for sent in doc.sents:
        feature_sent = []
        doc_sent = nlp(str(sent))
        for token in doc_sent:
            feature = {
                    'bias': 1.0,
                    'word': token.lower_,
                    'isTitle': token.is_title,
                    'entity': token.ent_type_,
                    'tag': token.tag_,
                    'Len': len(token)/max_len_token,
                    'PoS': token.pos,
                    'TF*IDF': Word_frequencies[str(token)] * math.log(doc_num/IDF_dic[str(token)]),
                    'DEP': 1,
                    'vect_norm': token.vector_norm,
                   }
            if token.i > 0:
                    feature.update({
                        '-1:word': token.nbor(-1).lower_,
                        '-1:isTitle': token.nbor(-1).is_title,
                        '-1:entity': token.nbor(-1).ent_type_,
                        '-1:tag': token.nbor(-1).tag_,
                        '-1:Len': len(token.nbor(-1))/max_len_token,
                        '-1:PoS': token.nbor(-1).pos,
                        '-1:TF*IDF': Word_frequencies[str(token.nbor(-1))] * math.log(doc_num/IDF_dic[str(token.nbor(-1))]),
                        '-1:DEP': 1,
                        '-1:vect_norm': token.nbor(-1).vector_norm,
                    })
            else:
                    feature['BOS'] = True

            if token.i < len(doc_sent)-1:
                    feature.update({
                        '+1:word': token.nbor(1).lower_,
                        '+1:isTitle': token.nbor(1).is_title,
                        '+1:entity': token.nbor(1).ent_type_,
                        '+1:tag': token.nbor(1).tag_,                       
                        '+1:Len': len(token.nbor(1))/max_len_token,
                        '+1:PoS': token.nbor(1).pos,
                        '+1:TF*IDF': Word_frequencies[str(token.nbor(1))] * math.log(doc_num/IDF_dic[str(token.nbor(1))]),
                        '+1:DEP': 1,
                        '+1:vect_norm': token.nbor(1).vector_norm,
                    })
            else:
                    feature['EOS'] = True
            feature_sent.append(feature)
        features.append(feature_sent)
    return features