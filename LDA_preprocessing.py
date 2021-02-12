from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import spacy
import unidecode
import contractions
import re
import pandas as pd

nlp = spacy.load('en', disable=['parser', 'ner'])
stop_words = stopwords.words('english')


def text_preprocessing(text, strip_html=True, extra_whitespace=True, accented_chars=True, contraction=True,
                       lowercase=True, space_correction=True):
    """preprocess text with default option set to true for all steps
    Input: Vector with text in each field
    Process: Stripping of HTML, additional whitespace, etc.
    Output: text vector
    """
    # convert all characters to lowercase
    if strip_html:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    # remove extra whitespaces
    if extra_whitespace:
        text = text.strip()
        text = " ".join(text.split())
    # remove accented characters
    if accented_chars:
        text = unidecode.unidecode(text)
    # expand contractions
    if contraction:
        text = contractions.fix(text)
    # convert all characters to lowercase
    if lowercase:
        text = text.lower()
    # The regular expression looks for cases where there is no whitespace after the end of a sentence
    if space_correction:
        text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)
    return text


def sent_to_words(sentences):
    """
    Input: Vector with text
    Process: Removing punctuation and making sentences into words
    Output: Vector of words
    """
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def bigram_trigram(data, bi_tri_count, bi_tri_thresh):
    """
    Input: Vector of text, default value 5 for count and 100 for threshold in the bi_gram model
    Process: Extracting bi_gram and tri_gram models of the data
    Output: Returns dictionary with bi_gram and tri_gram objects
    """

    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data, min_count=bi_tri_count, threshold=bi_tri_thresh)
    trigram = gensim.models.Phrases(bigram[data], threshold=bi_tri_thresh)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return {'bigram_mod': bigram_mod, 'trigram_mod': trigram_mod}


def remove_stopwords(stop_text):
    """
    Input: Vector of text
    Process: Removing stopwords
    Output: Returns vector of text
    """

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in stop_text]


def make_bigrams(bi_model, make_b_text):
    """
    Input: Vector of text
    Process: making bi_grams of the input vector
    Output: Returns vector of text
    """

    return [bi_model[doc] for doc in make_b_text]


def make_trigrams(tri_model, bi_model, make_t_text):
    """
    Input: Vector of text
    Process: making tri_grams of the input vector
    Output: Returns vector of text
    """
    return [tri_model[bi_model[doc]] for doc in make_t_text]


def lemmatization(lem_text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Input: Vector of text
    Process: Lemmatize the input vector
    Output: Returns vector of text
    """
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in lem_text:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def lda_processing(lda_input, model, in_strip_html=True, in_extra_whitespace=True, in_accented_chars=True,
                   in_contraction=True, in_lowercase=True, in_space_correction=True, count=5, thresh=100):
    """
    Input: Vector of text, model either bi_gram or tri_gram, other input values for the sub functions
    Process: performing the sub-functions in the correct order
    Output: Returns processed text corpus and id2word in a dictionary
    """

    lda_input = lda_input.apply(lambda x: text_preprocessing(x, strip_html=in_strip_html,
                                                             extra_whitespace=in_extra_whitespace,
                                                             accented_chars=in_accented_chars,
                                                             contraction=in_contraction, lowercase=in_lowercase,
                                                             space_correction=in_space_correction))
    # print('Initial processing completed')

    lda_input = list(sent_to_words(lda_input))
    # print('Sentence to word complete')

    models = bigram_trigram(lda_input, bi_tri_count=count, bi_tri_thresh=thresh)
    # print('Bigram, Trigram model derived')

    lda_data_words_nostops = remove_stopwords(lda_input)

    if model == 'bi_gram':
        lda_data_words_model = make_bigrams(models['bigram_mod'], lda_data_words_nostops)
        # print('Bigram words derived')
    elif model == 'tri_gram':
        lda_data_words_model = make_trigrams(models['trigram_mod'], models['bigram_mod'], lda_data_words_nostops)
        # print('Trigram words derived')
    else:
        print('Select model as bi_gram or tri_gram')

    # Do lemmatization keeping only noun, adj, vb, adv
    lda_data_lemmatized = lemmatization(lda_data_words_model, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # print('Data lemmatized')

    # Create Dictionary 
    lda_id2word = corpora.Dictionary(lda_data_lemmatized)
    # Create Corpus 
    lda_texts = lda_data_lemmatized
    # Term Document Frequency 
    lda_corpus = [lda_id2word.doc2bow(text) for text in lda_texts]
    # print('Processing completed')

    return {'text_corpus': lda_corpus, 'id2word': lda_id2word, 'bi_tri_models': models,
            'data_output': lda_data_lemmatized}


def getIndexes(dfObj, value): 
    """
    Input:
    Datafram which to look for the specific value
    vlaue to look for
    
    Returns column and row of the value
    """
      
    listOfPos = [] 
       
    result = dfObj.isin([value]) 
      
    seriesObj = result.any() 
  
    columnNames = list(seriesObj[seriesObj == True].index) 
     
    for col in columnNames: 
        rows = list(result[col][result[col] == True].index) 
  
        for row in rows: 
            listOfPos.append((row, col)) 

    return listOfPos


def lda_get_topics(model_lda, in_dat, tops, topic_order=0):
    """
    Input:
    in_data: Text for which the topic shall be evaluated
    tops: dataframe of the topics
    
    Functions processes the text and gives the estimated topic from the model
    """
    
    doc_vector = model_lda.id2word.doc2bow(in_dat.split())
    doc_topics = model_lda[doc_vector]
    df =pd.DataFrame(doc_topics[0]).sort_values(1, ascending=False)
    return tops.loc[df[0].iloc[topic_order], 'Topic_name']
