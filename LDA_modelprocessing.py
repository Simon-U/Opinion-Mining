import pandas as pd
from gensim.models import CoherenceModel
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


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
    if topic_order == 0:
        doc_vector = model_lda.id2word.doc2bow(in_dat.split())
        doc_topics = model_lda[doc_vector]
        df = pd.DataFrame(doc_topics[0]).sort_values(1, ascending=False)
        df = tops.loc[df[0].iloc[topic_order], 'Topic_name']
    else:
        doc_vector = model_lda.id2word.doc2bow(in_dat.split())
        doc_topics = model_lda[doc_vector]
        df = pd.DataFrame(doc_topics[0]).sort_values(1, ascending=False)
        if len(df)<2:
            df = float("nan")
        else:
            df = tops.loc[df[0].iloc[topic_order], 'Topic_name']
    return df



def lda_get_prob(model_lda, in_dat, topic_order=0):
    """
    Input:
    in_data: Text for which the topic shall be evaluated
    tops: dataframe of the topics
    
    Functions processes the text and gives the estimated probaility from the model
    """
    if topic_order == 0:
        doc_vector = model_lda.id2word.doc2bow(in_dat.split())
        doc_topics = model_lda[doc_vector]
        df =pd.DataFrame(doc_topics[0]).sort_values(1, ascending=False)
        df = df[1].iloc[topic_order]
    else:
        doc_vector = model_lda.id2word.doc2bow(in_dat.split())
        doc_topics = model_lda[doc_vector]
        df = pd.DataFrame(doc_topics[0]).sort_values(1, ascending=False)
        if len(df)<2:
            df = float("nan")
        else:
            df = df[1].iloc[topic_order]
    return df


def compute_coherence_values(corpus_c,  dictionary_c, k, a, b):
    """
    Input:
    in_data: Text data, lemmatized vector, word dictionary, number of topics, alpha, beta
   
    Functions processes the inout and returns the coherence score for the model
    """ 
    lda_model_c = gensim.models.LdaModel(corpus=corpus_c,
                                           id2word=dictionary_c,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model_c, corpus=corpus_c, coherence='u_mass')
    
    return coherence_model_lda.get_coherence()