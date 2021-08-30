import pandas as pd
import numpy as np
from gensim.models import CoherenceModel, LdaModel
import gensim
from textblob import TextBlob


def get_doc_topic_matrix(data, model, dictionary):
    """
    This function uses the input to produce a document to topic matrix with probabilities of each topic per document
    Args:
        data: Vector of documents, where each document is a list of token
        model: Multicore LDA model produced by gensim
        dictionary: Word dictionary for gensim lda model

    Returns:
        DataFrame with distribution of topics per doc, dominant topic and second topic

    """

    corpus = [dictionary.doc2bow(text) for text in data]
    topic_names_temp = [i for i in range(1, model.num_topics + 1)]
    topic_name_final = ["Topic" + str(i) for i in range(1, model.num_topics + 1)]
    topic_name_final.append('dominant_topic')
    topic_name_final.append('second_topic')
    document_topic = pd.DataFrame(np.nan, index=list(range(len(data))), columns=topic_names_temp)

    for index, value in enumerate(corpus):
        temp = pd.DataFrame(model.get_document_topics(value))
        for col in range(len(temp[0])):
            document_topic.iloc[index, temp[0][col]] = np.round(temp[1][col], 2)

    document_topic.fillna(0, inplace=True)
    temp = document_topic.apply(pd.Series.nlargest, axis=1, n=2)
    new = temp.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=2)
    document_topic = pd.merge(document_topic, new, left_index=True, right_index=True)
    document_topic.columns = topic_name_final
    return document_topic


def get_topic_word_matrix(model, relevant_word=10):
    """
    The function extracts a topic to term matrix with the most relevant words per topic
    Args:
        model: Multicore LDA model from gensim
        relevant_word: Number of top n words. Default is 10

    Returns:
        DataFrame topic to term with words as values
    """
    
    topics = model.show_topics(num_words=relevant_word, formatted=False)
    word_names_temp = ["Term" + str(i) for i in range(1, relevant_word + 1)]
    topic_word = pd.DataFrame(np.nan, index=list(range(len(topics))), columns=word_names_temp)
    access_word_list = 1
    access_word = 0

    for topic_id in range(len(topics)):
        for index in range(relevant_word):
            topic_word.iloc[topic_id, index] = topics[topic_id][access_word_list][index][access_word]
    topic_word['Topic'] = ["Topic" + str(i) for i in range(1, model.num_topics + 1)]
    topic_word.set_index('Topic', inplace=True)
    return topic_word


def merge_doc_word_matrix(data, word):
    """
    This function merges the content of the topic to words matrix to the dataset. Each rwo is a list of words.
    Args:
        data: DataFrame to merge the matrix on
        word: DataFrame, topic to word matrix

    Returns:
        Merged DataFrame
    """
    
    df_topic_keywords = word.copy()
    df_topic_keywords['Topic_keywords'] = df_topic_keywords.values.tolist()
    df_topic_keywords['Topic_number'] = range(1, len(word)+1, 1)
    df_topic_keywords = df_topic_keywords[['Topic_keywords', 'Topic_number']]
    df_topic_keywords = df_topic_keywords.reset_index().drop(columns=['Topic'])

    tmp = []

    for i in df_topic_keywords['Topic_keywords']:
        tmp.append([x for x in i if x is not None])

    df_topic_keywords['Topic_keywords'] = tmp
    
    result = pd.merge(data, df_topic_keywords, left_on='dominant_topic', right_on='Topic_number')
    result.drop(columns=['Topic_number'], inplace=True)
    return result


def get_importance_normalisation(_doc_topic_matrix):
    """
    This function calculates the importance of each topic. Idea is that the contribution of the topic to the document
    measures its importance. Normalised to 0-10
    Args:
        _doc_topic_matrix: DataFrame document to topic matrix with probabilities

    Returns:
        Series: Importance measure per topic
    """
    _doc_topic_matrix = _doc_topic_matrix[_doc_topic_matrix.columns.difference(['dominant_topic', 'second_topic'])]
    sum_doc_topic_matrix = _doc_topic_matrix.sum(axis=0)
    return sum_doc_topic_matrix.apply(lambda x: 10*(x - sum_doc_topic_matrix.min()) / (sum_doc_topic_matrix.max() -
                                                                                       sum_doc_topic_matrix.min()))


def get_sentiment_normalisation_rake_words(_rake_terms):
    """
    This function processes the Keywords from Rapid Keyword Extraction, measures their sentiment with TextBlob and
    and returns the normalised sentiment, scale 0-10
    Args:
        _rake_terms: DataFrame with topic number and corresponding term

    Returns:
        Series: Sentiment measure per topic
    """
    topic_names_temp = ["Topic" + str(i) for i in list(_rake_terms['topic_number'].unique())]
    _rake_terms['sentiment'] = [TextBlob(term).sentiment[0] for term in _rake_terms['term']]
    test = _rake_terms[['topic_number', 'sentiment']].groupby(['topic_number']).sum()['sentiment']
    test = test.apply(lambda x: 10*(x - test.min()) / (test.max() - test.min()))
    test.index = topic_names_temp
    return test


def get_sentiment_normalisation_model(model):
    """
    This function returns the sentiment per topic based on the lad model. The sentiment for each word is measured by
    TextBlob multiplied with the contribution to hte topic and normalised on a scale from 0-10
    Args:
        model: Lda Multicore model from gensim

    Returns:
        Series: Sentiment measure per topic
    """
    topic_names_temp = ["Topic" + str(i) for i in range(1, model.num_topics + 1)]
    top_word = model.show_topics(formatted=False, num_topics=model.num_topics, num_words=len(model.id2word))
    column_words = list(model.id2word.values())
    access_word_list = 1
    access_word = 0

    mat = pd.DataFrame(np.nan, index=list(range(len(top_word))), columns=column_words)
    for topic_id in range(len(top_word)):
        for index in range(len(column_words)):
            candidate = top_word[topic_id][access_word_list][index]
            mat[top_word[topic_id][access_word_list][index][access_word]][topic_id] = \
                TextBlob(''.join(candidate[0].split('_'))).sentiment[0] * candidate[1]
    temp = mat.sum(axis=1)
    temp = temp.apply(lambda x: 10*(x - temp.min()) / (temp.max() - temp.min()))
    temp.index = topic_names_temp
    return temp
