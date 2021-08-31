import spacy
from bs4 import BeautifulSoup
import unidecode
import re
import contractions
import unicodedata
import string
from collections import Counter
import nltk
import pandas as pd
from rake_nltk import Rake
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from textwrap import TextWrapper

wordnet_lemma = WordNetLemmatizer()

nlp = spacy.load('en_core_web_lg')


def text_preprocessing(doc, stopwords=None):
    """
    The function takes a document as input and performs preprocessing
    Args:
        doc: A document, one string
        stopwords: A list of stopwords to be removed

    Returns:
        A list of tokens for the document
    """

    # convert all characters to lowercase
    doc = doc.lower()

    # The regular expression looks for cases where there is no whitespace after the end of a sentence
    pattern = r'(?<=[.,])(?=[^\s])'
    doc = re.sub(pattern, r' ', doc)

    # strip html
    soup = BeautifulSoup(doc, "html.parser")
    doc = soup.get_text(separator=" ")

    # remove accented characters
    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # expand contractions
    doc = contractions.fix(doc)
    doc = doc.replace("_", " ")

    # Remove numbers
    pattern = r'[\d]'
    doc = re.sub(pattern, '', doc)
    
    doc = doc.translate(str.maketrans('', '', string.punctuation))

    # remove single character
    pattern = r'(?<![\w])(?:[a-zA-Z0-9](?: |$))'
    doc = re.sub(pattern, '', doc)

    # remove extra whitespaces
    doc = re.sub(' +', ' ', doc)

    # Make word lemma and remove entities, stopwords
    nlp_data = nlp(doc)

    pre_result = [word.lemma_ for word in nlp_data if word.ent_type_ == '']

    if stopwords:
        result = [word for word in pre_result if word not in stopwords]
        return result
    else:
        return pre_result


def get_word_frequency(data_frame):
    """
    Processes the input and returns a dictionary with word occurrence
    Args:
        data_frame: Series of documents, where one document is a list of token

    Returns:
        Returns dictionary with word frequency in the series
    """

    word_count_dict = Counter()
    texts = [
        [word for word in document]
        for document in data_frame
    ]
    for text in texts:
        for word in text:
            word_count_dict[word] += 1

    return word_count_dict


def rake_preprocessing(doc):
    """
    Basic preprocessing for Rapid Keyword Extraction
    Args:
        doc: Input is on document in string representation

    Returns:
        Returns document in string representation
    """
    
    contractions.add("wo n't", 'will not')
    contractions.add("can n't", 'can not')
    contractions.add("n't", 'not')
    contractions.add("p m ", 'pm')
    contractions.add("a m ", 'am')
    
    soup = BeautifulSoup(doc, "html.parser")
    doc = soup.get_text(separator=" ")
    pattern = r'(?<=[.,])(?=[^\s])'
    doc = re.sub(pattern, r' ', doc)
    doc = doc.lower()
    doc = contractions.fix(doc)
    doc = doc.replace("_", " ")
    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    pattern = r'(?<![\w])(?:[a-zA-Z0-9](?: |$))'
    doc = re.sub(pattern, '', doc)
    doc = re.sub(' +', ' ', doc)
    
    nlp_data = nlp(doc)
    return ' '.join([word.text for word in nlp_data if word.ent_type_ == ''])

    

def get_rake_phrases(data, max_length=5, stopwords=None):
    """
    Processes the input vector and extracts keywords.
    Args:
        data: DataFrame witch columns rake_text (preprocessed text string), dominant_topic (dominant topic per review),
        Topic_keywords (list of keywords relevant to the topic from LDA)
        max_length: Maximum length of the keyword phrase. Default is 5
        stopwords: List of stopwords. Default None, than the nltk stopwords list is used

    Returns:
        Returns a dataframe with a parent keyword per topic and child keywords.
    """

    all_results = []

    if stopwords:
        rake = Rake(stopwords=stopwords, max_length=max_length)
    else:
        rake = Rake(max_length=max_length)
    
    for topic_id in data['dominant_topic'].unique():
        topic = data[data['dominant_topic'] == topic_id]

        # Extracting keywords per review
        rake_key_words = []

        for review in topic['rake_text'].values.tolist():

            rake.extract_keywords_from_text(review)
            temp_rake_keywords = rake.get_ranked_phrases_with_scores()

            for keyword in temp_rake_keywords:
                if keyword not in rake_key_words:
                    rake_key_words.append(keyword)

        rake_key_words = pd.DataFrame(rake_key_words, columns=['score', 'term'])
        rake_key_words = rake_key_words.sort_values('score', ascending=False)
        rake_key_words = rake_key_words.drop_duplicates(subset=['term'])
        rake_key_words['topic_number'] = topic_id

        rake_key_words['term_list'] = rake_key_words.term.apply(lambda x: x.split())

        # Phrase bigram from rake_key_words and produce the frame_rake_key_words
        temp_new_rake_keywords_frame = []

        for row_rake_keywords in rake_key_words.values.tolist():
            keyword_list = row_rake_keywords[3]
            new_keyword_list = []

            #TODO add parameter to maybe include tri grams as well
            temp_bi_grams = ngrams(keyword_list, 2)

            for bi_gram in temp_bi_grams:
                new_keyword_list.append('_'.join(bi_gram))

            for keyword in keyword_list:
                new_keyword_list.append(keyword)
                new_keyword_list.append(wordnet_lemma.lemmatize(keyword))

            row_rake_keywords.remove(keyword_list)
            row_rake_keywords.append(list(set(new_keyword_list)))
            temp_new_rake_keywords_frame.append(row_rake_keywords)

        frame_rake_key_words = pd.DataFrame(temp_new_rake_keywords_frame,
                                            columns=['score', 'term', 'topic_number', 'term_list'])

        # Validate which keywords from LDA are used in the rake keywords. Only keep these rake keywords
        LDA_keywords = topic['Topic_keywords'].values.tolist()
        flat_list = [item for sublist in LDA_keywords for item in sublist]
        list_LDA_keywords = list(set(flat_list))

        tmp_intersection_rake_lda = []

        for LDA_keyword in list_LDA_keywords:

            mask = frame_rake_key_words.term_list.apply(lambda x: LDA_keyword in x)
            key_words_processed = frame_rake_key_words[mask]

            if key_words_processed.empty:
                pass
            else:
                for row in key_words_processed[['score', 'term', 'topic_number']].values.tolist():
                    if row not in tmp_intersection_rake_lda:
                        tmp_intersection_rake_lda.append(row)

        keyword_intersection_rake_lda = pd.DataFrame(tmp_intersection_rake_lda,
                                                     columns=['score', 'term', 'topic_number'])
        
        tw = TextWrapper()
        tw.width = 50
        for index in range(len(keyword_intersection_rake_lda['term'])):
            keyword_intersection_rake_lda['term'][index] = "<br>".join(tw.wrap(keyword_intersection_rake_lda['term'][index]))
        
        # Top score of rake keywords are used as parent term, the rest as child
        max_value = keyword_intersection_rake_lda['score'].max()
        top_key_words = keyword_intersection_rake_lda[keyword_intersection_rake_lda.score == max_value]
        child_keywords = keyword_intersection_rake_lda[keyword_intersection_rake_lda.score != max_value]

        # if there are more than 1 keyword in the topic title, aggregate them with a <br> so it can be plotted by plotly
        top_key_words = top_key_words.copy()
        top_key_words = top_key_words.groupby(['score', 'topic_number']).agg({'term': lambda x: ','
                                                                             .join(map(str, x))})

        top_key_words = top_key_words.reset_index()
        top_key_words['parent'] = ''

        child_keywords = child_keywords.copy()
        child_keywords['topic_number'] = child_keywords['topic_number']
        child_keywords['parent'] = top_key_words['term'].values.tolist()[0]

        all_topics = pd.concat([top_key_words, child_keywords], sort=False)

        for t in all_topics.to_dict(orient='records'):
            all_results.append(t)

    all_topics_df = pd.DataFrame(all_results)
    all_topics_df = all_topics_df.sort_values('topic_number', ascending=True)

    return all_topics_df
