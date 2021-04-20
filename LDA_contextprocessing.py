from gensim.models import CoherenceModel
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

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



def context_processing(lda_input, model, count=5, thresh=100, lower_limit=10, upper_limit=0.5):
    """
    Input: Vector of text, model either bi_gram or tri_gram, other input values for the sub functions
    Process: performing the sub-functions in the correct order
    Output: Returns processed text corpus and id2word in a dictionary
    """   
    
    lda_input = list(sent_to_words(lda_input))

    models = bigram_trigram(lda_input, bi_tri_count=count, bi_tri_thresh=thresh)

    if model == 'bi_gram':
        lda_data_words_model = make_bigrams(models['bigram_mod'], lda_input)
    elif model == 'tri_gram':
        lda_data_words_model = make_trigrams(models['trigram_mod'], models['bigram_mod'], lda_input)
    else:
        print('Select model as bi_gram or tri_gram')

    # Create Dictionary 
    lda_id2word = corpora.Dictionary(lda_data_words_model)
    lda_id2word.filter_extremes(no_below=lower_limit, no_above=upper_limit)
    
    # Term Document Frequency 
    lda_corpus = [lda_id2word.doc2bow(text) for text in lda_data_words_model]

    return {'text_corpus': lda_corpus, 'id2word': lda_id2word, 'bi_tri_models': models}

