import gensim
from gensim.models import Phrases
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


def context_processing(dataframe, lower_limit=20, upper_limit=0.5):
    """
    Takes the document as input and evaluates a bigram model.
    Args:
        dataframe: Series of documents where one document is represented by a list of token
        lower_limit: Words need to occur in at least n documents. Default 20
        upper_limit: Words cannot occur in more than x% of the documents. Default is 0.5 equals 50%

    Returns:
        Corpus and dictionary
    """

    bigram = Phrases(dataframe, min_count=20)
    for idx in range(len(dataframe)):
        result = []
        for token in bigram[dataframe[idx]]:
            if token.count('_') == 1:
                result.append(token)
        dataframe[idx].extend(result)
        
    # Create Dictionary 
    id2word = corpora.Dictionary(dataframe)
    id2word.filter_extremes(no_below=lower_limit, no_above=upper_limit)
    
    # Corpus
    corpus = [id2word.doc2bow(text) for text in dataframe]

    return corpus, id2word
