# Topic Analysis

In the following notebook the first steps to evaluate a Latent Dirichlet allocation model are performed.


## Prerequisites

To work with the project the following technologies need to be installed:

 - Jupyter Notebook
          

To run the code successfully in the notebook the following packages need to be installed with the pip install command:

```
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
import numpy as np
from LDA_preprocessing import *
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import CoherenceModel
```


## Data sources and table structure

The data is from Kaggle and can be found under the following link:

https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews


## ToDo

- [x] Prepare data preprocessing
- [x] Evaluate the ideal number of topics
- [ ] Fine tune hyperparametrization
- [ ] Use Cosine Similarity to set up force graphs
- [ ] Add Sentiment to each document
- [ ] Build Flask App



## Authors

* **Simon Unterbusch**