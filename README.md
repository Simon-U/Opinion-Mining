# Topic Analysis

In the following notebook the first steps to evaluate a Latent Dirichlet allocation model are performed.
In this case latent topics for Tripadvisor reviews are modeled. The coherence measure is used to find the optimal parametrisation of the model.
This model then can be used to find the topics of negative reviews which might have leads to improve the service or lead to reviews
which should be answered by the customer service.
This logic could also be applied to other areas to find latent topics in the text and improve customer satisfaction.

The relevant files are LDA.ipynb and LDA_preprocessing.py. 

The other files are experiments to implement further features at a later point of time.


## Prerequisites

To work with the project the following technologies need to be installed:

- Jupyter Notebook

To run the code successfully in the notebook the following packages need to be installed with the pip install command:

```
nltk
bs4
gensim
spacy
unidecode
contractions
re
pandas
numpy
pyLDAvis
matplotlib
seaborn
```


## Data sources and table structure

The data is from Kaggle and can be found under the following link:

https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews

## Steps

1. Loading the data
2. The data is processed in bi_gram and tri_gram modells and for each the LDA model is calculated with 1 to 29 topics
3. For each model the coherence score is calculated to finde the optimal model
4. Optimal model calulated
5. pyLDAvis and WordClounds for Topics plotted
6. Topics added to each review for later use
7. t-SNE Clustering used to plot the topics


## ToDo

- [x] Prepare data preprocessing
- [x] Fine tune hyperparametrization
- [x] Add topics to reviews
- [ ] Add Cosine Similarity
- [ ] Add Sentiment to each document
- [ ] Set up Graphs for visualisation 
- [ ] Build Flask App



## Authors

* **Simon Unterbusch**