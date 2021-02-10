# Topic Analysis

In the following notebook the first steps to evaluate a Latent Dirichlet allocation model are performed.
In this case latent topics for tripadvisoer reviews are modeled. The coherence measure is used to find the optimal parametrisation of the model.
This model then can be used to find the topics of negative reviews which might have leads to improve the service or lead to reviews
which should be answered by the customer service.
This logic could also be applied to customer complaints to find latent topics in the text and improve customer
satisfaction.


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


## ToDo

- [x] Prepare data preprocessing
- [x] Evaluate the ideal number of topics
- [ ] Fine tune hyperparametrization
- [ ] Use Cosine Similarity to set up force graphs
- [ ] Add Sentiment to each document
- [ ] Build Flask App



## Authors

* **Simon Unterbusch**