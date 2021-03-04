# Review analysis to find potential improvement for customer satisfacition

In the following notebook the TripAdvisor dataset with hotel reviews will be analysed.
The idea is to find improvement potentials for the hotels to increase customer satisfaction. 
Latent Dirichlet Allocation (LDA) is used to find the relevant topics and the rating to verify if the topic is negative.

Two approaches are used to find the optimal model.

1. First the LDA is used on the whole dataset.
2. Second LDA is used only on the negative reviews.

Reason is the hypothesis that the positive reviews are written more general and therefore introducing noise to the topics. Furthermore, the algorithm should be less likely to choose a topic with a high probability. 

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

The procedure is outlined in the LDA.ipynb with links to the respective chapters.


## ToDo

- [x] Prepare data preprocessing
- [ ] Filter extrems (word is in 50% of documents or in 10 or less documents)
- [x] Find context to use (bi_gram or tri_gram)
- [X] First approach
- [ ] First approach evaluation and topics visualisation
- [ ] Second approach
- [ ] Second approach evaluation and topic visualisation
- [ ] Conclusion
- [ ] Build Flask App



## Authors

* **Simon Unterbusch**