# Opinion mining from hotel reviews

In the following notebook the TripAdvisor dataset with hotel reviews will be analysed.
The idea is to mine the opinion of the hotel stay from customer reviews. 
Latent Dirichlet Allocation (LDA) is used to cluster the reviews in topics and afterwards Rapid Keyword Extraction to
provide more meaningful names for the clusters.

1. Preprocess the dataset
2. Find optimal model
3. Visualization
4. Build Dash web app

The relevant files is LDA.ipynb notebook and the corresponding modules. 


## Prerequisites

To work with the project the following technologies need to be installed:

- Jupyter Notebook

To run the code successfully in the notebook the following packages need to be installed with the pip install command:
The necessary dependencies can be installed with the requirements.txt and the command
pip install -r requirements.txt


## Data sources and table structure

The data is from Kaggle and can be found under the following link:

https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews

## Steps

The procedure is outlined in the LDA.ipynb with links to the respective chapters.


## ToDo

- [x] Prepare data preprocessing
- [x] Find context to use (bi_gram)
- [x] Find model parameters and number of topics
- [x] Visualization
- [ ] Build Dash app


## Authors

* **Simon Unterbusch**