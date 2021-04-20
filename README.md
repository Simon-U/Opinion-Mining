# Review analysis to find potential improvement for customer satisfaction

In the following notebook the TripAdvisor dataset with hotel reviews will be analysed.
The idea is to find improvement potentials for the hotels to increase customer satisfaction. 
Latent Dirichlet Allocation (LDA) is used to find the relevant topics, and the rating to verify if the topic is negative.

1. Preprocess the dataset
2. Find optimal model
3. Visualization
4. Build Dash web app
5. Use results to train a neural network model

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
- [x] Find context to use (bi_gram or tri_gram)
- [x] Find model parameters and number of topics
- [x] Visualization
- [ ] Build Dash app
- [ ] Train network


## Authors

* **Simon Unterbusch**