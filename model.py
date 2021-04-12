#runs a logistic regression
#dataset and methodology borrowed form the SHAP documentation, found here:
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Sentiment%20Analysis%20with%20Logistic%20Regression.html#Fit-a-linear-logistic-regression-model


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap

def give_shap_plot(): 
    
  corpus,y = shap.datasets.imdb()
  corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=7)

  vectorizer = TfidfVectorizer(min_df=10)
  X_train = vectorizer.fit_transform(corpus_train).toarray() # sparse also works but Explanation slicing is not yet supported
  X_test = vectorizer.transform(corpus_test).toarray()

  model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
  model.fit(X_train, y_train)

  explainer = shap.Explainer(model, X_train, feature_names=vectorizer.get_feature_names())
  shap_values = explainer(X_test)
  
  return explainer, shap_values
