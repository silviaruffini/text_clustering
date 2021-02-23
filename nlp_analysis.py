# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:29:27 2021

@author: s.ruffini
"""
import pandas as pd
import preprocessing_text as pt
from sklearn.feature_extraction.text import TfidfVectorizer


df= pd.read_csv("eu_projects.csv", sep=";")


pt.preprocessing_process(df,'description')


#define vectorizer parameters
vectorizer = TfidfVectorizer(ngram_range=(1,1))

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(df['description'])


