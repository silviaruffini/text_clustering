
import pandas as pd
import preprocessing_text as pt
from sklearn.feature_extraction.text import TfidfVectorizer


df= pd.read_csv("eu_projects.csv", sep=";")


pt.preprocessing_process(df,'description')


#define vectorizer parameters
vectorizer = TfidfVectorizer(ngram_range=(1,1))

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(df['description'])


