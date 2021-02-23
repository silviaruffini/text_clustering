# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:04:30 2020

@author: s.ruffini
"""

import spacy
import pandas as pd
import re

# Load a English model and create the nlp object
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words


#Personalized stopwords
stopwords_extended= ["eu","ms","august","czech", "republic","programme","new","aim","russian","response","project", "ban","use","tonne"]
stopwords_extended.extend(stopwords)


# Function removing HTML/XMLtags
def striphtml(data):
    pattern = re.compile(r'<.*?>')
    return pattern.sub('', data)



# Function to preprocess text
def text_preprocess_cleaning(text):
  	# Create Doc object
    doc = nlp(text)
    
    # Generate lemmas and lower case and remove punctuation
    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct]
  
    # Remove stopwords  characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords_extended ]
    #
    return ' '.join(a_lemmas)
  
    
##########################   
## TEXT PREPROCESSING 
##########################
#### this function perform all steps needed for preprocessing text 
#### input : df= dataframe
####         label = field's label of dataframe to apply text_preprocessing_cleaning
####output : df= dataframe cleaned
##########################    

def preprocessing_process (df, label):
    
    
    # 1. drop missing values
    df.dropna(subset=[label], inplace=True)
    
    
    # 2. Removing HTML/XMLtags
    for text, row in df.iterrows():
        df.loc[text, label] = striphtml(row[label])


    # 3. Apply preprocess text
    df[label] = df[label].apply(text_preprocess_cleaning)

    # 3. drop empty row resulting from text preprocess
    df = df.drop(df[df[label]==""].index)
    
    return df
