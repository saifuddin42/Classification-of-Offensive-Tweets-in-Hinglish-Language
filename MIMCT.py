# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:49:44 2020

@author: Home
"""
from collections import Counter
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
f = "B:\CS 695\Assignment3\Classification-of-Offensive-Tweets-in-Hinglish-Language\english/agr_en_train.csv"

# preprocessing english tweets.
#ingesting english csv file
df = pd.read_csv(f,names = ['source','comment','annotation'],encoding='UTF-8')
df['comment'] = df.comment.str.strip()   # removing spaces

comments = np.asarray(df['comment'])    # dividing the dataframe into comments and tags and converting to array
tags = np.asarray(df['annotation'])
print((len(comments)))
print(len(tags))

stop_words = set(stopwords.words('english'))  #english stop words list
processed_tokens = []
for comment in comments:
#    comment = "Also see ....hw ur RSS activist caught in Burkha .... throwing beef in d holy temples...https://www.google.co.in/amp/www.india.com/news/india/burkha-clad-rss-activist-caught-throwing-beef-at-temple-pictures-go-viral-on-facebook-593154/amp/,NAGfacebook_corpus_msr_403402,On the death of 2 jawans in LOC CROSS FIRING"
    comment = comment.lower()   #lower casing each tweets
    Digit_REMOVAL = re.sub(r'[0-9]+', '',comment) #removal of numbers 
    URL_REMOVAL = re.sub(r"http\S+", "", Digit_REMOVAL) # removal of URLS
    tokenizer = nltk.RegexpTokenizer(r"\w+")   # removal of punctuation and tokenizing
    new_words = tokenizer.tokenize(URL_REMOVAL)
    sentence = []
    for word in new_words:
        if word not in stop_words:           #checking for stop words on each sentence
            sentence.append(word)
    processed_tokens.append(sentence)


(processed_tokens[11])
len(tags)
set(stopwords.words('english'))
