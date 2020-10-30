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
import fuzzywuzzy
from fuzzywuzzy import fuzz
from  nltk import word_tokenize


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)





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



#-----------------For hinglish dataset


Hindi_text  = "B:\CS 695\Assignment3\Classification-of-Offensive-Tweets-in-Hinglish-Language\hindi/agr_hi_dev.csv"
df1 = pd.read_csv(Hindi_text,names = ['source','comment','annotation'],encoding='UTF-8')
df1['comment'] = df.comment.str.strip()   # removing spaces
hindi_comments = np.asarray(df1['comment'])    # dividing the dataframe into comments and tags and converting to array
hindi_tags = np.asarray(df1['annotation'])
print((hindi_comments[11])) 
processed_Hindi_tokens = []
for comment in hindi_comments:
#    comment = "Also see ....hw ur RSS activist caught in Burkha .... throwing beef in d holy temples...https://www.google.co.in/amp/www.india.com/news/india/burkha-clad-rss-activist-caught-throwing-beef-at-temple-pictures-go-viral-on-facebook-593154/amp/,NAGfacebook_corpus_msr_403402,On the death of 2 jawans in LOC CROSS FIRING"
    comment = comment.lower()   #lower casing each tweets
    Digit_REMOVAL = re.sub(r'[0-9]+', '',comment) #removal of numbers 
    URL_REMOVAL = re.sub(r"http\S+", "", Digit_REMOVAL) # removal of URLS
    Emoji_removal = remove_emoji(URL_REMOVAL)
    if (isEnglish(Emoji_removal) == True):
        Emoji_removal = re.sub(r'[^\w\s]','',Emoji_removal)# removal of punctuation and tokenizing
    processed_Hindi_tokens.append(word_tokenize(Emoji_removal))
processed_Hindi_tokens[6]
processed_Hindi_tokens[11]
processed_Hindi_tokens[6]
processed_Hindi_tokens[11]
transliteration_dict = "B:\CS 695\Assignment3\Classification-of-Offensive-Tweets-in-Hinglish-Language/transliterations.hi-en.csv"
t_dict = pd.read_csv(transliteration_dict,names = ['Hinglish','Hindi'],encoding='UTF-8',sep='\t')
t_dict['Hinglish'] = t_dict['Hinglish'].str.strip()
t_dict['Hindi'] = t_dict['Hindi'].str.strip()
t_dict = np.asarray(t_dict)
print(t_dict)
processed_Hindi_tokens[11]
for i in range(0,len(processed_Hindi_tokens)):
    for j in range (0,len(processed_Hindi_tokens[i])):
        Str1 = (processed_Hindi_tokens[i][j])
        max_ratio = 50
        print(processed_Hindi_tokens[i][j])
        for k in range(0,len(t_dict)):
            Str2 = t_dict[k][0]
            Ratiostr1 = fuzz.ratio(Str1,Str2)
            if (Ratiostr1 > max_ratio):
                max_ratio = Ratiostr1
                processed_Hindi_tokens[i][j] = t_dict[k][1]
processed_Hindi_tokens[11]
print((hindi_comments[11])) 
processed_Hindi_tokens[12]
(processed_tokens[12])
print((hindi_comments[12])) 




def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True