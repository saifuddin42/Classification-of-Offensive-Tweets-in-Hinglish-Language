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
import torch.optim as optim
import utils

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





f = "english/agr_en_train.csv"
 
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


Hindi_text  = "agr_hi_fb_test.csv"
df1 = pd.read_csv(Hindi_text,names = ['source','comment','annotation'],encoding='UTF-8')
df1['comment'] = df1.comment.str.strip()   # removing spaces
hindi_comments = np.asarray(df1['comment'])    # dividing the dataframe into comments and tags and converting to array
hindi_tags = np.asarray(df1['annotation'])
print((hindi_comments[1])) 
processed_Hindi_tokens = []
for comment in hindi_comments:
#    comment = "Also see ....hw ur RSS activist caught in Burkha .... throwing beef in d holy temples...https://www.google.co.in/amp/www.india.com/news/india/burkha-clad-rss-activist-caught-throwing-beef-at-temple-pictures-go-viral-on-facebook-593154/amp/,NAGfacebook_corpus_msr_403402,On the death of 2 jawans in LOC CROSS FIRING"
#    comment = comment.lower()   #lower casing each tweets
    Digit_REMOVAL = re.sub(r'[0-9]+', '',str(comment)) #removal of numbers 
    URL_REMOVAL = re.sub(r"http\S+", "", Digit_REMOVAL) # removal of URLS
    Emoji_removal = remove_emoji(URL_REMOVAL)
    Emoji_removal = Emoji_removal.lower()
    if (isEnglish(Emoji_removal) == True):
        Emoji_removal = re.sub(r'[^\w\s]','',Emoji_removal)# removal of punctuation and tokenizing
    processed_Hindi_tokens.append(word_tokenize(Emoji_removal))
processed_Hindi_tokens[0]
processed_Hindi_tokens[11]
processed_Hindi_tokens[6]
processed_Hindi_tokens[11]



#-----------Transliteration and translation
transliteration_dict = "B:\CS 695\Assignment3\Classification-of-Offensive-Tweets-in-Hinglish-Language/transliterations.hi-en.csv"
t_dict = pd.read_csv(transliteration_dict,names = ['Hinglish','Hindi'],encoding='UTF-8',sep='\t')
t_dict['Hinglish'] = t_dict['Hinglish'].str.strip()
t_dict['Hindi'] = t_dict['Hindi'].str.strip()
t_dict = np.asarray(t_dict)

#--------------profanity dictionary
profanity_dict = "ProfanityText.txt"
P_dict = pd.read_csv(profanity_dict,names = ['Hinglish','English'],encoding='UTF-8',sep='\t')
P_dict['Hinglish'] = P_dict['Hinglish'].str.strip()
P_dict['English'] = P_dict['English'].str.strip()
P_dict = np.asarray(P_dict)


print(t_dict)
processed_Hindi_tokens[4]
for i in range(0,len(processed_Hindi_tokens)):
    print(i)
    for j in range (0,len(processed_Hindi_tokens[i])):
        flag = 0
        Str1 = (processed_Hindi_tokens[i][j])
        max_ratio = 60
        max_ratio_P = 75   #needs to be adjusted
        if (Str1 in EH_dict): # check whether the values exists in english dictionary or not.
            continue;
        for l in range(0,len(P_dict)):
            Str2 = P_dict[l][0]
            Ratiostr1 = fuzz.ratio(Str1,Str2)
            if (Ratiostr1 >= max_ratio_P):
                print(Ratiostr1)
                max_ratio_P = Ratiostr1
                processed_Hindi_tokens[i][j] = P_dict[l][1]
                flag = 1 
                print(flag)
                break;
        for p in EH_dict_F:
            Ratiostr1 = fuzz.ratio(Str1,str(p))
            if(Ratiostr1 >= 98):
                flag = 1
                break;
        if (flag == 1):
            continue;
        
        else:
            for k in range(0,len(t_dict)):
                Str2 = t_dict[k][0]
                Ratiostr1 = fuzz.ratio(Str1,Str2)
                if (Ratiostr1 > max_ratio):
                    max_ratio = Ratiostr1
                    processed_Hindi_tokens[i][j] = t_dict[k][1]
processed_Hindi_tokens[0]
processed_Hindi_tokens[1]
(processed_tokens[12]) 



#------------------Translation of hindi text back to english-------

Hindi_dict = "Hindi_English_dict.csv"
H_dict = pd.read_csv(Hindi_dict,names = ['Hindi','English'],encoding='UTF-8')

HE_dict_F = "HE_dictionary_functions.csv"
H_dict_F = pd.read_csv(HE_dict_F,names = ['Hindi','English'],encoding='UTF-8')
H_dict_F['Hindi'] = H_dict_F['Hindi'].str.strip()
H_dict_F['English'] = H_dict_F['English'].str.strip()
H_hindi_F = np.asarray(H_dict_F['Hindi'])
H_english_F = np.asarray(H_dict_F['English'])


H_dict['Hindi'] = H_dict['Hindi'].str.strip()
H_dict['English'] = H_dict['English'].str.strip()
H_hindi = np.asarray(H_dict['Hindi'])
H_english = np.asarray(H_dict['English'])

HE_dict = dict(zip(H_hindi,H_english))
H_dict_F = dict(zip(H_hindi_F,H_english_F))

EH_dict = {v:k for k, v in HE_dict.items()}
EH_dict_F = {v:k for k, v in H_dict_F.items()}

for i in range(0,len(processed_Hindi_tokens)):
    print(i)
    for j in range (0,len(processed_Hindi_tokens[i])):
        Str = processed_Hindi_tokens[i][j]
        if(Str in HE_dict):
            processed_Hindi_tokens[i][j] = HE_dict[Str]
        elif(Str in H_dict_F):
            processed_Hindi_tokens[i][j] = H_dict_F[Str]
            
            
#------------------------------MODEL
print(processed_tokens[0])
print(tags)            
            
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.functional as F

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_tags(seq, to_ix):
    idxs = to_ix[seq]
    idxs = torch.tensor(idxs, dtype=torch.long)
    idxs = idxs.view(1)
    return idxs

def sentence_to_padded_sentence(sentence,word_to_ix):
    
    # map sentences to vocab
    sentence =  [[word_to_ix[word] for word in sent] for sent in sentence]
    # sentence now looks like:  
    # [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]
    sentence_lengths = [len(sent) for sent in sentence]
    pad_token = word_to_ix['<PAD>']
    longest_sent = max(sentence_lengths)
    batch_size = len(sentence)
    padded_sentence = np.ones((batch_size, longest_sent)) * pad_token
    for i, x_len in enumerate(sentence_lengths):
        sequence = sentence[i]
        padded_sentence[i, 0:x_len] = sequence[:x_len]
  
    return padded_sentence


word_to_ix = {}
ix_to_word = {}
tag_to_ix = {}
ix_to_tag = {}
word_to_ix = {"<PAD>":0}
for sent in training_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
			ix_to_word[word_to_ix[word]] = word
for tag in tags:
	if tag not in tag_to_ix:
		tag_to_ix[tag] = len(tag_to_ix)
		ix_to_tag[tag_to_ix[tag]] = tag

sentence= []
for sent in training_data:
     sentence.append(sent[:50])
     
test_sentence= []
for sent in testing_data:
     test_sentence.append(sent[:50])


training_data = utils.substitute_with_UNK(processed_tokens,1)
testing_data = utils.substitute_with_UNK_for_TEST(processed_tokens,word_to_ix)
print(len(training_data))


padded_sentence = sentence_to_padded_sentence(sentence, word_to_ix)
test_padded_sentence = sentence_to_padded_sentence(test_sentence, word_to_ix)

class MIMCT(nn.Module):   
    def __init__(self,input_channel,vocab_size,word_to_ix,output_channel,embedding_dim,hidden_dim,kernel_size,feature_linear):
        super(MIMCT, self).__init__()
        self.CNN_Layers = nn.Sequential( 
            nn.Conv1d(input_channel, output_channel,kernel_size[0], stride=1),
            nn.Conv1d(input_channel, output_channel, kernel_size[1], stride=1),
            nn.Conv1d(input_channel, output_channel, kernel_size[2], stride=1),
            nn.Flatten(),nn.Dropout(p=0.25),
            nn.Linear(feature_linear, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax()
            )
        
        
        #create LSTM.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim,3)
        self.dropout = nn.Dropout(p=0.20)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.linear = nn.Linear(50+1,3)
    def forward(self,x):
      #  y = self.LSTM_Layers(x)
        embeds = self.word_embeddings(x)
        
        embeds_cnn = embeds.view(1,embeds.size(0),embeds.size(1))
        cnn_output = self.CNN_Layers(embeds_cnn)
        
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        lstm_out= self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(x), -1))
        
        lstm_output = self.softmax(tag_space)
        #concat the outputs the compile layer with categorical cross-entropy the loss function,
        lstm_output = lstm_output.view(lstm_output.size(0),-1)
        cnn_output = cnn_output.view(cnn_output.size(0),-1)
        X = torch.cat((lstm_output,cnn_output))
        X = X.view(1,X.size(0),X.size(1))
        X = self.maxpool(X)
        X = self.linear(X.view(X.size(2), -1))
        X = self.softmax(X)
        print(X)
        return X



#try with output channel 1 as well.
        
    
    
    
batch_size = 1
input_channel = 50 #vocab size
vocab_size = len(word_to_ix) 
embedding_dim = 200 
output_channel = 50
kernel_size = [20,15,10]
Feature_layer1 = embedding_dim - kernel_size[0] + 1
Feature_layer2 = Feature_layer1 - kernel_size[1] + 1
Feature_layer3 = Feature_layer2 - kernel_size[2] + 1
feature_linear = Feature_layer3 * input_channel

#Parameters for LSTM
hidden_dim = 128
dropout = 0.25, 
#recurrent_dropout = 0.3

model = MIMCT(input_channel,vocab_size,word_to_ix,output_channel,embedding_dim,hidden_dim,kernel_size,feature_linear)
loss_function = nn.CrossEntropyLoss()
#Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
original_data = training_data
training_data = padded_sentence
sentence1 = training_data[0]

print(len(tags))
for epoch in range(5):  # running for 20 epoch
    print(f"Starting epoch {epoch}...")
    #for sentence in training_data:
    for index,sentence in enumerate(training_data):
        model.zero_grad()
        targets = tags[index]
        sentence_in=torch.tensor(sentence, dtype=torch.long)
        
        targets = prepare_sequence_tags(targets, tag_to_ix)

        tag_scores = model(sentence_in)
 
        loss = loss_function(tag_scores, targets)
        print(loss) 
        loss.backward()
        optimizer.step()
    print(epoch)

with open("mymodel_output_2.4.txt", 'a',encoding='UTF-8') as op:
    formatted_output = 'Helo Wrold'
    op.write(formatted_output + '\n')







testing_data = test_padded_sentence
with torch.no_grad():
	# this will be the file to write the outputs
    with open("mymodel_output.txt", 'w',encoding='UTF-8') as op:
        for instance in testing_data:
			# Convert the test sentence into a word ID tensor
            test_sentence_in=torch.tensor(instance, dtype=torch.long)

            tag_scores = model(test_sentence_in)
	
			# Find the tag with the highest probability in each position
            outputs = [int(np.argmax(ts)) for ts in tag_scores.detach().numpy()]
			# Prepare the output to be written in the same format as the test file (word|tag)
            formatted_output = ix_to_tag[outputs[0]]
			# Write the output
            op.write(formatted_output + '\n')
            
        print(i)
        print(len(test_data))














'''
m = nn.Conv1d(1, 2,1,stride=2)
input1 = torch.randn(10)
cnn1d_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
input1 = input1.unsqueeze(0).unsqueeze(0)
input1.shape
cnn1d_2(input1)



#------------CNN_Class----------------#
kernel_size = [4,3,2]
embedding_dim = 10
Feature_layer1 = embedding_dim - kernel_size[0] + 1
Feature_layer2 = Feature_layer1 - kernel_size[1] + 1
Feature_layer3 = Feature_layer2 - kernel_size[2] + 1

Layer1 = nn.Conv1d(16, 16, 4, stride=1)
Layer2 = nn.Conv1d(16, 16, 3, stride=1)
Layer3 = nn.Conv1d(16, 16, 2, stride=1)
Dropout_layer = nn.Dropout(p=0.25)
dense_layer = nn.Linear(Feature_layer3, 3)
nn.Flatten

#------------Forward------------------#
batch_size = 1
input_channel = 16
embedding_dim = 10
input1 = torch.randn(batch_size,input_channel,embedding_dim)
output = Layer1(input1)
output1 = Layer2(output)
output2 = Layer3(output1)
output3 = Dropout_layer(output2)
output4 = dense_layer(output3)
nn.flatten



x = torch.tensor([[1.0, -1.0],[0.0,  1.0],[0.0,  0.0]])

in_features = x.shape[1]  # = 2
out_features = 2

m = nn.Linear(in_features, out_features)
m.weight







class MIMCT(nn.Module):   
    def __init__(self,input_channel,output_channel,embedding_dim,kernel_size,feature_linear):
        super(MIMCT, self).__init__()
        self.CNN_Layers = nn.Sequential( nn.Conv1d(input_channel, output_channel,kernel_size[0], stride=1),nn.Conv1d(input_channel, output_channel, kernel_size[1], stride=1),nn.Conv1d(input_channel, output_channel, kernel_size[2], stride=1),nn.Flatten(),nn.Dropout(p=0.25),nn.Linear(feature_linear, 3),nn.Softmax())

    def forward(self,x):
        x = self.CNN_Layers(x)
        return x

#try with output channel 1 as well.
batch_size = 1
input_channel = 16
embedding_dim = 200
output_channel = 16
kernel_size = [20,15,10]
embedding_dim = 200
Feature_layer1 = embedding_dim - kernel_size[0] + 1
Feature_layer2 = Feature_layer1 - kernel_size[1] + 1
Feature_layer3 = Feature_layer2 - kernel_size[2] + 1
feature_linear = Feature_layer3 * input_channel
model = MIMCT(input_channel,output_channel,embedding_dim,kernel_size,feature_linear)

output = model(input1)
'''




''''
kernel_size = [20,15,10]
embedding_dim = 200
Feature_layer1 = embedding_dim - kernel_size[0] + 1
Feature_layer2 = Feature_layer1 - kernel_size[1] + 1
Feature_layer3 = Feature_layer2 - kernel_size[2] + 1
feature_linear = Feature_layer3 * input_channel
m = nn.Sequential( nn.Conv1d(input_channel,16,kernel_size[0], stride=1),nn.Conv1d(input_channel, output_channel, kernel_size[1], stride=1),nn.Conv1d(input_channel, output_channel, kernel_size[2], stride=1),nn.Flatten(),nn.Dropout(p=0.25),nn.Linear(feature_linear, 3),nn.Softmax())

#------------Forward------------------#
batch_size = 1
input_channel = 16
embedding_dim = 200
input1 = torch.randn(batch_size,input_channel,embedding_dim)
test_input = input1
output = m(test_input)
nn.flatten
'''
import pandas as pd
import numpy as np
x = pd.read_csv("B:\CS 695\Assignment3\Classification-of-Offensive-Tweets-in-Hinglish-Language/hindi_tokens_translated_to_English_list.csv",sep='/t',encoding="UTF-8")
 = np.asarray(x)

x = torch.rand(4,3)
y.size() = torch.squeeze(x,1)
sig_out=x.view(1, -1)
sig_out=sig_out[:, -1]


original_data
a = []
for sent in sentence:
    a.append(len(sent))


import matplotlib.pyplot as plt

plt.hist(a, bins =30)
plt.show()

i =0 
for value in a:
    if(int(value) < 50):
        i=i+1    