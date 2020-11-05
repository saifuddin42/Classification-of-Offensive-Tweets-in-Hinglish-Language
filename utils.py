from collections import Counter

def read_data(f):
	with open(f,encoding='UTF-8') as inp:
		lines = inp.readlines()
	data = []
	for line in lines:
		line = line.strip().split()
		sentence = []
		for token in line:
			token = token.split('|')
			word = token[0]
			tag = token[1]
			sentence.append((word,tag))
		data.append(sentence)
	return data





def convert_data_for_training(data):
	#for d in data:
	#	tokens = [t[0] for t in d]
	#	tags = [t[1] for t in d]
	return [([t[0] for t in d],[t[1] for t in d]) for d in data]

def substitute_with_UNK(data, n=1):
    token_frequency = {}
    for sent in data:
        for word in sent:
            if word not in token_frequency:
                token_frequency[word] = 1
            else:
                token_frequency[word]+=1
    print( )
	# Iterate through the corpus and substitute the rare words with UNK
    alist=list(data)
    for r in range(len(data)):
        for j in range(len(data[r])):
            if data[r][j] in token_frequency:
                if (token_frequency[data[r][j]])==1:
                       # print(str(r),str(j),str(k), alist[r][j][k])
                    alist[r][j] = "UNK"
	# Return data
    return data


def substitute_with_UNK_for_TEST(test_data,word_to_ix):
    word_to_ix1 = {}
    ix_to_word1 = {}
    tag_to_ix1 = {}
    ix_to_tag1 = {}
    # converting the words to UNK which are not in vocabulory
    for sent1, tags1 in test_data:
	    for word1 in sent1:
		    if word1 not in word_to_ix1:
                
			    word_to_ix1[word1] = len(word_to_ix1)
			    ix_to_word1[word_to_ix1[word1]] = word1
	    for tag1 in tags1:
		    if tag1 not in tag_to_ix1:
			    tag_to_ix1[tag1] = len(tag_to_ix1)
			    ix_to_tag1[tag_to_ix1[tag1]] = tag1

    Tlist=list(test_data)
    for r in range(len(test_data)):
        for j in range(len(test_data[r])):
            for k in range (len(test_data[r][j])):
                if (test_data[r][j][k] not in word_to_ix) :
                    if(test_data[r][j][k] in word_to_ix1):
                        Tlist[r][j][k] = "UNK"
    return (test_data)