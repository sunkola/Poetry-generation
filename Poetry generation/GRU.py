import requests
import re
import random
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GRU, Dense, Activation, Dropout

with open('./poetry.txt', 'r', encoding = 'utf-8') as f:
    raw_text = f.read()
lines = raw_text.split("\n")[:-1]
poem_text = [i.split(':')[-1] for i in lines]
char_list = [re.findall('[\x80-\xff]{3}|[\w\W]', s) for s in poem_text]
all_words = []
for i in char_list:
    all_words.extend(i)
print(poem_text[0:5])
print(char_list[0:5])
print(all_words[0:5])

word_dataframe = pd.DataFrame(pd.Series(all_words).value_counts())
print(word_dataframe.iloc[0:5])
word_dataframe['id']=list(range(1,len(word_dataframe)+1))
print(word_dataframe.iloc[0:5])
word_index_dict = word_dataframe['id'].to_dict()
print(word_index_dict)

index_dict = {}
for k in word_index_dict:
    index_dict.update({word_index_dict[k]:k})
print(index_dict)

worjson = json.dumps(word_index_dict)
f = open("./wordict.json","w")
f.write(worjson)
f.close()

indjson = json.dumps(index_dict)
f = open("./inddict.json","w")
f.write(indjson)
f.close()

seq_len = 2
dataX = []
dataY = []
for i in range(0, len(all_words) - seq_len, 1):
    seq_in = all_words[i : i + seq_len]
    seq_out = all_words[i + seq_len]
    dataX.append([word_index_dict[x] for x in seq_in])
    dataY.append(word_index_dict[seq_out])
X = np.array(dataX)
# y = utils.to_categorical(np.array(dataY)) # one hot encoding use 'categorical_crossentropy'
y = np.array(dataY) # label encoding use 'sparse_categorical_crossentropy'

model = Sequential()
model.add(Embedding(len(word_dataframe)+1, 32, input_length=seq_len))
model.add(GRU(32, return_sequences=True))
model.add(Dropout(0.25))
model.add(GRU(16))
model.add(Dropout(0.25))
model.add(Dense(len(word_dataframe)+1))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam') 

model.fit(X, y, batch_size=32, epochs = 20)

model.save('./model.h5') 