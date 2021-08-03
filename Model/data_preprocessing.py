import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import string
import re
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from scipy import spatial

from utils import process, generate_padded_sequences

if not os.path.isdir('outputs'):
    os.mkdir('outputs')

# Reading the data from .csv 
data = pd.read_csv('data/topical_chat.csv', index_col=False)
print(data.head())
print(data.shape)

# Remove the duplicate rows
data.drop_duplicates(subset=['conversation_id', 'message'], inplace=True)

data.message = data.message.apply(process)

# Vectorize the data.
input_texts = []
target_texts = []
target_dic = {}
for conversation_index in tqdm(range(data.shape[0])):
    
    if conversation_index == 0:
        continue
        
    input_text = data.iloc[conversation_index - 1]
    target_text = data.iloc[conversation_index]
    if input_text.conversation_id == target_text.conversation_id:
        
        input_text = input_text.message
        target_text = target_text.message
        
        if len(input_text.split()) > 2 and \
            len(target_text.split()) > 0 and \
            len(input_text.split()) < 50 and \
            len(target_text.split()) < 5 and \
            input_text and \
            target_text:
                
            input_texts.append(input_text)
            target_texts.append(target_text.lower())

# Saving the texts
with open('outputs/input_texts.pickle', 'wb') as handle:
    pickle.dump(input_texts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#saving
with open('outputs/target_texts.pickle', 'wb') as handle:
    pickle.dump(target_texts, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(len(target_texts))


# Tokenizing data
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts) 
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts) 
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

input_total_words = len(input_tokenizer.word_index) + 1
input_sequences, input_max_sequence_len = generate_padded_sequences(input_sequences)

from annoy import AnnoyIndex
import random

inputAnnoyIndex = AnnoyIndex(input_max_sequence_len, 'angular')  # Length of item vector that will be indexed
for i, row in enumerate(input_sequences):
    inputAnnoyIndex.add_item(i, row)

inputAnnoyIndex.build(100) # 100 trees
inputAnnoyIndex.save('outputs/input_annoy.ann')

query_index = 7
neighbor_index, distances = inputAnnoyIndex.get_nns_by_item(query_index, 10, include_distances=True)

print(input_texts[query_index])
print("\n-------------Nearest Neighbours-------------\n")
for i in range(len(neighbor_index)):
    print(input_texts[neighbor_index[i]], "--------- neighbor_index -", neighbor_index[i], ", distance -", distances[i])



# generating similarity matrix for input texts
with open('outputs/input_similarity_matrix.txt', 'w') as input_similarity_matrix_file:
    for i in tqdm(range(len(input_texts))):
        neighbor_index, distances = inputAnnoyIndex.get_nns_by_item(i, len(input_texts), include_distances=True)

        input_similarity_row = [-1] * len(input_texts)
        for index in range(len(neighbor_index)):
            j = neighbor_index[index]
            input_similarity_row[j] = distances[index]
        
        
        input_similarity_matrix_file.write(str(input_similarity_row)[1:-1])
        if i != len(input_texts) - 1:
            input_similarity_matrix_file.write("\n")
            

# input_similarity_matrix_file.flush()
input_similarity_matrix_file.close()

target_total_words = len(target_tokenizer.word_index) + 1
target_sequences, target_max_sequence_len = generate_padded_sequences(target_sequences)

targetAnnoyIndex = AnnoyIndex(target_max_sequence_len, 'angular')  # Length of item vector that will be indexed
for i, row in enumerate(target_sequences):
    targetAnnoyIndex.add_item(i, row)

targetAnnoyIndex.build(100) # 100 trees
targetAnnoyIndex.save('outputs/target_annoy.ann')

query_index = 7
neighbor_index, distances = targetAnnoyIndex.get_nns_by_item(query_index, 10, include_distances=True)

print(target_texts[query_index])
print("\n-------------Nearest Neighbours-------------\n")
for i in range(len(neighbor_index)):
    print(target_texts[neighbor_index[i]], "--------- neighbor_index -", neighbor_index[i], ", distance -", distances[i])

# generating similarity matrix for target
with open('outputs/target_similarity_matrix.txt', 'w') as target_similarity_matrix_file:
    for i in tqdm(range(len(target_texts))):
        neighbor_index, distances = targetAnnoyIndex.get_nns_by_item(i, len(target_texts), include_distances=True)

        target_similarity_row = [-1] * len(target_texts)
        for index in range(len(neighbor_index)):
            j = neighbor_index[index]
            target_similarity_row[j] = distances[index]
        
        st = str(target_similarity_row)[1:-1]
        target_similarity_matrix_file.write(st)
        if i != len(target_texts) - 1:
            target_similarity_matrix_file.write("\n")

target_similarity_matrix_file.close()