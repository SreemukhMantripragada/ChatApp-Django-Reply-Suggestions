import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import pandas as pd
from utils import generate_padded_sequences, create_model
with open('outputs/target_texts.pickle', 'rb') as handle:
    output_texts = pickle.load(handle)
    
with open('outputs/target_dbscan.pickle', 'rb') as handle:
    dbscan = pickle.load(handle)

with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
max_sequence_len = 50
total_words = len(tokenizer.word_index) + 1

model = tf.saved_model.load('model/model')

def get_responses(seed_text, n):
    print("Input -", seed_text)
    print("----------------------")
    responses = list()
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    token_list = token_list.astype('float32')
    predictions = model(token_list).numpy()
    predicted_indices = predictions.argsort()[0][::-1][:n]
    
    for predicted_index in predicted_indices:
        score = 0
        if predicted_index == len(set(dbscan.labels_)) - 1:
            print("Predicting outside clusters")
            predicted_index = -1
        else:
            score = predictions[0][predicted_index]
            
        # randomly pick 1 index
        possible_response = np.where(dbscan.labels_==predicted_index)[0]
        response_index = random.sample(possible_response.tolist(), 1)[0]
        # print(response_index)
        responses.append([output_texts[response_index].replace("\t", "").replace("\n", ""), score])


    # for i, response in enumerate(responses):
    #     print("Response", (i + 1), "->", response[0], " -> Score :", response[1])
    return responses

get_responses("goodbye", 5)
get_responses("Thanks have a great day!", 5)