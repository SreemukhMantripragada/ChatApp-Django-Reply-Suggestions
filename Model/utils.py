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

# Basic Preprocessing
def process(text):
    text = text.lower().replace('\n', ' ').replace('-', ' ').replace(':', ' ').replace(',', '') \
          .replace('"', '').replace("...", ".").replace("..", ".").replace("!", ".").replace("?", "").replace(";", ".").replace(":", " ")
    text = " ".join(text.split())
    return text

# Returns padded inputs
def generate_padded_sequences(sequences):
    max_sequence_len = max([len(x) for x in sequences])
    sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))
    return sequences, max_sequence_len

def sort_and_select_10_neigh_dist(row):
    return row.nsmallest(10).iloc[10-1]

def create_model(max_sequence_len, total_words, shape):
    input_len = max_sequence_len - 1
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_len))
    model.add(tf.keras.layers.Embedding(total_words, 512, input_length=input_len))
    
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    return model