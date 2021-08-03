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

with open('outputs/input_texts.pickle', 'rb') as handle:
    input_texts = pickle.load(handle)
    
with open('outputs/target_texts.pickle', 'rb') as handle:
    output_texts = pickle.load(handle)
    
with open('outputs/target_dbscan.pickle', 'rb') as handle:
    dbscan = pickle.load(handle)

# Fitting tokenizer on all short messages
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(input_texts)
total_words = len(tokenizer.word_index) + 1

with open('model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

input_tokens = tokenizer.texts_to_sequences(input_texts)

predictors, max_sequence_len = generate_padded_sequences(input_tokens)
labels = np.array(dbscan.labels_, copy=True)
labels[labels == -1] = len(set(dbscan.labels_)) - 1

encoder_labels = OneHotEncoder().fit(labels.reshape(-1, 1))
one_hot_labels = encoder_labels.transform(labels.reshape(-1, 1))

with open('model/encoder_labels.pickle', 'wb') as handle:
    pickle.dump(encoder_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = create_model(max_sequence_len, total_words, one_hot_labels.shape)
print(model.summary())

history = model.fit(predictors, one_hot_labels.todense(), epochs=25, verbose=1, validation_split=0.05, batch_size=512)

tf.saved_model.save(model, 'model/model')

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)
# quit()
predictions = model.predict_classes(predictors, verbose=0)
unique, counts = np.unique(predictions, return_counts=True)

def get_responses(seed_text, n):
    print("Input -", seed_text)
    print("----------------------")
    responses = list()
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predictions = model.predict(token_list, verbose=10)
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


    for i, response in enumerate(responses):
        print("Response", (i + 1), "->", response[0], " -> Score :", response[1])

get_responses("goodbye", 5)
get_responses("Thanks have a great day!", 5)