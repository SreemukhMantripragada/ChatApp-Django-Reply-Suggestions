## Data  
Text conversations of various topics from [link](https://www.kaggle.com/arnavsharmaas/chatbot-dataset-topical-chat).  

## Model and Approach
It is a retrival based approach for reply suggestions based on the messages recived. Top k closest neighbors to the predicted point are retrieved based on the clusters that the messages belong to.  
LSTM based neural network takes in the input messages and returns a n-dimensional output which maps into a embedding space. Closest points are the most probable replies.  

Hyperparameters include, the epsilon value for clustering using DBSCAN, maximum length of reply messages, model size, LSTM hidden units and others.  

## After cloning the repo locally, the following files must be run in order.
data_preprocessing.py  
clustering.py  
model.py  
inference.py to check the outputs using saved model.  
(Copy saved model, target_dbscan.pickle, target_texts.pickle, tokenizer.pickle into modelfiles directory of the django App for use in website).
