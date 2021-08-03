# ChatApp-Django-Reply-Suggestions

### Django Application
Uses ***AJAX*** for requesting the messages and sending new messages.  
Uses ***PostgreSQL*** for the database.  
Code for the Chat application at [link](https://github.com/SreemukhMantripragada/ChatApp-Django-Reply-Suggestions/tree/master/chatapp).  
 
### Reply Suggestions Model  
Uses Retrieval based approach on clustered reply messages. Given an input text, the LSTM model returns a point in the clustering space. The top neighbours are extracted and returned.   
Uses ***Tensorflow*** for LSTM model and ***DBSCAN*** for clustering and ***ANNOY*** for retrieving the nearest neighbors.  
Link for Model Development and training at [link](https://github.com/SreemukhMantripragada/ChatApp-Django-Reply-Suggestions/tree/master/Model).  
