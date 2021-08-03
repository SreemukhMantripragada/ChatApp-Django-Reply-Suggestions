# ChatApp-Django-Reply-Suggestions
(Check requirements.txt in each folder while cloning.)  
### Django Application
Uses ***AJAX*** for requesting the messages and sending new messages.  
Uses ***PostgreSQL*** for the database.  
Code for the Chat application at [link](https://github.com/SreemukhMantripragada/ChatApp-Django-Reply-Suggestions/tree/master/chatapp).  
 
### Reply Suggestions Model  
Uses Retrieval based approach on clustered reply messages. Given an input text, the LSTM model returns a point in the clustering space. The top neighbours are extracted and returned.   
Uses ***Tensorflow*** for LSTM model and ***DBSCAN*** for clustering and ***ANNOY*** for retrieving the nearest neighbors.  
Link for Model Development and training at [link](https://github.com/SreemukhMantripragada/ChatApp-Django-Reply-Suggestions/tree/master/Model).  

## Use in action:  
Predicted messages are shown this way.  
![image](https://user-images.githubusercontent.com/55551443/127990276-70b857fd-7aad-4028-a7f9-f34374226ca1.png)
  
  
Example conversation:  
![image](https://user-images.githubusercontent.com/55551443/127991452-1249876d-9eb0-4298-937b-c71d11c05539.png)
![image](https://user-images.githubusercontent.com/55551443/127991547-819864a0-0f73-4130-b43b-5c7fd8a719db.png)
![image](https://user-images.githubusercontent.com/55551443/127991577-fcb426a7-4998-48fc-9107-ef1ff5a3947a.png)
![image](https://user-images.githubusercontent.com/55551443/127991582-c318cb49-54f0-4cf4-b9e9-ae43f6b98f90.png)
