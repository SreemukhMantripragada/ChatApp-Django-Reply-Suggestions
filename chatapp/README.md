## Project structure  
There is one django app called chat that contains the entire workflow of the website.  
There are two webpages home and chatroom. Once a user enters a room name and his username, if the given room exists those messages are loaded, otherwise a new room is created.
There are two models, one for storing each message(Time, who sent, which room, message contents) and room(Room id, room name). The room id acts as a foreign key in the room table.  
DBMS used is PostgreSQL(change database details in settings.py file to use with others).  

Front end HTML files use AJAX to query the new messages at a regular interval by passing in the room name, username and the id of last message(primary key of message table). All the messages which are for the same room and sent after the specified ID are returned.  
