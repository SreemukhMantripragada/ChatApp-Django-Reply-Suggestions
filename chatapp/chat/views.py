from django.core.checks import messages
from django.shortcuts import redirect, render
from .models import Message, Room
from django.http import HttpResponse, JsonResponse
from django.conf import settings
# import numpy as np
# import random
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import tensorflow as tf
# Create your views here.
def home(request):
    return render(request, 'home.html')
def room(request, room):
    username = request.GET.get('username')
    room_details = Room.objects.get(name=room)
    params = {
        'username': username,
        'room': room,
        'room_details': room_details
    }
    return render(request, 'room1.html', params)
def checkview(request):
    room = request.POST['room_name']
    username = request.POST['username']
    if (not room) or (not username):
        return redirect('/')
    if Room.objects.filter(name=room).exists():
        pass
    else:
        new_room = Room.objects.create(name=room)
        new_room.save()
    return redirect('/'+room+'/?username='+username)
def send(request):
    message = request.POST['message']
    username = request.POST['username']
    room_id = request.POST['room_id']
    if message:
        new_message = Message.objects.create(value=message, user=username, room=room_id)
        new_message.save()
    return HttpResponse('sent successfully!')


def getMessages(request, room, username, lastmsgid):

    room_details = Room.objects.get(name=room)
    messages = Message.objects.raw("select  * from chat_message where room='"+str(room_details.id)+"' and id>"+lastmsgid+"")
    msgvals = []
    for msg in messages:
        msgvals.append({
            'value':msg.value,
            'date':msg.date,
            'user':msg.user,
            'id':msg.id
        })
    predictions = {}
    # if len(messages):
    #     predictions = get_responses(msgvals[-1]['value'], 4) 
    # ind=len(predictions)
    # if ind<3:
    #     predictions[ind] = 'Hello everyone!'
    #     ind += 1
    # if ind<3:
    #     predictions[ind] = 'How was your day'
    #     ind += 1
    # if ind<3:
    #     predictions[ind] = 'Thank you'
    #     ind += 1
    return JsonResponse({
        "messages": msgvals,
        "predictions": predictions
    })

# def get_responses(seed_text, n):
#     print(seed_text)
#     if(settings.LASTTEXT!=None and settings.LASTTEXT==seed_text):
#         return settings.LASTANS
#     max_sequence_len = 48
#     responses = {}
#     token_list = settings.TOKENIZER.texts_to_sequences([seed_text])[0]
#     token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
#     token_list = token_list.astype('float32')
#     predictions = settings.MODEL(token_list).numpy()
#     predicted_indices = predictions.argsort()[0][::-1][:n]
#     print(predicted_indices)
#     count=0
#     for predicted_index in predicted_indices:
#         score = 0
#         if predicted_index == len(set(settings.DBSCAN.labels_)) - 1:
#             print("Predicting outside clusters")
#             predicted_index = -1
#         else:
#             score = predictions[0][predicted_index]
            
#         # randomly pick 1 index
#         possible_response = np.where(settings.DBSCAN.labels_==predicted_index)[0]
#         response_index = random.sample(possible_response.tolist(), 1)[0]
#         # print(response_index)
#         if(score>0):
#             # responses.append([])
        
#             responses[count]=settings.OUTPUT_TEXTS[response_index].replace("\t", "").replace("\n", "")
#         # print(responses[count])
#             count+=1
#     settings.LASTTEXT = seed_text
#     settings.LASTANS=responses
#     print(responses)
#     return responses