import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()


from nltk import word_tokenize,sent_tokenize
import numpy as np
import tflearn
import tensorflow
import random 
import json
import logging
import pickle
logging.getLogger('tensorflow').setLevel(logging.ERROR)
with open("intents.json") as file:
  data=json.load(file)
                                                                     

try:
    with open("data.pickle","rb") as f:
        words, labels,training, output = pickle.load(f)
except:
    words=[]
    docs_x=[]
    docs_y=[]
    labels=[]
    for val in data["intents"]:           #to loop through all dictionaries
        for pattern in val["patterns"]:
            wd = nltk.word_tokenize(pattern)
            words.extend(wd)
            docs_x.append(wd)
            docs_y.append(val["tag"])
        if val["tag"] not in labels:
            labels.append(val["tag"])
    words=[stemmer.stem(w.lower()) for w in words if w!= "?"]      #to check how many words it has seen
    words=sorted(list(set(words)))          # set so that their are no duplicates its a data typpe in its own way

    labels=sorted(labels)
    training =[]
    output =[]
    out_empty=[0 for _ in range(len(labels))]
    for x,doc in enumerate(docs_x):
       bag=[]
   
       wd = [stemmer.stem(w) for w in doc]                  
    
       for w in words:
         if w in wd:
            bag.append(1)                          #if word is present than give 1 otherwise 0
         else:
            bag.append(0)
       output_row=out_empty[:]
       output_row[labels.index(docs_y[x])]=1
   
       training.append(bag)
       output.append(output_row) 
    training=np.array(training)
    output=np.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels,training, output),f)
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)          #add a fully connected neural network input layer to our data of 8 neurons
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax") #output layer
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:    
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model.tflearn")
def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower())for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w==se:
                bag[i] = (1)
        return np.array(bag)
def chat():
    print("lets start talking yap1006 bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower()=="quit":
            break
        results = model.predict([bag_of_words(inp,words)])
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag']==tag:
                   responses = tg['responses']
        print(random.choice(responses))           
        
chat()                
        