#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## https://www.researchgate.net/publication/312422191_Human_Emotion_Detection_via_Brain_Waves_Study_by_Using_Electroencephalogram_EEG
# Joy = Alpha
# Anger = Theta 
# Sadness = Delta + Theta
# Shock = Delta + Theta


# In[ ]:


import tensorflow as tf
import struct
import socket
import codecs
import time
import matplotlib.pyplot as plt
import pylab
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


# In[ ]:


def _getDecDigit(digit):
    digits = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    for x in range(len(digits)):
        if digit.lower() == digits[x]:
            return(x)
        
def hexToDec(hexNum):
    decNum = 0
    power = 0

    for digit in range(len(hexNum), 0, -1):
        try:
            decNum = decNum + 16 ** power * _getDecDigit(hexNum[digit-1])
            power += 1
        except:
            return
    return(int(decNum))

UDP_IP = "192.168.0.15"
UDP_PORT = 6082

sock = socket.socket(socket.AF_INET,  # Internet
                    socket.SOCK_DGRAM)  # UDP
sock.bind((UDP_IP, UDP_PORT))

random.seed(10)


# In[ ]:


def waves(num, name):
    count = 0
    final = []
    
    while count < num:
        count += 1
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        
        if '{}_absolute'.format(name) in str(data):
            newData = str(data).split(",")
            newData = newData[1].split("x")
            newData = newData[1:]
            out = [hexToDec(i[:-1]) for i in newData]
            NewOut = []

            for i in out:
                if i != None and i != 0:
                    NewOut += [i]

            final += [sum(NewOut)]
            
    out = []
    for i in final:
        if i < 80000:
            out += [i]
    
    return(sum(out)/len(out))


# In[ ]:


def full_cycle(cycles):
    timer = time.time()
    data = []
    small_spin = []
    alpha_list = []
    theta_list = []
    delta_list = []
    
    print("Iteration:")
    for i in range(cycles):
        for t in range(4):
            alpha = [waves(500, "alpha")]
            small_spin += alpha
            alpha_list += alpha
            
            theta = [waves(500, "theta")]
            small_spin += theta
            theta_list += theta
            
            delta = [waves(500, "delta")]
            small_spin += delta
            delta_list += delta
        data += [small_spin]
        small_spin = []
        print(i)
    X = np.array(data)
    Y = np.array([float(random.randint(0,1)) for i in range(len(X))])
    

    pylab.rcParams["figure.figsize"] = (20,10)
    pylab.title("Brainwave Visualization",size = 40)
    pylab.plot([i for i in range(len(alpha_list))],alpha_list,'-b',label='Alpha')
    pylab.plot([i for i in range(len(theta_list))],theta_list,'-r',label='Theta')
    pylab.plot([i for i in range(len(delta_list))],delta_list,'-g',label='Delta')
    pylab.legend(loc='upper right')
    pylab.show()

    
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=12, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=100, batch_size=100,  verbose=0)

    # calculate predictions
    predictions = model.predict(X)
    
    accuracy = 0
    for i in range(len(X)):
        if round(predictions[i][0]) == Y[i]:
            accuracy += 1
    print("Training accuracy:")
    print((accuracy/len(X))*100,"%\n")
    
    ################################################### TESTING ###################################################
    
    data = []
    small_spin = []
    print("Iteration:")
    for i in range(cycles):
        for t in range(4):
            small_spin += [waves(500, "alpha")]
            small_spin += [waves(500, "theta")]
            small_spin += [waves(500, "delta")]
        data += [small_spin]
        small_spin = []
        print(i)
    X = np.array(data)
    Y = np.array([float(random.randint(0,1)) for i in range(len(X))])
    
    accuracy = 0
    model.predict(X)
    for i in range(len(X)):
        if round(predictions[i][0]) == Y[i]:
            accuracy += 1
    print("\nTesting accuracy: ")
    print((accuracy/len(X))*100,"%")
    print("\nTime:",time.time()-timer)


# In[ ]:


full_cycle(8)

