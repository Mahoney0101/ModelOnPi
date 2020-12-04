#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
model = keras.models.load_model('/home/pi/Downloads/CNNModel.h5')


# In[2]:


from numpy import load
wheezes = load('/home/pi/Downloads/wheezedata.npy',allow_pickle=True)
both = load('/home/pi/Downloads/bothdata.npy',allow_pickle=True)
crackles = load('/home/pi/Downloads/cracklesdata.npy',allow_pickle=True)
none = load('/home/pi/Downloads/data.npy',allow_pickle=True)


# In[3]:


model.compile()


# In[4]:


import numpy as np


# In[9]:


clip=[]
clip.append(none[1][0].reshape(50,245,1))
clip.append(wheezes[1][0].reshape(50,245,1))
clip.append(wheezes[2][0].reshape(50,245,1))
clip.append(wheezes[2][0].reshape(50,245,1))
clip.append(both[8][0].reshape(50,245,1))
clip.append(both[4][0].reshape(50,245,1))
clip.append(none[5][0].reshape(50,245,1))
clip.append(crackles[8][0].reshape(50,245,1))
clip.append(crackles[4][0].reshape(50,245,1))
clip.append(crackles[5][0].reshape(50,245,1))
clip = np.array(clip)
prediction = model.predict(clip)


# In[10]:


classes = np.argmax(prediction, axis = 1)
print(classes)

