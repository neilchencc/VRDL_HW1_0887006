#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import json

def DataFrameToDatabase(df, img_folder, img_size):
    df.reset_index(drop=True,inplace=True)
    path = os.getcwd() + "/" + img_folder + "/"
    X = list()
    for i in range(df.shape[0]):
        #print(i)
        img_tmp = cv2.imread(path + df.filename[i])
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
        img_tmp = img_tmp/255
        img_tmp = cv2.resize(img_tmp, img_size)
        #plt.imshow(img_tmp)
        #plt.show()
        X.append(img_tmp)
    return np.array(X)

def prediction(model, x):
    p = model.predict(x)
    p = (p == p.max(axis=1, keepdims=1)).astype(float)
    return p

def VectorToLabel(df):
    df_new = list()
    for i in range(df.shape[0]):
        L = df[i, :]
        df_new.append([np.where(L == 1)[0][0]])
#        [x[0] for x in df_new]
    return [x[0] for x in df_new]

def LabelToClasses(label, Dict_LabelToClass):   # vector is the one-hot vector of the prediction
    # Convert the result to the label
    classes = list()
    for i in range(len(label)):
        classes.append(Dict_LabelToClass[str(label[i])])
    
    return classes


# In[2]:


with open("Dict_LabelToClass.json", "r") as fp:
    Dict_LabelToClass = json.load(fp)
#Dict_LabelToClass


# In[ ]:





# ## Predict training image

# In[3]:


df_train= pd.read_csv("training_labels.txt", sep = " ", header = None)
df_train.columns = ["filename", "classes"]
df_train["label"] = -9
df_train["pred"] = -9
print(df_train.head())


# load training image
img_folder = "training_images"
x_train = DataFrameToDatabase(df_train, img_folder = img_folder, img_size = (112,112))


# In[4]:


filename = "model"

# Read model
model  =  tf.keras.models.load_model(filename)

# Prediction
pred_train = prediction(model, x_train)

# Convert the result to the label
df_train.label = VectorToLabel(pred_train)
df_train.pred = LabelToClasses(df_train.label, Dict_LabelToClass)


# Save the result to the answer.txt
df_train.to_csv("answer_train.txt", header = None, index = None, sep = " ")


# In[5]:


df_train


# In[10]:


df_train[df_train.classess == df_train.pred]


# ## Predict test images

# In[3]:


df_test= pd.read_csv("testing_img_order.txt", sep = " ", header = None)
df_test.columns = ["filename"]
df_test["label"] = -9
df_test["classes"] = -9
print(df_test.head())


# load testing image
img_folder = "testing_images"
x_test = DataFrameToDatabase(df_test, img_folder = "testing_images", img_size = (112,112))


# In[4]:


filename = "model"

# Read model
model  =  tf.keras.models.load_model(filename)

# Prediction
pred_test = prediction(model, x_test)

# Convert the result to the label
df_test.label = VectorToLabel(pred_test)
df_test.classes = LabelToClasses(df_test.label, Dict_LabelToClass)
df_test = df_test[['filename', 'classes']]

# Save the result to the answer.txt
df_test.to_csv("answer.txt", header = None, index = None, sep = " ")


# In[ ]:




