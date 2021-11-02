#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle
import tensorflow as tf
import joblib
import cv2
import gc
from random import choice, sample
import json


# ## Function building

# In[35]:


# 將DataFrame之index重新排序
def df_randomization(df):
    new_index = [x for x in range(df.shape[0])]
    shuffle(new_index)
    df_new = df.iloc[new_index,]
    df_new.reset_index(drop=True,inplace=True)
    return df_new

# padding the image to a square image
def image_padding_black(img):
    (h, w, _) = img.shape
    img = img/img.max()
    if h > w:
        side_R = np.zeros((h, (h-w)//2, 3))
        side_L = np.zeros((h, h - w - (h-w)//2, 3))
        return np.hstack((side_R, img, side_L))
    elif h < w:
        side_U = np.zeros(((w-h)//2, w, 3))
        side_D = np.zeros((w-h-(w-h)//2, w, 3))
        return np.vstack((side_U, img, side_D))
    else:
        return img
    
def image_augmentation(image, angle, flip, scale, reshape, padding):  
    # reshape == "False", "True"
    # padding == "black", "background", "complementary_backgroud" 
    
    import cv2
    
    
    # flip
    #print(flip)
    if flip == -9:
        image = image
    else:
        image = cv2.flip(image, flip)


    
    # reshape or not
    image_reshaped = cv2.resize(image, (max(image.shape), max(image.shape)))  # as background
    image_black_padded = image_padding_black(image)
    
    if reshape == "True" or True:
        img = image_reshaped.copy()
    elif reshape == "False" or False:
        img = image_black_padded.copy()
    else:
        print("reshape_error")
    
    # rotation
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.warpAffine(img, M, (w, h),borderValue=(0,0,0))
        
    # background
    
    img_tmp = img.copy()
    img_tmp[img_tmp == 0] = -1
    img_tmp[img_tmp > 0] = 0
    img_tmp = np.abs(img_tmp)
        
    if padding == "black":
        return img
    
    elif padding == "background":
        return img + image_reshaped * img_tmp
        
    elif padding == "complementary_backgroud":
        return img + (1-image_reshaped) * img_tmp
    else:
        print("Padding_error")


# read the information from dataframe to create X (image) and Y (label)
def DataFrameToDatabase(df, img_folder, img_size):
    df.reset_index(drop=True,inplace=True)
    path = os.getcwd() + "/" + img_folder + "/"
    X = list()
    Y = list()
    
    
    filename = list()
    classes = list()
    
    
    for i in range(df.shape[0]):
        print(i)
        img_tmp = cv2.imread(path + df.filename[i])
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
        img_tmp = img_tmp/255
        
        img_tmp = image_augmentation(img_tmp, df.angle[i], df.flip[i], df.scale[i], df.reshape[i], df.padding[i])
        img_tmp = cv2.resize(img_tmp, img_size)
        X.append(img_tmp)
        Y.append(df.label[i])
        
        
        filename.append(df.filename[i])
        classes.append(df.classes[i])
        
    return X, Y, filename, classes

def LabelToVector(df, num_class): # df is a list
    df_new = np.zeros((len(df), num_class))
    for i in range(len(df)):
        #print(i, df[i])
        df_new[i, df[i]] = 1
    return df_new

def VectorToLabel(df):
    df_new = list()
    for i in range(df.shape[0]):
        L = df[i, :]
        df_new.append([np.where(L == 1)[0][0]])

    return [x[0] for x in df_new]

def LabelToClasses(label, Dict_LabelToClass):   # vector is the one-hot vector of the prediction
    # Convert the result to the label
    classes = list()
    for i in range(len(label)):
        #classes.append(Dict_LabelToClass[str(label[i])])
        classes.append(Dict_LabelToClass[label[i]])
    
    return classes

def model_acc(model, x, y):
    p = model.predict(x)
    #print(np.sum(p, axis = 1))
    p = (p == p.max(axis=1, keepdims=1)).astype(float)
    error = np.abs(p -y)
    error_sum = np.sum(error, axis = 1)
    error_sum[error_sum > 0] = 1 
    error_sum
    acc = 1 - sum(error_sum)/error_sum.shape[0]
    return acc

def ParameterExpansion(df, num_of_expansion = 5): # df is a DataFrame, contaioning one row
    df.reset_index(drop=True,inplace=True)


    columns_new = ["filename", "classes", "label","flip", "scale", "angle", "reshape", "padding"]

    df_new = pd.DataFrame(np.zeros((num_of_expansion+1, 8)), columns = columns_new)

    for r in columns_new[:3]:
        for c in range(num_of_expansion + 1):
            df_new[r][c] = df[r][0]

    index_of_expansion = sample(list(df_app.index), num_of_expansion)
    #print(index_of_expansion)
    for r in columns_new[3:]:
        df_new[r][0] = df_raw[r][0]
    
        for j in range(len(index_of_expansion)):
            df_new[r][j+1] = df_app[r][index_of_expansion[j]]
    
    return df_new


# ### Analyzation of the training image

# In[3]:


homepath = os.getcwd()
L_train = os.listdir(homepath + "\\training_images\\")
Size_train = []
for i in L_train:
    img = cv2.imread(homepath + "\\training_images\\" + i)
    Size_train.append(img.shape)
Size_train = set(Size_train)
print("There are %d training images and %d kinds of image size." %(len(L_train), len(Size_train)))
print("The dimension of the largest image is %s and the dimesion of the smallest image is %s." %(max(Size_train), min(Size_train)))


# In[4]:


homepath = "E:\\Jupyter\\2021VRDL_HW1\\db_112\\"
image_size = (112, 112)


# ### Read classes

# In[5]:


df_tmp= pd.read_csv("classes.txt", header = None)
Dict_ClassToLabel = dict()
Dict_LabelToClass = dict()
for i in df_tmp.iloc[:,0]:
    Dict_ClassToLabel[i] = int(i[:3])-1
    Dict_LabelToClass[int(i[:3])-1] = i

print("Label : Classes")
for i in range(5):
    print(list(Dict_ClassToLabel.keys())[i], ":", Dict_ClassToLabel[list(Dict_ClassToLabel.keys())[i]])

print("Classes : Label")
for i in range(5):
    print(list(Dict_LabelToClass.keys())[i], ":", Dict_LabelToClass[list(Dict_LabelToClass.keys())[i]])

    
# Save as json file
#homepath = os.getcwd()
with open(homepath + "Dict_ClassToLabel.json", "w") as file:
    json.dump(Dict_ClassToLabel, file)

with open(homepath + "Dict_LabelToClass.json", "w") as file:
    json.dump(Dict_LabelToClass, file)


# ## Data augmentation
# ### step 1: create DataFrame for future image augmentation

# In[6]:


df_all= pd.read_csv("training_labels.txt", sep = " ", header = None)
df_all.columns = ["filename", "classes"]
df_all["label"] = -9
for i in range(df_all.shape[0]):
    df_all.label[i] = Dict_ClassToLabel[df_all.classes[i]]


# In[7]:


#df_app = pd.DataFrame(columns = ["flip", "scale", "angle", "reshape", "padding"])


flip = [-9, 1] # -9: no flip, 1: horizontal flip
scale = [0.9, 1, 1.1, 1.2]
angle =  [-20, -10, 0, 10, 20]             
reshape = ["True", "False"]
padding = ["background"]



df_raw = {
    "flip":  [-9],
    "scale": [1],
    "angle": [0],
    "reshape": ["True"],
    "padding": ["background"]
}
df_raw = pd.DataFrame(df_raw)


L_columns = ["flip", "scale", "angle", "reshape", "padding"]

df_app =dict()
for i in L_columns:
    df_app[i] = list()
    
for f in flip:
    for s in scale:
        for a in angle:
            for r in reshape:
                for p in padding:
                    df_app["angle"].append(a)
                    df_app["flip"].append(f)
                    df_app["scale"].append(s)
                    df_app["reshape"].append(r)
                    df_app["padding"].append(p)
df_app = pd.DataFrame(df_app, columns = L_columns)

# 取得df_app中，與df_raw相同的index，用以去除之
i_same = df_app.loc[(df_app['flip'] == df_raw["flip"][0]) 
           & (df_app['angle'] == df_raw["angle"][0]) 
           & (df_app['scale'] == df_raw["scale"][0]) 
           & (df_app['reshape'] == df_raw["reshape"][0]) 
           & (df_app['padding'] == df_raw["padding"][0])].index[0]
print(i_same)
df_app.drop([i_same], inplace = True)
df_app


# In[8]:


# Expand each image to (num_of_expansion + 1) times

num_of_expansion = 8

df_training = pd.DataFrame(columns = ["filename", "classes", "label","flip", "scale", "angle", "reshape", "padding"])
df_val = pd.DataFrame(columns = ["filename", "classes", "label","flip", "scale", "angle", "reshape", "padding"])



for i in range(200):
    df_tmp = df_all[df_all.label == i]
    df_tmp.reset_index(drop=True,inplace=True)
        
    for t in range(13):
        df_training = df_training.append(ParameterExpansion(df_tmp.iloc[t:t+1], num_of_expansion = num_of_expansion))
        
    for t in range(13,15):
        df_val_single = {
            "filename": [df_tmp.filename[t]],
            "classes": [df_tmp.classes[t]],
            "label" : [df_tmp.label[t]],
            "flip":  [-9],
            "scale": [1],
            "angle": [0],
            "reshape": ["True"],
            "padding": ["background"]
        }
        df_val_single = pd.DataFrame(df_val_single)
        df_val = df_val.append(df_val_single)
    

df_training.label = df_training.label.astype(int)
df_training.flip = df_training.flip.astype(int)

    
df_training = df_randomization(df_training)
df_val = df_randomization(df_val)



# Save DataFrame to csv
df_training.to_csv(homepath + "df_training.csv", index = None)
df_val.to_csv(homepath + "df_val.csv", index = None)


df_training


# In[9]:


df_training


# In[10]:


df_val


# ### Step 2: Image augmentation and save images and labels

# In[11]:


folder_training_image = "training_images"
folder_validation_image = "training_images"

num_class = 200


# Save training images to npy files
x_train, y_train, filename_train, classes_train = DataFrameToDatabase(df_training, folder_training_image, image_size)
y_train = [int(x) for x in y_train]
y_train = LabelToVector(y_train, num_class) # covert label to one-hot vector
np.save(homepath + "x_train.npy", x_train)
np.save(homepath + "y_train.npy", y_train)
    

for i in range(5):
    plt.imshow(x_train[i])
    plt.show
    
del x_train, y_train
gc.collect()



    
# Save val images to npy files
x_val, y_val, filename_val, classes_val = DataFrameToDatabase(df_val, folder_validation_image, image_size)
y_val = LabelToVector(y_val, num_class) # covert label to one-hot vector

np.save(homepath + "x_val.npy", x_val)
np.save(homepath + "y_val.npy", y_val)

for i in range(5):
    plt.imshow(x_val[i])
    plt.show

del x_val, y_val
gc.collect()


# ## Confirming data

# In[12]:


homepath = "E:\\Jupyter\\2021VRDL_HW1\\db_112\\"
image_size = (112, 112)


# In[13]:


x_train = np.load(homepath + "x_train.npy")
y_train = np.load(homepath + "y_train.npy")
    
x_val = np.load(homepath + "x_val.npy")
y_val = np.load(homepath + "y_val.npy")


# In[14]:


y_train_L = VectorToLabel(y_train)
y_train_L
y_train_c = LabelToClasses(y_train_L, Dict_LabelToClass)
df_training_t = pd.DataFrame(df_training.copy())
df_training_t["pred"] = y_train_c
df_training_t


# In[15]:


y_val_L = VectorToLabel(y_val)
y_val_L
y_val_c = LabelToClasses(y_val_L, Dict_LabelToClass)
df_val_t = pd.DataFrame(df_val.copy())
df_val_t["pred"] = y_val_c
df_val_t


# ## Model building and training function

# In[16]:


def model_building(model_name, trainable=False, input_shape = (64,64,3), num_classes=200, weights = None):
    #model_name = ["VGG19", "ResNet50", "ResNet152", "ResNet"101", "Inception", "InceptionResNet", "DenseNet201", "DenseNet121"]
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras import regularizers
    
    
    if model_name == "VGG16":
        from tensorflow.keras.applications import VGG16
        cnn = VGG16(include_top=False, weights=weights,
                  input_shape=input_shape)
    
    if model_name == "VGG19":
        from tensorflow.keras.applications import VGG19
        cnn = VGG19(include_top=False, weights=weights,
                  input_shape=input_shape)
    
    if model_name == "ResNet50":    
        from tensorflow.keras.applications.resnet50 import ResNet50
        cnn = ResNet50(include_top=False, weights=weights,
                  input_shape=input_shape)
        
    if model_name == "ResNet152":
        from tensorflow.keras.applications import ResNet152V2
        cnn = ResNet152V2(include_top=False, weights=weights,
                  input_shape=input_shape)
        
    if model_name == "ResNet101":
        from tensorflow.keras.applications import ResNet101V2
        cnn = ResNet101V2(include_top=False, weights=weights,
                  input_shape=input_shape)
    
    if model_name == "Inception":
        from tensorflow.keras.applications import InceptionV3
        cnn = InceptionV3(include_top=False, weights=weights,
                  input_shape=input_shape)
    
    if model_name == "InceptionResNet":
        from tensorflow.keras.applications import InceptionResNetV2
        cnn = InceptionResNetV2(include_top=False, weights=weights,
                  input_shape=input_shape)
        
    if model_name == "DenseNet201":
        from tensorflow.keras.applications import DenseNet201
        cnn = DenseNet201(include_top=False, weights=weights,
                  input_shape=input_shape)
        
    if model_name == "DenseNet121":
        from tensorflow.keras.applications import DenseNet121
        cnn = DenseNet121(include_top=False, weights=weights,
                  input_shape=input_shape)
    
    
    
    cnn.trainable = trainable
    x = cnn.layers[-1].output
    x = BatchNormalization()(x)
    x = Flatten()(x)
    
    #x = Dropout(0.5)(x)
#    x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), bias_regularizer=l2(0.001),activity_regularizer=regularizers.l2(0.001))(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001), bias_regularizer=l2(0.001))(x)
    #x = Dense(512, activation='relu', bias_regularizer=l2(0.1))(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001), bias_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    #x = Dense(256, activation='relu', bias_regularizer=l2(0.1))(x)
    #x = Dense(256, activation='relu', bias_regularizer=l2(0.1))(x)
    x = Dense(num_classes, activation='softmax')(x)
    """
    
    cnn.trainable = trainable
    x = cnn.layers[-1].output
    x = BatchNormalization()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', bias_regularizer=l2(0.1))(x)
    x = Dense(num_classes, activation='softmax')(x)
    """
    
    # Create my own model     
    model = tf.keras.models.Model(inputs=cnn.input, outputs=x) 
    
        
    # Hyperparameters
    #opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
    opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    acc = tf.keras.metrics.CategoricalAccuracy()
    #mae = tf.keras.metrics.MeanAbsoluteError()

    model.compile(
        optimizer = opt,
        loss = tf.keras.losses.CategoricalCrossentropy(),
        #metrics = [acc, mae]
        metrics = [acc]
    )
    
    print(model.summary())
    return model


def model_training(model, model_name, homepath, epochs=10, batch_size=64):
    
    x_train = np.load(homepath + "x_train.npy")
    y_train = np.load(homepath + "y_train.npy")
    
    x_val = np.load(homepath + "x_val.npy")
    y_val = np.load(homepath + "y_val.npy")
    print(x_val.shape, y_val.shape)

    acc_train = list()
    acc_val = list()
    loss_train = list()
    loss_val = list()

               
    training_history = model.fit(x_train, y_train, validation_data = (x_val, y_val),epochs = epochs, batch_size = batch_size)
        
    acc_train = training_history.history["categorical_accuracy"]
    acc_val = training_history.history["val_categorical_accuracy"]
    loss_train = training_history.history["loss"]
    loss_val = training_history.history["val_loss"]
        
    
            
    Evaluation = {
        "acc_train": acc_train,
        "acc_val": acc_val,
        "loss_train": loss_train,
        "loss_val": loss_val
    }
    
    np.save(homepath + "Evaluation_"+model_name+".npy", Evaluation)
        
    model.save(homepath + "model_"+ model_name)
    model.save(homepath + "model")
    return model


# ## Model buidling and training

# In[17]:


homepath = "E:\\Jupyter\\2021VRDL_HW1\\db_112\\"
image_size = (112, 112)
input_shape = (112,112,3)


# In[19]:


#model_name = ["VGG16", VGG19", "ResNet50", "ResNet152", "ResNet101", "Inception", "InceptionResNet", "DenseNet201", "DenseNet121"]

model_name = "DenseNet121"
trainable = True

model = model_building(model_name= model_name, trainable=trainable, input_shape = input_shape, num_classes=200, weights = "imagenet")
model = model_training(model=model, model_name=model_name, homepath = homepath, epochs=10, batch_size=64)


# ## Evaluation

# In[20]:


evaluation = np.load(homepath + "Evaluation_"+model_name+".npy", allow_pickle=True)
evaluation = dict(enumerate(evaluation.flatten(), 1))[1]

loss_train = evaluation["loss_train"]
loss_val = evaluation["loss_val"]
acc_train = evaluation["acc_train"]
acc_val = evaluation["acc_val"]


# In[21]:


# Loss of ResNet50
plt.figure(dpi = 300, figsize = (10, 5))
ax = plt.subplot()
spot1 = ax.plot([x for x in range(1, len(loss_train)+1, 1)], loss_train, 'b-', label = 'Loss of Train')
spot2 = ax.plot([x for x in range(1, len(loss_val)+1, 1)], loss_val, 'g-', label = 'Loss of Validation')
ax.legend(loc='center right') 
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('Loss.png')
plt.show


# In[22]:


# acc of ResNet50
plt.figure(dpi = 300, figsize = (10, 5))
ax = plt.subplot()
spot1 = ax.plot([x for x in range(1, len(acc_train)+1, 1)], acc_train, 'b-', label = 'Loss of Train')
spot2 = ax.plot([x for x in range(1, len(acc_val)+1, 1)], acc_val, 'g-', label = 'Loss of Validation')
ax.legend(loc='center right') 
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('Accuracy.png')
plt.show


# In[ ]:





# ## Load the trained model and train it again with whole training image (3000 images)

# ### create training data from whole training images (3000 images)

# In[23]:


#df_app = pd.DataFrame(columns = ["flip", "scale", "angle", "reshape", "padding"])


flip = [-9, 1] # -9: no flip, 1: horizontal flip
scale = [0.9, 1, 1.1, 1.2]
angle =  [-20, -10, 0, 10, 20]             
reshape = ["True", "False"]
padding = ["background"]



df_raw = {
    "flip":  [-9],
    "scale": [1],
    "angle": [0],
    "reshape": ["True"],
    "padding": ["background"]
}
df_raw = pd.DataFrame(df_raw)


L_columns = ["flip", "scale", "angle", "reshape", "padding"]

df_app =dict()
for i in L_columns:
    df_app[i] = list()
    
for f in flip:
    for s in scale:
        for a in angle:
            for r in reshape:
                for p in padding:
                    df_app["angle"].append(a)
                    df_app["flip"].append(f)
                    df_app["scale"].append(s)
                    df_app["reshape"].append(r)
                    df_app["padding"].append(p)
df_app = pd.DataFrame(df_app, columns = L_columns)

# 取得df_app中，與df_raw相同的index，用以去除之
i_same = df_app.loc[(df_app['flip'] == df_raw["flip"][0]) 
           & (df_app['angle'] == df_raw["angle"][0]) 
           & (df_app['scale'] == df_raw["scale"][0]) 
           & (df_app['reshape'] == df_raw["reshape"][0]) 
           & (df_app['padding'] == df_raw["padding"][0])].index[0]
print(i_same)
df_app.drop([i_same], inplace = True)
df_app


# In[24]:


homepath = "E:\\Jupyter\\2021VRDL_HW1\\db_112\\"
image_size = (112, 112)
input_shape = (112,112,3)


# In[28]:


df_training = pd.read_csv(homepath + "df_training.csv")
df_val_raw = pd.read_csv(homepath + "df_val.csv")

df_val = pd.DataFrame(columns = df_val_raw.columns)
num_of_expansion = 8

for i in range(df_val_raw.shape[0]):
    df_val = df_val.append(ParameterExpansion(df_val_raw.iloc[i:i+1], num_of_expansion = num_of_expansion))
    
df_training_all = df_val.append(df_training)

df_training_all.label = df_training_all.label.astype(int)
df_training_all.flip = df_training_all.flip.astype(int)

df_training_all = df_randomization(df_training_all)

df_training_all


# In[39]:


folder_training_image = "training_images"
num_class = 200

# Save training images to npy files
x_train_all, y_train_all, filename_train, classes_train = DataFrameToDatabase(df_training_all, folder_training_image, image_size)
y_train_all = [int(x) for x in y_train_all]
y_train_all = LabelToVector(y_train_all, num_class) # covert label to one-hot vector
np.save(homepath + "x_train_all.npy", x_train_all)
np.save(homepath + "y_train_all.npy", y_train_all)
    


# ### Retraining model

# In[40]:


x_train_all = np.load(homepath + "x_train_all.npy")
y_train_all = np.load(homepath + "y_train_all.npy")

filename = "model_DenseNet121"

epochs=10
batch_size=64

# Read model
model  =  tf.keras.models.load_model(homepath + filename)

training_history = model.fit(x_train_all, y_train_all, epochs = epochs, batch_size = batch_size)

model.save(homepath + "model")


# In[ ]:




